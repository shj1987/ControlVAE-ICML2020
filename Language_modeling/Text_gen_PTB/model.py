import torch
import torch.nn as nn
from torch import Tensor
# from torch.optim.lr_scheduler import ExponentialLR
from typing import Any, Dict, Optional, Tuple, Union

import texar.torch as tx
from texar.torch.custom import MultivariateNormalDiag


def kl_divergence(means: Tensor, logvars: Tensor) -> Tensor:
    """Compute the KL divergence between Gaussian distribution
    """
    kl_cost = -0.5 * (logvars - means ** 2 -
                      torch.exp(logvars) + 1.0)
    kl_cost = torch.mean(kl_cost, 0)
    return torch.sum(kl_cost)


class VAE(nn.Module):
    _latent_z: Tensor

    def __init__(self, vocab_size: int, config_model):
        super().__init__()
        # Model architecture
        self._config = config_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_w_embedder = tx.modules.WordEmbedder(
            vocab_size=vocab_size, hparams=config_model.enc_emb_hparams)

        self.encoder = tx.modules.UnidirectionalRNNEncoder[tx.core.LSTMState](
            input_size=self.encoder_w_embedder.dim,
            hparams={
                "rnn_cell": config_model.enc_cell_hparams,
            })

        self.decoder_w_embedder = tx.modules.WordEmbedder(
            vocab_size=vocab_size, hparams=config_model.dec_emb_hparams)

        if config_model.decoder_type == "lstm":
            self.lstm_decoder = tx.modules.BasicRNNDecoder(
                input_size=(self.decoder_w_embedder.dim +
                            config_model.latent_dims),
                vocab_size=vocab_size,
                token_embedder=self._embed_fn_rnn,
                hparams={"rnn_cell": config_model.dec_cell_hparams})
            sum_state_size = self.lstm_decoder.cell.hidden_size * 2

        elif config_model.decoder_type == 'transformer':
            # position embedding
            self.decoder_p_embedder = tx.modules.SinusoidsPositionEmbedder(
                position_size=config_model.max_pos,
                hparams=config_model.dec_pos_emb_hparams)
            # decoder
            self.transformer_decoder = tx.modules.TransformerDecoder(
                # tie word embedding with output layer
                output_layer=self.decoder_w_embedder.embedding,
                token_pos_embedder=self._embed_fn_transformer,
                hparams=config_model.trans_hparams)
            sum_state_size = self._config.dec_emb_hparams["dim"]

        else:
            raise ValueError("Decoder type must be 'lstm' or 'transformer'")

        self.connector_mlp = tx.modules.MLPTransformConnector(
            config_model.latent_dims * 2,
            linear_layer_dim=self.encoder.cell.hidden_size * 2)

        self.mlp_linear_layer = nn.Linear(
            config_model.latent_dims, sum_state_size)

    def forward(self,  # type: ignore
                data_batch: tx.data.Batch,
                kl_weight: float, start_tokens: torch.LongTensor,
                end_token: int) -> Dict[str, Tensor]:
        # encoder -> connector -> decoder
        text_ids = data_batch["text_ids"].to(self.device)
        input_embed = self.encoder_w_embedder(text_ids)
        _, encoder_states = self.encoder(
            input_embed,
            sequence_length=data_batch["length"].to(self.device))

        mean_logvar = self.connector_mlp(encoder_states)
        mean, logvar = torch.chunk(mean_logvar, 2, 1)
        kl_loss = kl_divergence(mean, logvar)
        dst = MultivariateNormalDiag(
            loc=mean, scale_diag=torch.exp(0.5 * logvar))

        latent_z = dst.rsample()
        helper = None
        if self._config.decoder_type == "lstm":
            helper = self.lstm_decoder.create_helper(
                decoding_strategy="train_greedy",
                start_tokens=start_tokens,
                end_token=end_token)

        # decode
        seq_lengths = data_batch["length"].to(self.device) - 1
        outputs = self.decode(
            helper=helper, latent_z=latent_z,
            text_ids=text_ids[:, :-1], seq_lengths=seq_lengths)

        logits = outputs.logits

        # Losses & train ops
        rc_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=text_ids[:, 1:], logits=logits,
            sequence_length=seq_lengths)

        nll = rc_loss + kl_weight * kl_loss

        ret = {
            "nll": nll,
            "kl_loss": kl_loss,
            "rc_loss": rc_loss,
            "lengths": seq_lengths,
            "mu":mean,
        }
        
        return ret

    def _embed_fn_rnn(self, tokens: torch.LongTensor) -> Tensor:
        r"""Generates word embeddings
        """
        embedding = self.decoder_w_embedder(tokens)
        latent_z = self._latent_z
        if len(embedding.size()) > 2:
            latent_z = latent_z.unsqueeze(0).repeat(tokens.size(0), 1, 1)
        return torch.cat([embedding, latent_z], dim=-1)

    def _embed_fn_transformer(self,
                              tokens: torch.LongTensor,
                              positions: torch.LongTensor) -> Tensor:
        r"""Generates word embeddings combined with positional embeddings
        """
        output_p_embed = self.decoder_p_embedder(positions)
        output_w_embed = self.decoder_w_embedder(tokens)
        output_w_embed = output_w_embed * self._config.hidden_size ** 0.5
        output_embed = output_w_embed + output_p_embed
        return output_embed
        
    @property
    def decoder(self) -> tx.modules.DecoderBase:
        if self._config.decoder_type == "lstm":
            return self.lstm_decoder
        else:
            return self.transformer_decoder

    def decode(self,
               helper: Optional[tx.modules.Helper],
               latent_z: Tensor,
               text_ids: Optional[torch.LongTensor] = None,
               seq_lengths: Optional[Tensor] = None,
               max_decoding_length: Optional[int] = None) \
            -> Union[tx.modules.BasicRNNDecoderOutput,
                     tx.modules.TransformerDecoderOutput]:
        self._latent_z = latent_z
        fc_output = self.mlp_linear_layer(latent_z)

        if self._config.decoder_type == "lstm":
            lstm_states = torch.chunk(fc_output, 2, dim=1)
            outputs, _, _ = self.lstm_decoder(
                initial_state=lstm_states,
                inputs=text_ids,
                helper=helper,
                sequence_length=seq_lengths,
                max_decoding_length=max_decoding_length)
        else:
            transformer_states = fc_output.unsqueeze(1)
            outputs = self.transformer_decoder(
                inputs=text_ids,
                memory=transformer_states,
                memory_sequence_length=torch.ones(transformer_states.size(0)),
                helper=helper,
                max_decoding_length=max_decoding_length)
        return outputs
        
    