# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example for building the Variational Auto-Encoder.

This is an implementation of Variational Auto-Encoder for text generation

To run:

$ python vae_train.py

Hyperparameters and data path may be specified in config_trans.py

"""

import argparse
import importlib
import math
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR

import texar.torch as tx
from texar.torch.custom import MultivariateNormalDiag
from model import VAE
from PID import PIDControl
from annealing import _cost_annealing, _cyclical_annealing
from tqdm import tqdm
from utils import _active_unit

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

## assign gpu

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help="cuda id.")
parser.add_argument('--config', type=str, default=None, help="The config to use.")
parser.add_argument('--mode', type=str, default='train', help="Train or predict.")
parser.add_argument('--model', type=str, default=None, help="Model path for generating sentences.")
parser.add_argument('--out', type=str, default=None, help="Generation output path.")
parser.add_argument('--model_name', type=str, default='cost_anneal', help="Generation output path.")
parser.add_argument('--exp_kl', type=float, default=0, help="desired KL divergence.")
parser.add_argument('--Kp', type=float, default=0.01, help="Kp for pid.")
parser.add_argument('--Ki', type=float, default=-0.0001, help="Kp for pid.")
parser.add_argument('--cycle', type=float, default=4, help="Kp for pid.")
parser.add_argument('--anneal_steps', type=float, default=10000, help="steps for anneal.")
parser.add_argument('--max_steps', type=int, default=80000, help="steps for anneal.")


args = parser.parse_args()

torch.cuda.set_device(args.gpu)


def main():
    """Entrypoint.
    """
    config: Any = importlib.import_module(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_data = tx.data.MonoTextData(config.train_data_hparams, device=device)
    # val_data = tx.data.MonoTextData(config.val_data_hparams, device=device)
    # test_data = tx.data.MonoTextData(config.test_data_hparams, device=device)

    train_data = tx.data.MonoTextData(config.train_data_hparams, device=torch.device("cpu"))
    val_data = tx.data.MonoTextData(config.val_data_hparams, device=torch.device("cpu"))
    test_data = tx.data.MonoTextData(config.test_data_hparams, device=torch.device("cpu"))
    
    iterator = tx.data.DataIterator(
        {"train": train_data, "valid": val_data, "test": test_data})

    opt_vars = {
        'learning_rate': config.lr_decay_hparams["init_lr"],
        'best_valid_nll': 1e100,
        'steps_not_improved': 0,
        'kl_weight': config.kl_anneal_hparams["start"]
    }

    decay_cnt = 0
    max_decay = config.lr_decay_hparams["max_decay"]
    decay_factor = config.lr_decay_hparams["decay_factor"]
    decay_ts = config.lr_decay_hparams["threshold"]

    if 'pid' in args.model_name:
        save_dir = args.model_name + '_'+ str(config.dataset) + '_KL' + str(args.exp_kl)
    elif 'cost' in args.model_name:
         save_dir = args.model_name + '_'+ str(config.dataset) + '_step' + str(args.anneal_steps)
    elif 'cyclical' in args.model_name:
         save_dir = args.model_name + '_'+ str(config.dataset) + '_cyc_' + str(args.cycle)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    suffix = f"{config.dataset}_{config.decoder_type}Decoder.ckpt"

    save_path = os.path.join(save_dir, suffix)
    
    # KL term annealing rate warm_up=10
    ## replace it with sigmoid function
    anneal_r = 1.0 / (config.kl_anneal_hparams["warm_up"] *
                      (len(train_data) / config.batch_size))

    vocab = train_data.vocab
    model = VAE(train_data.vocab.size, config)
    model.to(device)

    start_tokens = torch.full(
        (config.batch_size,),
        vocab.bos_token_id,
        dtype=torch.long).to(device)
    end_token = vocab.eos_token_id
    optimizer = tx.core.get_optimizer(
        params=model.parameters(),
        hparams=config.opt_hparams)
    scheduler = ExponentialLR(optimizer, decay_factor)

    ## max iteration
    max_iter = config.num_epochs*len(train_data)/config.batch_size
    max_iter = min(max_iter, args.max_steps)
    print('max steps:', max_iter)
    pbar = tqdm(total = int(max_iter))
    
    if args.mode == "train":
        outFile = os.path.join(save_dir, 'train.log')
        fw_log = open(outFile, "w")
    
    global_steps = {}
    global_steps['step'] = 0
    pid = PIDControl()
    opt_vars["kl_weight"] = 0.0
    Kp = args.Kp
    Ki = args.Ki
    exp_kl = args.exp_kl

    ## train model
    def _run_epoch(epoch: int, mode: str, display: int = 10) \
            -> Tuple[Tensor, float]:
        iterator.switch_to_dataset(mode)

        if mode == 'train':
            model.train()
            kl_weight = opt_vars["kl_weight"]
        else:
            model.eval()
            kl_weight = 1.0
            # kl_weight = opt_vars["kl_weight"]
            
        start_time = time.time()
        num_words = 0
        nll_total = 0.
        
        avg_rec = tx.utils.AverageRecorder()

        ## compute active unit
        if mode == 'test':
            ac_count = _active_unit(model, iterator, start_tokens, end_token)
            print("num of active unit: ", ac_count)

        for batch in iterator:
            ## run model to get loss function
            if global_steps['step']>= args.max_steps:
                break
            ret = model(batch, kl_weight, start_tokens, end_token)
            # mean = ret['mu']

            if mode == "train":
                pbar.update(1)
                global_steps['step'] += 1
                kl_loss = ret['kl_loss'].item()
                rec_loss = ret['rc_loss'].item()
                total_loss = ret["nll"].item()
                if 'cost' in args.model_name:
                    kl_weight = _cost_annealing(global_steps['step'], 1.0, args.anneal_steps)
                elif 'pid' in args.model_name:
                    kl_weight = pid.pid(exp_kl, kl_loss, Kp, Ki)
                elif 'cyclical' in args.model_name:
                    kl_weight = _cyclical_annealing(global_steps['step'], max_iter/args.cycle)

                opt_vars["kl_weight"] = kl_weight
                
                ## total loss
                ret["nll"].backward()
                optimizer.step()
                optimizer.zero_grad()
                fw_log.write('epoch:{0} global_step:{1} total_loss:{2:.3f} kl_loss:{3:.3f} rec_loss:{4:.3f} kl_weight:{5:.4f}\n'\
                            .format(epoch, global_steps['step'], total_loss, kl_loss, rec_loss, kl_weight))
                fw_log.flush()

            batch_size = len(ret["lengths"])
            num_words += torch.sum(ret["lengths"]).item()
            nll_total += ret["nll"].item() * batch_size
            avg_rec.add(
                [ret["nll"].item(),
                 ret["kl_loss"].item(),
                 ret["rc_loss"].item()],
                batch_size)
                
            if global_steps['step'] % display == 1 and mode == 'train':
                nll = avg_rec.avg(0)
                klw = opt_vars["kl_weight"]
                KL = avg_rec.avg(1)
                rc = avg_rec.avg(2)
                writer.add_scalar(f'Loss/Rec_loss_{args.model_name}', rc, global_steps['step'])
                writer.add_scalar(f'Loss/KL_diverg_{args.model_name}', KL, global_steps['step'])
                writer.add_scalar(f'Loss/KL_weight_{args.model_name}', klw, global_steps['step'])
                
        nll = avg_rec.avg(0)
        KL = avg_rec.avg(1)
        rc = avg_rec.avg(2)
        log_ppl = nll_total / num_words
        ppl = math.exp(log_ppl)
        print(f"\n{mode}: epoch {epoch}, nll {nll:.4f}, KL {KL:.4f}, "
              f"rc {rc:.4f}, log_ppl {log_ppl:.4f}, ppl {ppl:.4f}")
        return nll, ppl  # type: ignore
        
        
    args.model = save_path
    @torch.no_grad()
    def _generate(start_tokens: torch.LongTensor,
                  end_token: int,
                  filename: Optional[str] = None):
        ckpt = torch.load(args.model)
        model.load_state_dict(ckpt['model'])
        model.eval()

        batch_size = train_data.batch_size

        dst = MultivariateNormalDiag(
            loc=torch.zeros(batch_size, config.latent_dims),
            scale_diag=torch.ones(batch_size, config.latent_dims))

        # latent_z = dst.rsample().to(device)
        latent_z = torch.FloatTensor(batch_size, config.latent_dims).uniform_(-1, 1).to(device)
        # latent_z = torch.randn(batch_size, config.latent_dims).to(device)

        helper = model.decoder.create_helper(
            decoding_strategy='infer_sample',
            start_tokens=start_tokens,
            end_token=end_token)
        outputs = model.decode(
            helper=helper,
            latent_z=latent_z,
            max_decoding_length=100)

        if config.decoder_type == "transformer":
            outputs = outputs[0]

        sample_tokens = vocab.map_ids_to_tokens_py(outputs.sample_id.cpu())

        if filename is None:
            fh = sys.stdout
        else:
            fh = open(filename, 'a', encoding='utf-8')

        for sent in sample_tokens:
            sent = tx.utils.compat_as_text(list(sent))
            end_id = len(sent)
            if vocab.eos_token in sent:
                end_id = sent.index(vocab.eos_token)
            fh.write(' '.join(sent[:end_id + 1]) + '\n')

        print('Output done')
        fh.close()

    if args.mode == "predict":
        out_path = os.path.join(save_dir,'results.txt')
        for _ in range(10):
            _generate(start_tokens, end_token, out_path)
        return

    # Counts trainable parameters
    total_parameters = sum(param.numel() for param in model.parameters())
    print(f"{total_parameters} total parameters")
    
    best_nll = best_ppl = 0.
    nll_list = []
    ppl_list = []
    ## start running model
    if args.mode == "train":
        for epoch in range(config.num_epochs):
            _, _ = _run_epoch(epoch, 'train', display=200)
            val_nll, _ = _run_epoch(epoch, 'valid')
            test_nll, test_ppl = _run_epoch(epoch, 'test')
            nll_list.append(test_nll)
            ppl_list.append(test_ppl)

            if val_nll < opt_vars['best_valid_nll']:
                opt_vars['best_valid_nll'] = val_nll
                opt_vars['steps_not_improved'] = 0
                best_nll = test_nll
                best_ppl = test_ppl

                states = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(states, save_path)
            else:
                opt_vars['steps_not_improved'] += 1
                if opt_vars['steps_not_improved'] == decay_ts:
                    old_lr = opt_vars['learning_rate']
                    opt_vars['learning_rate'] *= decay_factor
                    opt_vars['steps_not_improved'] = 0
                    new_lr = opt_vars['learning_rate']
                    ckpt = torch.load(save_path)
                    model.load_state_dict(ckpt['model'])
                    optimizer.load_state_dict(ckpt['optimizer'])
                    scheduler.load_state_dict(ckpt['scheduler'])
                    scheduler.step()
                    print(f"-----\nchange lr, old lr: {old_lr}, "
                          f"new lr: {new_lr}\n-----")

                    decay_cnt += 1
                    if decay_cnt == max_decay:
                        break
            if global_steps['step'] >= args.max_steps:
                break

    elif args.mode == "test":
        test_nll, test_ppl = _run_epoch(1, 'test')
        nll_list.append(test_nll)
        ppl_list.append(test_ppl)
        
    # print(f"\nbest testing nll: {best_nll:.4f},"
    #       f"best testing ppl {best_ppl:.4f}\n")
    # avg_nll = np.mean(nll_list)
    # avg_ppl = np.mean(ppl_list)
    
    # print(f"\navg testing nll: {avg_nll:.4f},"
    #       f"avg testing ppl {avg_ppl:.4f}\n")
    
    if args.mode == "train":
        fw_log.write(f"\nbest testing nll: {best_nll:.4f},"
          f"best testing ppl {best_ppl:.4f}\n")
        fw_log.close()
        

if __name__ == '__main__':
    main()
    print("well done!!!!!")



    