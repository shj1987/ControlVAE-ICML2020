import torch


def _active_unit(model, test_data_batch, start_tokens, end_token, delta=0.01):
    """compute the number of active units
    """
    cnt = 0
    kl_weight = 1
    for batch_data in test_data_batch:
        ret = model(batch_data, kl_weight, start_tokens, end_token)
        mean = ret['mu']
        means_sum += mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)

    # (1, nz)
    mean_mean = means_sum / cnt

    cnt = 0
    for batch_data in test_data_batch:
        ret = model(batch_data, kl_weight, start_tokens, end_token)
        mean = ret['mu']
        var_sum += ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)

    # (nz)
    au_var = var_sum / (cnt - 1)

    return (au_var >= delta).sum().item()
    