import torch


def generate_causal_mask(size):


    mask = torch.triu(
        torch.ones(size, size),
        diagonal=1
    )



    mask = mask.bool()


    return mask
