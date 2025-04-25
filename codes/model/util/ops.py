import torch


def safe_unsqueeze(tensor, dim):
    assert tensor.size(dim) == 1
    return tensor.unsqueeze(dim)


def safe_squeeze(tensor, dim):
    assert tensor.size(dim) == 1
    return tensor.squeeze(dim)


def safe_stack(tensors, dim):
    if tensors is None:
        return None
    elif isinstance(tensors, (list, tuple)):
        valid_tensors = [tensor for tensor in tensors if tensor is not None]
        if len(valid_tensors) == 0:
            return None
        else:
            return torch.stack(valid_tensors, dim)
    else:
        return tensors.unsqueeze(dim)
