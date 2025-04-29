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


def slice_segments(x, ids_start, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_start[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = (
            torch.Tensor(
                [
                    t,
                ]
                * b
            )
            .to(device=x.device)
            .to(dtype=torch.long)
        )
    ids_start_max = x_lengths - segment_size + 1
    ids_start = (torch.rand([b]).to(device=x.device) * ids_start_max).clamp(min=0.0).to(dtype=torch.long)
    ret = slice_segments(x, ids_start, segment_size)
    segment_size_tensor = (
        torch.Tensor(
            [
                segment_size,
            ]
            * b
        )
        .to(device=x.device)
        .to(dtype=torch.long)
    )
    slice_lengths = x_lengths * (ids_start_max <= 0) + segment_size_tensor * (ids_start_max > 0)
    return ret, ids_start, slice_lengths
