from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_mask(sequence_lengths: Tensor, valid: bool = True, end: bool = False, max_len: Optional[int] = None):
    if max_len is None:
        max_len = sequence_lengths.max()
    batch_size = sequence_lengths.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    device = sequence_lengths.get_device()
    if device >= 0:
        seq_range_expand = seq_range_expand.to(torch.device("cuda:" + str(device)))
    seq_length_expand = sequence_lengths.unsqueeze(1).expand_as(seq_range_expand)
    if end:
        if valid:
            return seq_range_expand == (seq_length_expand - 1)
        else:
            return seq_range_expand != (seq_length_expand - 1)
    else:
        if valid:
            return seq_range_expand < seq_length_expand
        else:
            return seq_range_expand >= seq_length_expand


def create_batch_segmented_attention_mask(input_ids, eos_token_id=5000):
    """
    批量高效分段注意力掩码生成（支持二维输入）

    参数:
        input_ids (Tensor): 输入序列的索引张量，形状为 [batch_size, seq_len]
        eos_token_id (int): EOS标记的ID，默认为5000

    返回:
        Tensor: 注意力掩码矩阵，形状为 [batch_size, seq_len, seq_len]
                True表示允许注意力，False表示屏蔽
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # 生成段标识 (核心优化)
    eos_mask = input_ids == eos_token_id

    # 创建shifted_eos_mask标记段分界点（EOS的下一个位置）
    shifted_eos_mask = torch.zeros_like(input_ids)
    shifted_eos_mask[:, 1:] = eos_mask[:, :-1]  # 将EOS后的位置标记为分界点

    # 生成段号序列 (每个样本独立计算)
    segment_ids = shifted_eos_mask.cumsum(dim=1)

    # 生成因果掩码模板 [1, seq_len, seq_len]
    causal_mask = torch.tril(torch.ones(1, seq_len, seq_len, dtype=torch.bool, device=device))

    # 生成同段掩码 [batch_size, seq_len, seq_len]
    same_segment = segment_ids.unsqueeze(-1) == segment_ids.unsqueeze(-2)

    # 合并条件 [batch_size, seq_len, seq_len]
    return causal_mask & same_segment


def get_inner_score_mask(mask, q_l, kv_l, device, sample, sample_t, offset_t=None, offset_tf=None):
    # returns a mask of shape 1 x 1 x q_l x kv_l or None if masking is not needed.
    if mask is None or q_l == 1:
        return None
    else:
        if kv_l is None:
            kv_l = q_l
        if offset_t is None:
            offset = sample_t - q_l if sample else max(kv_l - q_l, 0)
        else:
            offset = offset_t
        if offset_tf is None:
            offset_tf = offset
        ones = torch.ones(q_l, kv_l, device=device)
        if mask in ["autoregressive", "ar"]:
            # Masked dense
            mask = ones.tril(offset)
        elif mask == "window":
            mask = ones.triu(-1 * offset).tril(offset_tf)
        elif mask == "window_ar":
            mask = ones.triu(-1 * offset + 1).tril(0)
        else:
            raise ValueError(f"{mask} mode is not supported!")

        return mask.view(1, 1, q_l, kv_l).bool()


def get_outter_score_mask(
    mask: Optional[Tensor] = None, q_mask: Optional[Tensor] = None, kv_mask: Optional[Tensor] = None
):
    # mask: B, C, T_q, T_kv
    # q_mask: [B T C], C = 1
    # kv_mask: [B T C], C = 1
    if mask is None and q_mask is None:
        return None
    elif mask is not None:
        return mask
    else:
        x_mask = q_mask.squeeze(dim=-1)
        assert x_mask.dim() == 2, "Query mask should not be expanded in last dim"
        if kv_mask is None:
            x_mask_ex = x_mask.unsqueeze(dim=1)
            mask = torch.logical_and(x_mask_ex.unsqueeze(dim=-1), x_mask_ex.unsqueeze(dim=2))
        else:
            y_mask = kv_mask.squeeze(dim=-1)
            assert y_mask.dim() == 2, "Key-Value mask should not be expanded in last dim"
            q_mask_ex = x_mask.unsqueeze(dim=1).unsqueeze(dim=-1)  # B 1 T 1
            kv_mask_ex = y_mask.unsqueeze(dim=1).unsqueeze(dim=2)  # B 1 1 T
            mask = torch.logical_and(q_mask_ex, kv_mask_ex)

        return mask


def get_chunks_ceil_score_mask(seq_l, chunk_size, device, return_folded=False):
    chunks = (seq_l - 1) // chunk_size + 1
    seq_l_ceil = chunks * chunk_size
    x_ceil_mask = torch.zeros(seq_l_ceil, seq_l_ceil, device=device, dtype=torch.bool)
    chunk_score_mask = x_ceil_mask.reshape(chunks, chunk_size, chunks, chunk_size)
    chunk_score_mask = chunk_score_mask.permute(0, 2, 1, 3).reshape(-1, chunk_size, chunk_size)
    chunk_score_mask[0 :: (chunks + 1)] = True
    if not return_folded:
        chunk_score_mask = chunk_score_mask.reshape(chunks, chunks, chunk_size, chunk_size)
        chunk_score_mask = chunk_score_mask.permute(0, 2, 1, 3).reshape(seq_l_ceil, seq_l_ceil)

    return chunk_score_mask, chunks


def pad_list(xs, pad_value=0.0):
    """Perform padding for the list of tensors.
    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).
    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    device = xs[0].get_device()
    if device >= 0:
        pad = pad.to(torch.device("cuda:" + str(device)))

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def mask_reduce(x, sequence_lengths=None, dim=-1, mask=None, max_len=None, reduction="mean", keepdim=False):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    assert not (sequence_lengths is None and mask is None), "One of `sequence_lengths` and `mask` should be provided."
    # if weight is specified, apply element-wise weight
    if mask is None:
        mask = sequence_mask(sequence_lengths, valid=True, max_len=max_len)

    x = x * mask
    if reduction == "mean":
        x = x.sum(dim=dim, keepdim=keepdim) / mask.sum(dim=dim, keepdim=keepdim)
    elif reduction == "sum":
        x = x.sum(dim=dim, keepdim=keepdim)
    else:
        raise ValueError('can not be used beyond reduction="sum" and "mean"')
    return x


@torch.jit.export
def generate_path(duration, mask):
    """
    duration: [b, t_x]
    mask: [b, t_y, t_x]
    """
    b, t_x = duration.shape
    t_y = mask.shape[-1]
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, max_len=t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y).long()

    path = path - F.pad(path, (0, 0, 1, 0))[:, :-1]
    path = path * mask
    return path


@torch.jit.export
def expand_by_duration(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    if d.dtype == torch.float32 or d.dtype == torch.float64 or d.dtype == torch.half:
        d = torch.round(d).long()
    return pad_list([torch.repeat_interleave(x, d, dim=0) for x, d in zip(x, d)], 0.0)


if __name__ == "__main__":
    # 示例输入序列 (包含两个EOS)
    input_ids = torch.tensor([[10, 20, 5000, 30, 40, 300, 5000, 50], [10, 20, 30, 5000, 40, 300, 50, 5000]])

    # 生成注意力掩码
    mask = create_batch_segmented_attention_mask(input_ids)

    # 可视化掩码矩阵
    print("注意力掩码矩阵:")
    print(mask.int().numpy())
