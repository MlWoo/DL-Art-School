import torch
from torch import nn


def apply_compile_transformer(model: nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, block in model.layers.named_children():
        new_block = torch.compile(block, dynamic=True, fullgraph=True)  # 最大优化级别
        model.layers.register_module(layer_id, new_block)
    print("Compiling each TransformerBlock with torch.compile")
