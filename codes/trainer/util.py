from math import inf

import torch


def set_requires_grad(nets, requires_grad: bool = False, set_to_none: bool = False):
    """Set requires_grad for all the networks.
    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for p in net.parameters():
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                        p.requires_grad = requires_grad
                    else:
                        p.grad.zero_()
                        p.requires_grad = requires_grad
                else:
                    if set_to_none and requires_grad:
                        p.grad = torch.zeros_like(p)
                        p.requires_grad = requires_grad


def clip_grad_norm(parameters: list, parameter_names: list, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    r"""
    Equivalent to torch.nn.utils.clip_grad_norm_() but with the following changes:
    - Takes in a dictionary of parameters (from get_named_parameters()) instead of a list of parameters.
    - When NaN or inf norms are encountered, the parameter name is printed.
    - error_if_nonfinite removed.

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:``parameters`` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm


# Recursively detaches all tensors in a tree of lists, dicts and tuples and returns the same structure.
def recursively_detach(v, reuse_out=None):
    if isinstance(v, torch.Tensor):
        return v.detach().clone()
    elif isinstance(v, list) or isinstance(v, tuple):
        out = [recursively_detach(i) for i in v]
        if isinstance(v, tuple):
            return tuple(out)
        return out
    elif isinstance(v, dict):
        out = {}
        for k, t in v.items():
            if reuse_out is None or (k not in reuse_out and k != reuse_out):
                out[k] = recursively_detach(t)
            else:
                out[k] = t
        return out
