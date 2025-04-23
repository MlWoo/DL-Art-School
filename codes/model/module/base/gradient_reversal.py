import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_, clip_value_):
        ctx.save_for_backward(lambda_, clip_value_)
        return x

    @staticmethod
    def backward(ctx, grads):
        lambda_, clip_value_ = ctx.saved_tensors
        if clip_value_ > 0:
            grads = torch.clamp(grads, -clip_value_, clip_value_)
        dx = -lambda_ * grads
        return dx, None, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1.0, clip_value_=-1.0):
        super(GradientReversal, self).__init__()
        self.register_buffer("lambda_", torch.tensor(lambda_, dtype=torch.float32))
        self.register_buffer("clip_value_", torch.tensor(clip_value_, dtype=torch.float32))

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_, self.clip_value_)
