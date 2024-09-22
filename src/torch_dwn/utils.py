import torch

def pad_if_needed(x, n):
    if not (x.size(-1) % n == 0):
        pad = torch.zeros(len(x.shape), dtype=torch.int)
        pad[-1] = n - (x.size(-1) % n)
        pad = tuple(pad.numpy().tolist())
        x = torch.nn.functional.pad(x, pad)
    return x

class GroupSum(torch.nn.Module):
    def __init__(self, k, tau=1, randperm=False):
        super().__init__()
        self.k = k
        self.tau = tau
        self.randperm = torch.randperm(randperm) if randperm else None

    def forward(self, x):
        if self.randperm is not None:
            x = x[:, self.randperm]
        x = pad_if_needed(x, self.k)
        x = x.view(*x.shape[:-1], self.k, int(x.shape[-1]/self.k))
        return x.sum(dim=-1) / self.tau

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad # torch.nn.functional.hardtanh(output_grad)

class STE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return STEFunction.apply(x)