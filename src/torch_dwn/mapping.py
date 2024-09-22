import torch
from torch.nn.functional import softmax

def layer_mapping(input_size, tuple_lenght, output_size, random=True):
    mapp = lambda : torch.randperm(input_size) if random else torch.arange(input_size)
    num_complete, offset = divmod(tuple_lenght*output_size, input_size)
    if num_complete == 0:
        return mapp()[:offset].reshape(output_size, tuple_lenght).type(torch.int32)
    else:
        complete_mappings = torch.cat([mapp() for i in range(num_complete)])
        offset_mapping = mapp()[:offset]
        return torch.cat([complete_mappings, offset_mapping]).reshape(output_size, tuple_lenght).type(torch.int32)

class LearnableMappingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, tau):
        mapping = weights.argmax(dim=0)
        output = x[:, mapping]
        ctx.save_for_backward(x, weights, tau)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        x, weights, tau  = ctx.saved_tensors
        weights_grad = ((2*x-1).T @ output_grad)
        input_grad = output_grad @ softmax(weights/tau, dim=0).T
        return input_grad, weights_grad, None, None
    

class LearnableMapping(torch.nn.Module):
    def __init__(self, input_size, output_size, tau=0.001):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand(input_size, output_size, dtype=torch.float32), requires_grad=True)
        self.tau = tau

    def forward(self, x):
        return LearnableMappingFunction.apply(x, self.weights, torch.tensor(self.tau))