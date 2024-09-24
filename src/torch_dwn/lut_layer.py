import torch
if torch.cuda.is_available():
    import efd_cuda
from .mapping import layer_mapping, LearnableMapping
from .utils import STEFunction

class EFDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mapping, luts, alpha, beta):
        if x.is_cuda:
            output = efd_cuda.forward(x, mapping, luts)
        else:
            raise "EFDFunction CPU not Implemented"
        ctx.save_for_backward(x, mapping, luts, alpha, beta)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.is_cuda:
            input_grad, luts_grad = efd_cuda.backward(*ctx.saved_tensors, output_grad.contiguous())
        else:
            raise "EFDFunction CPU not Implemented"
        return input_grad, None, luts_grad, None, None

class LUTLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, n, mapping='random', alpha=None, beta=None, ste=True, 
                 clamp_luts=True, lm_tau=0.001):
        super().__init__()
        
        # Input Check
        assert input_size > 0
        assert output_size > 0
        assert n > 0
        assert mapping in ('arange', 'random', 'learnable') or (isinstance(mapping, torch.Tensor) and mapping.dtype == torch.int32 and mapping.shape == torch.Size([output_size, n]))
        assert isinstance(ste ,bool)
        assert isinstance(clamp_luts ,bool)

        # Vars
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.n = int(n)
        self.ste = ste
        self.clamp_luts = clamp_luts

        # Alpha and beta
        if alpha is None:
            alpha = 0.5 * 0.75**(n-1)
        if beta is None:
            beta = 0.25/0.75
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        assert self.alpha.dtype in (torch.float16, torch.float32, torch.float64)
        assert self.beta.dtype in (torch.float16, torch.float32, torch.float64)

        # Mapping
        if isinstance(mapping, torch.Tensor):
            self.mapping = torch.nn.Parameter(mapping, requires_grad=False)
        elif mapping == 'learnable':
            self.mapping = LearnableMapping(input_size, output_size*n, tau=lm_tau)
            self.__dummy_mapping = torch.nn.Parameter(torch.arange(output_size*n).reshape(output_size, n).int(), requires_grad=False)
        else:
            self.mapping = torch.nn.Parameter(layer_mapping(input_size, n, output_size, random=(mapping=='random')), requires_grad=False)
        
        # LUTs
        luts = torch.rand(output_size, 2**n, dtype=torch.float32)*2 - 1
        self.luts = torch.nn.Parameter(luts, requires_grad=True)
        
    def forward(self, x):
        
        # Clamp LUTs
        if self.training and self.clamp_luts:
            with torch.no_grad():
                self.luts.clamp_(-1, 1)
        
        # Learnable Mapping
        if isinstance(self.mapping, LearnableMapping):
            x = self.mapping(x)
            mapping = self.__dummy_mapping
        else:
            mapping = self.mapping
        
        # EFD
        x = EFDFunction.apply(x, mapping, self.luts, self.alpha, self.beta)

        # STE
        if self.ste:
            x = STEFunction.apply(x)
        
        return x
