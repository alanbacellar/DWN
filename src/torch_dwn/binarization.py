import torch

class Thermometer:
    def __init__(self, num_bits=1, feature_wise=True):
        
        assert num_bits > 0
        assert type(feature_wise) is bool

        self.num_bits = int(num_bits)
        self.feature_wise = feature_wise
        self.thresholds = None

    def get_thresholds(self, x):
        min_value = x.min(dim=0)[0] if self.feature_wise else x.min()
        max_value = x.max(dim=0)[0] if self.feature_wise else x.max()
        return min_value.unsqueeze(-1) + torch.arange(1, self.num_bits+1).unsqueeze(0) * ((max_value - min_value) / (self.num_bits + 1)).unsqueeze(-1)

    def fit(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        self.thresholds = self.get_thresholds(x)
        return self
    
    def binarize(self, x):
        if self.thresholds is None:
            raise 'need to fit before calling apply'
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        x = x.unsqueeze(-1)
        return (x > self.thresholds).float()

class GaussianThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        std_skews = torch.distributions.Normal(0, 1).icdf(torch.arange(1, self.num_bits+1)/(self.num_bits+1))
        mean = x.mean(dim=0) if self.feature_wise else x.mean()
        std = x.std(dim=0) if self.feature_wise else x.std() 
        thresholds = torch.stack([std_skew * std + mean for std_skew in std_skews], dim=-1)
        return thresholds
    
class DistributiveThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        data = torch.sort(x.flatten())[0] if not self.feature_wise else torch.sort(x, dim=0)[0]
        indicies = torch.tensor([int(data.shape[0]*i/(self.num_bits+1)) for i in range(1, self.num_bits+1)])
        thresholds = data[indicies]
        return torch.permute(thresholds, (*list(range(1, thresholds.ndim)), 0))

