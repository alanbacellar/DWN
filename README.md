
# DWN: *Differentiable Weightless Neural Networks*

📄 **Paper**: [ICML 2024](https://proceedings.mlr.press/v235/bacellar24a.html) | [Updated Version on arXiv](https://arxiv.org/pdf/2410.11112)

---

## Installation

```bash
pip install torch-dwn
```

Requirements: CUDA and PyTorch (matching the CUDA version).

---

## Quick Start

### MNIST Example

To quickly get started with DWN, here's an example using the MNIST dataset. Full training code is available in the [examples/mnist.py](examples/mnist.py) file.

```python
import torch
from torch import nn
import torch_dwn as dwn

# Load MNIST dataset
x_train, y_train, x_test, y_test = load_mnist() 

# Binarize using the Distributive Thermometer
thermometer = dwn.DistributiveThermometer(3).fit(x_train)
x_train = thermometer.binarize(x_train).flatten(start_dim=1)
x_test = thermometer.binarize(x_test).flatten(start_dim=1)

# Define the model
model = nn.Sequential(
    dwn.LUTLayer(x_train.size(1), 2000, n=6, mapping='learnable'),
    dwn.LUTLayer(2000, 1000, n=6),
    dwn.GroupSum(10, tau=1/0.3)
)

# Optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)

# Training and evaluation
train(model, x_train, y_train, optimizer, scheduler, batch_size=32, epochs=30)
evaluate(model, x_test, y_test)
```

---

## Quick Docs

### `DistributiveThermometer`

```python
DistributiveThermometer(
    num_bits,            # Number of bits per feature in the encoding.
    feature_wise=True,   # Whether to fit the encoding feature-wise or globally.
)
```
- `.fit(x)`: Fit the encoding using the training data.
- `.binarize(x)`: Binarize new data. **Returns**: Encoded input with shape: (*x.shape, num_bits).

### `LUTLayer`

```python
LUTLayer(
    input_size,         # Input size for the layer.
    output_size,        # Output size (number of LUTs).
    n,                  # Number of inputs each LUT receives (LUT size is 2^n).
    mapping='random',   # Mapping strategy for LUTs: 'random', 'learnable', or 'arange'.
    alpha=None,         # Linear scalar for EFD. Defaults to a value based on `n`.
    beta=None,          # Exponential scalar for EFD. Defaults to a value based on `n`.
    ste=True,           # Whether to use Straight-Through Estimator (STE) for binarization.
    clamp_luts=True,    # Clamps LUTs during training to [-1, 1] (for STE).
    lm_tau=0.001        # Temperature parameter for softmax when using learnable mapping.
)
```

### `LearnableMapping`

```python
LearnableMapping(
    input_size,  # Input size.
    output_size, # Output size, typically n * N (LUT inputs * number of LUTs in the next layer).
    tau=0.001    # Softmax temperature for the backpropagation.
)
```

### `GroupSum`
Applies a grouped sum (popcount) over the outputs, followed by softmax temperature:
```python
GroupSum(
    num_groups,   # Number of groups (e.g., classes).
    tau=1         # Temperature for the softmax
)
```

---

## Advanced Notes and Tips

### **EFD Implementation**

The EFD implementation in this repository improves upon the original paper's version. We utilize **exponential decay** instead of linear decay, which will be detailed in an upcoming paper. For now, users can take advantage of the latest version here.

### **Learnable Mapping**

We recommend using **learnable mapping** only in the first layer for faster convergence with negligible accuracy differences. Using it in all layers requires fine-tuning of the softmax temperature in the backprop and may slow down training.

### **Softmax Temperature in the GroupSum Layer**

We highly recommend fine-tuning the softmax temperature in the GroupSum layer of the DWN model. This was observed to be crucial for achieving high accuracy, similar to the behavior observed in DiffLogicNet.

### **Model Architecture Design**

DWN models benefit from **shallower architectures**, primarily due to the **learnable mapping** mechanism. With **random mapping**, deeper architectures are often needed to combine input features in later layers for better classification. However, **learnable mapping** allows this feature combination to happen earlier, making additional depth unnecessary.

---
