import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch_dwn as dwn

# Load Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the 28x28 image to 784
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

x_train, y_train = next(iter(train_loader))
x_test, y_test = next(iter(test_loader))

# Binarize with distributive thermometer
thermometer = dwn.DistributiveThermometer(3).fit(x_train)
x_train = thermometer.binarize(x_train).flatten(start_dim=1)
x_test = thermometer.binarize(x_test).flatten(start_dim=1)

model = nn.Sequential(
    dwn.LUTLayer(x_train.size(1), 2000, n=6, mapping='learnable'),
    dwn.LUTLayer(2000, 1000, n=6),
    dwn.GroupSum(k=10, tau=1/0.3)
)

model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)

def train(model, optimizer, scheduler, x_train, y_train, epochs, batch_size):
    model.train()
    n_samples = x_train.shape[0]
    
    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            optimizer.zero_grad()
            
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices].cuda(), y_train[indices].cuda()
            
            outputs = model(batch_x)
            loss = cross_entropy(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = (model(x_test.cuda()).cpu()).argmax(dim=1).numpy()
        acc = (pred == y_test.numpy()).sum() / y_test.shape[0]
    return acc

train(model, optimizer, scheduler, x_train, y_train, epochs=30, batch_size=32)
acc = evaluate(model, x_test, y_test)
print(f"Accuracy: {acc}")
