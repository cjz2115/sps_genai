import torch
from app.cnn import CNN

model = CNN()
x = torch.randn(1, 3, 64, 64)
print(model(x).shape)
