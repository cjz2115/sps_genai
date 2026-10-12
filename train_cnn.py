import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from app.cnn import CNN

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

data = datasets.CIFAR10("data", train=True, download=True, transform=transform)
loader = DataLoader(data, batch_size=64, shuffle=True)

model = CNN()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(3):
    for x, y in loader:
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    print("epoch", epoch + 1, "done, loss:", loss.item())

torch.save(model.state_dict(), "app/cnn.pth")
print("saved app/cnn.pth")
