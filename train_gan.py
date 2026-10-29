import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from app.gan import Generator, Discriminator

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 20

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
data = datasets.MNIST("./data", train=True, download=True, transform=tf)
loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

G = Generator().to(device)
D = Discriminator().to(device)
loss_fn = nn.BCELoss()
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(EPOCHS):
    for real, _ in loader:
        real = real.to(device)
        ones = torch.ones(BATCH_SIZE, 1, device=device)
        zeros = torch.zeros(BATCH_SIZE, 1, device=device)

        z = torch.randn(BATCH_SIZE, 100, device=device)
        fake = G(z)
        d_loss = loss_fn(D(real), ones) + loss_fn(D(fake.detach()), zeros)
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        g_loss = loss_fn(D(fake), ones)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

    print(f"Epoch {epoch + 1}/{EPOCHS}  D loss {d_loss.item():.3f}  G loss {g_loss.item():.3f}")

torch.save(G.state_dict(), "app/gan_generator.pth")
