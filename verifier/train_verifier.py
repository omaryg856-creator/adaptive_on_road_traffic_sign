import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from verifier_model import VerifierCNN


def get_dataloaders(
    data_root="data/verifier",
    batch_size=32,
    img_size=64,
):
    """
    Expects folder:
        data/verifier/pos
        data/verifier/neg
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_root, transform=transform)

    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train

    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    train_loader, val_loader = get_dataloaders()

    model = VerifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | ValAcc={acc:.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/verifier_cnn.pth")
    print("Saved to weights/verifier_cnn.pth")


if __name__ == "__main__":
    main()
