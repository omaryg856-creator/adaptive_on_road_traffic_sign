import torch
import torch.nn as nn
import torch.nn.functional as F

class VerifierCNN(nn.Module):
    """
    Simple binary classifier:
        class 0 -> SIGN NOT PRESENT (or incorrect)
        class 1 -> SIGN PRESENT (correct expected sign)
    Input image: 3 x 64 x 64
    """
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x, threshold=0.5):
        probs = self.predict_proba(x)[:, 1]
        return (probs >= threshold).long()
