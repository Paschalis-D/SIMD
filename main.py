import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass  # <--- Added missing import
from typing import Tuple  # <--- Added for type hinting

import torch
import torch.nn as nn  # <--- Added for loss functions
from torch.utils.data import DataLoader

from utils.model import SIMD
from utils.dataset import SIMDataset
from utils.loss import binary_diceCE


# 1. CONFIGURATION
@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 50
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-5


cfg = TrainConfig()

# 2. SETUP DEVICE
# This is critical for speed.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

ROOT = Path(os.getcwd())
data_dir = ROOT / "recodai-luc-scientific-image-forgery-detection"

# Setup Paths
train_images = data_dir / "train_images" / "forged"
train_masks = data_dir / "train_masks"
test_images = data_dir / "supplemental_images"
test_masks = data_dir / "supplemental_masks"

# Initialize Datasets
train_dataset = SIMDataset(image_dir=train_images, mask_dir=train_masks)
test_dataset = SIMDataset(image_dir=test_images, mask_dir=test_masks)

# Initialize Loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# 3. MODEL & LOSS
model = SIMD(depths=[64, 128, 256, 512, 1024])
model.to(device)  # <--- Move model to GPU

# Use BCEWithLogitsLoss for Binary Segmentation (Forged vs Real)
# This combines Sigmoid + BCE Loss in one numerically stable step.
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    params=model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
)

best_val = 0.0
# 4. TRAINING LOOP
for epoch in range(cfg.epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    # Progress bar for training
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

    for i, batch in loop:
        image, target_mask = batch

        # Move data to GPU and ensure correct types
        image = image.to(device).float()
        target_mask = target_mask.to(device).float()

        # Zero gradients
        optimizer.zero_grad()

        # Enable autocast to reduce memory
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Forward
            pred_mask = model(image)

            # Loss calculation
            loss = binary_diceCE(pred_mask, target_mask, dice_weight=0.5, ce_weight=0.5)

        # Backward & Step
        loss.backward()
        optimizer.step()

        # Logging
        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch + 1}/{cfg.epochs}]")
        loop.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)

    # 5. VALIDATION LOOP
    # Check performance on unseen data every epoch
    model.eval()  # Set to evaluation mode (disable dropout/batchnorm updates)
    val_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation for speed
        for batch in test_loader:
            image, target_mask = batch
            image = image.to(device).float()
            target_mask = target_mask.to(device).float()

            pred_mask = model(image)
            loss = criterion(pred_mask, target_mask)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(test_loader)

    if avg_val_loss > best_val:
        torch.save(model.state_dict(), "best.pth")
        best_val = avg_val_loss
        print("Saved best model with validation loss: ", best_val)

    print(
        f"Epoch {epoch + 1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
    )

torch.save(model.state_dict(), "two_headed.pth")
