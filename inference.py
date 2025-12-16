import argparse
import torch
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from utils.model import SIMD

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
args = parser.parse_args()

# --- 1. Load and Prepare Image ---
img = cv.imread(args.image)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (512, 512), interpolation=cv.INTER_CUBIC)
# img -> (512, 512, 3)

# Save original for visualization later
img_vis = img.copy()

# Convert to tensor
img_tensor = torch.from_numpy(img)
img_tensor = img_tensor.permute(2, 0, 1)  # img -> (3, 512, 512)
img_tensor = img_tensor.unsqueeze(0)  # img -> (1, 3, 512, 512)
img_tensor = img_tensor.float() / 255.0

# --- 2. Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SIMD(depths=[64, 128, 256, 512, 1024]).to(device)

model.load_state_dict(
    torch.load("saved_model.pth", map_location=device, weights_only=True)
)
model.eval()

# --- 3. Inference ---
img_tensor = img_tensor.to(device)

with torch.no_grad():
    logits = model(img_tensor)  # logits -> (1, 1, 512, 512) (Raw values)
    probs = torch.sigmoid(logits)  # probs  -> (1, 1, 512, 512) (0.0 to 1.0)

    # Create binary mask for visualization
    binary_mask = (probs > 0.5).float()  # mask   -> (1, 1, 512, 512)

# Convert to Numpy for plotting
pred_prob = probs.squeeze().cpu().numpy()  # pred_prob -> (512, 512)
pred_binary = binary_mask.squeeze().cpu().numpy()  # pred_binary -> (512, 512)

# --- 4. Visualization ---
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_vis)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(pred_prob, cmap="gray", vmin=0, vmax=1)
plt.title("Probability Map (Soft)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred_binary, cmap="gray", vmin=0, vmax=1)
plt.title("Binary Prediction (Hard)")
plt.axis("off")

plt.tight_layout()
plt.show()
