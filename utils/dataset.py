import argparse
from pathlib import Path

import torch
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class SIMDataset(Dataset):
    def __init__(
        self, image_dir: Path | str, mask_dir: Path | str | None = None, transform=None
    ):
        """
        Dataset for image manipulation detection.
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform

        self.images = sorted(list(self.image_dir.glob("*.png")))

        if self.mask_dir:
            self.masks = sorted(list(self.mask_dir.glob("*.npy")))
        else:
            self.masks = []

        self.is_train = True if self.mask_dir else False

        # Validate data only if we are in training mode
        if self.is_train:
            self._validate_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        # 1. Load Image
        image_bgr = cv.imread(str(self.images[idx]))
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

        target_size = (512, 512)

        # Resize Image
        image_rgb = cv.resize(
            image_rgb, dsize=target_size, interpolation=cv.INTER_CUBIC
        )  # (512, 512, 3)

        # 2. Load Mask
        if self.is_train:
            mask = np.load(str(self.masks[idx]))  # (C, H, W)

            # A. Collapse channels
            if mask.ndim == 3:
                mask = np.max(mask, axis=0)  # (H, W)

            # B. Resize Mask
            mask = cv.resize(
                mask, dsize=target_size, interpolation=cv.INTER_NEAREST
            )  # (512, 512)

            # C. Add channel dimension back: (512, 512) -> (512, 512, 1)
            mask = np.expand_dims(mask, axis=-1)

            # D. To Tensor (512, 512, 1) -> (1, 512, 512)
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        else:
            # Return dummy mask for inference
            mask = torch.zeros((1, target_size[1], target_size[0]))

        # 3. To Tensor (512, 512, 3) -> (3, 512, 512)
        image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        return image_rgb, mask

    def _validate_data(self):
        if len(self.images) != len(self.masks):
            raise ValueError(
                f"Mismatch: {len(self.images)} images vs {len(self.masks)} masks."
            )

        # Simple name check
        img_names = [p.stem for p in self.images]
        msk_names = [p.stem for p in self.masks]

        if img_names != msk_names:
            raise ValueError("Image and mask filenames do not match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to visualize each training image"
    )
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    image_dir = Path(
        "/home/paschalis/code/SIMD/recodai-luc-scientific-image-forgery-detection/train_images/forged"
    )
    mask_dir = Path(
        "/home/paschalis/code/SIMD/recodai-luc-scientific-image-forgery-detection/train_masks"
    )

    train_dataset = SIMDataset(image_dir=image_dir, mask_dir=mask_dir)

    # 1. Get Data
    image_rgb, mask = train_dataset[args.index]

    # 2. Prepare RGB (Tensor -> Numpy, C,H,W -> H,W,C)
    vis_rgb = image_rgb.permute(1, 2, 0).numpy()

    # 3. Prepare Mask
    vis_mask = mask.numpy()
    vis_mask = vis_mask.transpose(1, 2, 0)

    # Plot RGB
    plt.subplot(1, 3, 1)
    plt.imshow(vis_rgb)
    plt.title("RGB Input")
    plt.axis("off")

    # Plot Mask Overlay
    plt.subplot(1, 3, 2)
    plt.imshow(vis_rgb)

    # Single channel mask (H, W)
    plt.imshow(vis_mask[:, :], cmap="jet", alpha=0.5)

    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
