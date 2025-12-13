import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
from pathlib import Path


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

        # Validate data only if we are in training mode (have masks)
        if self.is_train:
            self._validate_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        # 1. Load Image
        image_bgr = cv.imread(str(self.images[idx]))
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

        # 2. Generate Noise Stream
        # We pass the RGB image, but the helper calculates the intensity sum internally
        image_noise = self._apply_srm(image_rgb)

        # 3. Load Mask
        if self.is_train:
            mask = np.load(str(self.masks[idx]))
            mask = torch.from_numpy(mask).float()  # Ensure float for loss functions
        else:
            # Return a dummy mask or handle differently for inference
            mask = torch.zeros((image_rgb.shape[0], image_rgb.shape[1]))

        # 4. To Tensor
        # PyTorch expects (C, H, W), OpenCV gives (H, W, C).
        image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_noise = torch.from_numpy(image_noise).permute(2, 0, 1).float()

        return image_rgb, image_noise, mask

    def _apply_srm(self, image: np.ndarray):
        """Apply the three SRM kernels to the image."""

        # Kernel 1
        k1 = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        k1 = k1 / 4.0

        # Kernel 2
        k2 = np.array(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=np.float32,
        )
        k2 = k2 / 12.0

        # Kernel 3
        k3 = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        k3 = k3 / 2.0

        # Compute "Sum" Channel (mimicking the 5x5x3 convolution behavior)
        sum_image = np.sum(image, axis=2, dtype=np.float32)

        # Apply Filters
        # depth=-1 means the output will have the same depth as the source (float32)
        n1 = cv.filter2D(sum_image, -1, k1)
        n2 = cv.filter2D(sum_image, -1, k2)
        n3 = cv.filter2D(sum_image, -1, k3)

        # FIX 2: Use stack to create the 3rd dimension.
        # Result shape: (H, W, 3)
        noise_image = np.stack([n1, n2, n3], axis=-1)

        return noise_image

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
