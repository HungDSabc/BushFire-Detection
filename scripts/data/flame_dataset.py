"""
Standardized PyTorch Dataset class for FLAME segmentation data.

This module provides a unified Dataset class for loading and preprocessing FLAME
segmentation images and masks. It ensures consistency across all SOTA model training
pipelines by providing:

- Image/mask loading from structured directories (Images/, Masks/)
- Configurable resizing (256x256 or 512x512)
- ImageNet normalization
- Training/validation split support
- Optional data augmentation

Usage:
    from flame_dataset import FLAMEDataset
    from torch.utils.data import DataLoader

    # Create dataset with 80/20 train/val split
    train_dataset = FLAMEDataset(
        root_dir="data/processed/Output/Segmentation_Augmented",
        split="train",
        img_size=256,
        train_ratio=0.8
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Validation dataset
    val_dataset = FLAMEDataset(
        root_dir="data/processed/Output/Segmentation_Augmented",
        split="val",
        img_size=256,
        train_ratio=0.8
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Literal
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# ImageNet normalization constants (consistent with existing preprocessing)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FLAMEDataset(Dataset):
    """
    PyTorch Dataset class for FLAME segmentation data.

    Loads paired RGB images (.jpg) and binary segmentation masks (.png) from
    structured directories. Supports flexible image sizes, augmentation, and
    train/validation splits.

    Attributes:
        root_dir (Path): Root directory containing Images/ and Masks/ subdirectories
        split (str): Dataset split - "train", "val", or "full"
        img_size (int): Target image size (256 or 512)
        train_ratio (float): Fraction of data to use for training (0.0 to 1.0)
        augment (bool): Whether to apply data augmentation (training only)
        image_files (list): Sorted list of image filenames for this split
        seed (int): Random seed for reproducible splits
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "full"] = "train",
        img_size: Literal[256, 512] = 256,
        train_ratio: float = 0.8,
        augment: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the FLAME Dataset.

        Args:
            root_dir: Path to directory containing Images/ and Masks/ subdirectories
            split: Dataset split - "train", "val", or "full"
                - "train": uses first (train_ratio * 100)% of images
                - "val": uses remaining images
                - "full": uses all images
            img_size: Target image size (256 or 512 pixels)
            train_ratio: Fraction of data for training (default: 0.8 for 80/20 split)
            augment: Apply augmentation for training split (default: True)
            seed: Random seed for reproducible splits (default: 42)

        Raises:
            ValueError: If img_size not in [256, 512] or train_ratio not in (0, 1)
            FileNotFoundError: If Images/ or Masks/ directories not found
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.train_ratio = train_ratio
        self.augment = augment and (split == "train")  # Only augment training
        self.seed = seed

        # Validation
        if img_size not in [256, 512]:
            raise ValueError(f"img_size must be 256 or 512, got {img_size}")
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

        # Check directory structure
        self.images_dir = self.root_dir / "Images"
        self.masks_dir = self.root_dir / "Masks"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        # Get all image files (sorted for reproducibility)
        all_images = sorted([f for f in self.images_dir.glob("*.jpg")])
        if not all_images:
            raise FileNotFoundError(f"No .jpg images found in {self.images_dir}")

        # Split into train/val if needed
        if split == "full":
            self.image_files = all_images
        else:
            # Use seed for reproducible splits
            np.random.seed(seed)
            indices = np.arange(len(all_images))
            np.random.shuffle(indices)

            split_idx = int(len(all_images) * train_ratio)
            if split == "train":
                selected_indices = indices[:split_idx]
            else:  # val
                selected_indices = indices[split_idx:]

            self.image_files = [all_images[i] for i in selected_indices]

        self.n_samples = len(self.image_files)
        self._verify_pairs()

        print(
            f"FLAME Dataset ({split}): {self.n_samples} samples, "
            f"size={img_size}x{img_size}, augment={self.augment}"
        )

    def _verify_pairs(self) -> None:
        """Verify that corresponding masks exist for all images."""
        missing = []
        for img_file in self.image_files:
            mask_file = self.masks_dir / img_file.stem / ".png"
            # Try both naming conventions
            mask_file_1 = self.masks_dir / (img_file.stem + ".png")
            mask_file_2 = self.masks_dir / (img_file.stem.replace(".jpg", "") + ".png")

            if not (mask_file_1.exists() or mask_file_2.exists()):
                missing.append((img_file.name, mask_file_1.name))

        if missing:
            warnings.warn(
                f"Missing {len(missing)} corresponding masks. "
                f"First missing: {missing[0]}"
            )

    def _get_mask_file(self, img_file: Path) -> Path:
        """Find the corresponding mask file for an image."""
        # Try naming convention 1: image_name.png
        mask_file = self.masks_dir / (img_file.stem + ".png")
        if mask_file.exists():
            return mask_file

        # Try naming convention 2: image_name_mask.png
        mask_file = self.masks_dir / (img_file.stem + "_mask.png")
        if mask_file.exists():
            return mask_file

        # If neither found, return default (will error when loading)
        return self.masks_dir / (img_file.stem + ".png")

    def _build_transforms(self) -> Tuple[T.Compose, T.Compose]:
        """
        Build transforms for images and masks.

        Returns:
            Tuple of (image_transform, mask_transform)
        """
        # Base transforms (applied to both image and mask)
        base_resize = [
            T.Resize(self.img_size, interpolation=Image.BILINEAR),
            T.CenterCrop(self.img_size),
        ]

        if self.augment:
            image_transforms = [
                T.RandomResizedCrop(
                    self.img_size, scale=(0.8, 1.0), interpolation=Image.BILINEAR
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        else:
            image_transforms = [
                *base_resize,
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]

        # Mask transform (no augmentation, no normalization)
        mask_transforms = [
            *base_resize,
            T.ToTensor(),
        ]

        image_transform = T.Compose(image_transforms)
        mask_transform = T.Compose(mask_transforms)

        return image_transform, mask_transform

    def __len__(self) -> int:
        """Return dataset size."""
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample (image-mask pair).

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys:
                - "image": Normalized RGB tensor (3, H, W)
                - "mask": Binary mask tensor (1, H, W)
                - "filename": Image filename for debugging
        """
        img_file = self.image_files[idx]
        mask_file = self._get_mask_file(img_file)

        # Load image and mask
        try:
            image = Image.open(img_file).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_file}: {e}")

        try:
            mask = Image.open(mask_file).convert("L")  # Grayscale
        except Exception as e:
            raise RuntimeError(f"Failed to load mask {mask_file}: {e}")

        # Build transforms
        image_transform, mask_transform = self._build_transforms()

        # Apply transforms
        image = image_transform(image)
        mask = mask_transform(mask)

        # Threshold mask to binary (0, 1)
        mask = (mask > 0.5).float()

        return {
            "image": image,
            "mask": mask,
            "filename": img_file.name,
        }


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: Literal[256, 512] = 256,
    train_ratio: float = 0.8,
    shuffle_train: bool = True,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for FLAME segmentation.

    Args:
        data_dir: Path to dataset root directory
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        img_size: Target image size (256 or 512)
        train_ratio: Train/val split ratio
        shuffle_train: Whether to shuffle training data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> train_loader, val_loader = create_dataloaders(
        ...     "data/processed/Output/Segmentation_Augmented",
        ...     batch_size=32,
        ...     num_workers=4
        ... )
    """
    train_dataset = FLAMEDataset(
        root_dir=data_dir,
        split="train",
        img_size=img_size,
        train_ratio=train_ratio,
        augment=True,
        seed=seed,
    )

    val_dataset = FLAMEDataset(
        root_dir=data_dir,
        split="val",
        img_size=img_size,
        train_ratio=train_ratio,
        augment=False,
        seed=seed,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    """Example usage."""
    data_dir = "data/processed/Output/Segmentation_Augmented"

    # Create datasets
    print("Creating FLAME datasets...")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        img_size=256,
        train_ratio=0.8,
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Show a sample
    print("\nSample batch contents:")
    batch = next(iter(train_loader))
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Mask shape: {batch['mask'].shape}")
    print(f"  Filenames: {batch['filename'][:2]}...")
    print(f"  Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    print(f"  Mask unique values: {torch.unique(batch['mask'])}")
