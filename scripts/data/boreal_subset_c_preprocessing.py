"""
Boreal Subset-C preprocessing utilities for PyTorch.

What this script does:
1) Loads Boreal-Forest-Fire Subset-C train/valid/test split directories.
2) Resizes RGB images to 254x254 and normalizes with ImageNet stats.
3) Loads segmentation masks as binary single-channel tensors (0/1).
4) Optionally adds No-Fire images from a separate clean UAV directory.
5) Rebalances train split toward a target Fire ratio (default: 0.63).
6) Creates DataLoaders for classification and/or segmentation experiments.

Expected Subset-C layout:
Boreal-Forest-Fire-Subset-C/
  images/
    train/
    valid/
    test/
  sam_masks/ or manual_masks/
    train/
    valid/
    test/

Examples:
  python boreal_subset_c_preprocessing.py --show-summary

  python boreal_subset_c_preprocessing.py \
      --no-fire-root frames/Test/No_Fire \
      --task both \
      --show-summary
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Resize to FLAME-matching resolution requested by user.
TARGET_SIZE = (254, 254)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_image_transform(train: bool = False) -> transforms.Compose:
    """Build RGB transform pipeline with optional training augmentation."""

    if train:
        return transforms.Compose(
            [
                transforms.Resize(TARGET_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_mask_transform() -> transforms.Compose:
    """Resize masks with nearest-neighbor to preserve binary boundaries."""

    return transforms.Compose(
        [
            transforms.Resize(TARGET_SIZE, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ]
    )


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int
    mask_path: Optional[Path] = None


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _resolve_mask(mask_dir: Optional[Path], image_name: str) -> Optional[Path]:
    """Resolve mask file by trying common naming conventions."""

    if mask_dir is None or not mask_dir.exists():
        return None

    stem = Path(image_name).stem
    candidates = [
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}_mask.png",
        mask_dir / f"{stem}.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def collect_subset_c_samples(
    split_img_dir: Path,
    split_mask_dir: Optional[Path],
    no_fire_dir: Optional[Path] = None,
) -> List[Sample]:
    """Collect Fire samples from Subset-C and optionally append No-Fire samples."""

    samples: List[Sample] = []

    # Subset-C images are Fire by design.
    for img_path in _list_images(split_img_dir):
        mask_path = _resolve_mask(split_mask_dir, img_path.name)
        samples.append(Sample(image_path=img_path, label=1, mask_path=mask_path))

    # Optional external No-Fire images.
    if no_fire_dir is not None and no_fire_dir.exists():
        for img_path in _list_images(no_fire_dir):
            samples.append(Sample(image_path=img_path, label=0, mask_path=None))

    return samples


def rebalance_samples(samples: Sequence[Sample], target_fire_ratio: float, seed: int = 42) -> List[Sample]:
    """
    Subsample majority class to approximate requested Fire ratio.

    Returns original samples if target cannot be achieved due to class scarcity.
    """

    if not samples:
        return []

    fire = [s for s in samples if s.label == 1]
    no_fire = [s for s in samples if s.label == 0]

    if not fire or not no_fire:
        return list(samples)

    if not (0.0 < target_fire_ratio < 1.0):
        raise ValueError("target_fire_ratio must be in (0, 1)")

    rng = random.Random(seed)

    desired_no_fire = int(round(len(fire) * (1.0 - target_fire_ratio) / target_fire_ratio))
    desired_no_fire = max(1, desired_no_fire)

    if desired_no_fire >= len(no_fire):
        # Not enough No-Fire examples; keep all.
        selected_no_fire = no_fire
    else:
        selected_no_fire = rng.sample(no_fire, desired_no_fire)

    balanced = fire + selected_no_fire
    rng.shuffle(balanced)
    return balanced


class BorealDataset(Dataset):
    """Dataset for classification-only or joint segmentation+classification."""

    def __init__(
        self,
        samples: Sequence[Sample],
        image_transform: transforms.Compose,
        mask_transform: Optional[transforms.Compose] = None,
        return_mask: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.return_mask = return_mask

    def __len__(self) -> int:
        return len(self.samples)

    def _load_binary_mask(self, mask_path: Optional[Path]) -> torch.Tensor:
        """Load binary mask as shape [1, H, W] with values in {0, 1}."""

        if mask_path is None or not mask_path.exists():
            return torch.zeros((1, TARGET_SIZE[0], TARGET_SIZE[1]), dtype=torch.float32)

        mask_img = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_img, dtype=np.uint8)
        mask_np = np.where(mask_np > 128, 255, 0).astype(np.uint8)
        bin_mask_img = Image.fromarray(mask_np, mode="L")

        if self.mask_transform is None:
            mask_t = transforms.ToTensor()(bin_mask_img)
        else:
            mask_t = self.mask_transform(bin_mask_img)

        return (mask_t > 0.5).float()

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = Image.open(sample.image_path).convert("RGB")
        image_t = self.image_transform(image)
        label_t = torch.tensor(sample.label, dtype=torch.long)

        if self.return_mask:
            mask_t = self._load_binary_mask(sample.mask_path)
            return image_t, mask_t, label_t

        return image_t, label_t


def create_split_datasets(
    subset_root: Path,
    mask_root_name: str = "sam_masks",
    no_fire_root: Optional[Path] = None,
    target_fire_ratio: float = 0.63,
    task: str = "both",
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Create train/valid/test datasets from Subset-C with optional balancing."""

    images_root = subset_root / "images"
    masks_root = subset_root / mask_root_name

    split_map = {
        "train": (images_root / "train", masks_root / "train"),
        "valid": (images_root / "valid", masks_root / "valid"),
        "test": (images_root / "test", masks_root / "test"),
    }

    samples_by_split = {}
    for split_name, (img_dir, mask_dir) in split_map.items():
        split_no_fire = no_fire_root / split_name if no_fire_root is not None else None
        samples = collect_subset_c_samples(img_dir, mask_dir, split_no_fire)

        # Balance only training split by default.
        if split_name == "train" and no_fire_root is not None:
            samples = rebalance_samples(samples, target_fire_ratio=target_fire_ratio, seed=seed)

        samples_by_split[split_name] = samples

    return_mask = task in {"segmentation", "both"}

    train_ds = BorealDataset(
        samples_by_split["train"],
        image_transform=build_image_transform(train=True),
        mask_transform=build_mask_transform(),
        return_mask=return_mask,
    )
    valid_ds = BorealDataset(
        samples_by_split["valid"],
        image_transform=build_image_transform(train=False),
        mask_transform=build_mask_transform(),
        return_mask=return_mask,
    )
    test_ds = BorealDataset(
        samples_by_split["test"],
        image_transform=build_image_transform(train=False),
        mask_transform=build_mask_transform(),
        return_mask=return_mask,
    )

    return train_ds, valid_ds, test_ds


def create_dataloaders(
    train_ds: Dataset,
    valid_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap datasets with PyTorch DataLoaders."""

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader


def summarize_dataset(ds: BorealDataset, split_name: str) -> None:
    labels = [s.label for s in ds.samples]
    fire = sum(labels)
    total = len(labels)
    no_fire = total - fire
    ratio = (fire / total) if total else 0.0
    print(f"[{split_name}] total={total}, fire={fire}, no_fire={no_fire}, fire_ratio={ratio:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Boreal-Forest-Fire Subset-C for PyTorch training")
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=Path("Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-C"),
        help="Path to Boreal-Forest-Fire-Subset-C",
    )
    parser.add_argument(
        "--mask-root-name",
        type=str,
        default="sam_masks",
        choices=["sam_masks", "manual_masks"],
        help="Mask root to use inside subset root",
    )
    parser.add_argument(
        "--no-fire-root",
        type=Path,
        default=None,
        help=(
            "Optional root with train/valid/test No-Fire folders. "
            "Example: path/to/no_fire_root/train/*.jpg"
        ),
    )
    parser.add_argument("--target-fire-ratio", type=float, default=0.63, help="Target Fire ratio for train split")
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "segmentation", "both"],
        default="both",
        help="Return mode from dataset",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balancing")
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Print split sizes, class ratios, and one batch shape check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_ds, valid_ds, test_ds = create_split_datasets(
        subset_root=args.subset_root,
        mask_root_name=args.mask_root_name,
        no_fire_root=args.no_fire_root,
        target_fire_ratio=args.target_fire_ratio,
        task=args.task,
        seed=args.seed,
    )

    train_loader, valid_loader, test_loader = create_dataloaders(
        train_ds,
        valid_ds,
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.show_summary:
        summarize_dataset(train_ds, "train")
        summarize_dataset(valid_ds, "valid")
        summarize_dataset(test_ds, "test")

        batch = next(iter(train_loader))
        if args.task in {"segmentation", "both"}:
            images, masks, labels = batch
            print(f"train batch image shape: {tuple(images.shape)}")
            print(f"train batch mask shape:  {tuple(masks.shape)}")
            print(f"train batch label shape: {tuple(labels.shape)}")
        else:
            images, labels = batch
            print(f"train batch image shape: {tuple(images.shape)}")
            print(f"train batch label shape: {tuple(labels.shape)}")

    # Keep references alive and explicit for notebook imports.
    _ = (train_loader, valid_loader, test_loader)


if __name__ == "__main__":
    main()
