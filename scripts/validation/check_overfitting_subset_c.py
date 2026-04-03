"""Run an overfitting check on Boreal-Forest-Fire Subset-C."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet_segmentation import UltraOptimizedFireNet
import scripts.data.boreal_subset_c_preprocessing as boreal_utils

from scripts.data.boreal_subset_c_preprocessing import BorealDataset, collect_subset_c_samples


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(image_size: int) -> transforms.Compose:
    resize = transforms.Resize((image_size, image_size))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def transform(image):
        image = resize(image)
        rgb = to_tensor(image)
        rgb_norm = normalize(rgb)

        thermal = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        thermal = (thermal - 0.5) / 0.5

        return torch.cat([rgb_norm, thermal.unsqueeze(0)], dim=0)

    return transform


def build_mask_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.Resampling.NEAREST),
            transforms.ToTensor(),
        ]
    )


def dice_loss_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.reshape(probs.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (probs * target).sum(dim=1)
    denominator = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def build_split_dataset(subset_root: Path, split: str, image_size: int, max_samples: int | None) -> BorealDataset:
    boreal_utils.TARGET_SIZE = (image_size, image_size)
    image_dir = subset_root / "images" / split
    mask_dir = subset_root / "sam_masks" / split
    samples = collect_subset_c_samples(image_dir, mask_dir, no_fire_dir=None)
    if max_samples is not None:
        samples = samples[:max_samples]
    return BorealDataset(
        samples=samples,
        image_transform=build_image_transform(image_size),
        mask_transform=build_mask_transform(image_size),
        return_mask=True,
    )


def evaluate_split(model: torch.nn.Module, dataset: BorealDataset, device: torch.device, batch_size: int) -> Dict[str, float]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_batches = len(loader)

    total_samples = 0
    cls_correct = 0
    total_loss = 0.0
    cls_loss_sum = 0.0
    seg_loss_sum = 0.0
    seg_ce_sum = 0.0
    dice_sum = 0.0
    bce_sum = 0.0

    intersection = torch.zeros(2, dtype=torch.float64)
    union = torch.zeros(2, dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        for batch_index, (images, masks, labels) in enumerate(loader, start=1):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(images)
            cls_logits = outputs["classification"]
            seg_logits = outputs["segmentation"]

            cls_loss = F.cross_entropy(cls_logits, labels)
            seg_target = masks.squeeze(1).long()
            seg_ce = F.cross_entropy(seg_logits, seg_target)

            seg_probs = torch.softmax(seg_logits, dim=1)[:, 1]
            seg_float = seg_target.float()
            dice = dice_loss_from_probs(seg_probs, seg_float)
            bce = F.binary_cross_entropy(seg_probs, seg_float)
            seg_loss = 0.5 * dice + 0.5 * bce
            batch_loss = 0.25 * cls_loss + 3.2 * seg_loss

            cls_pred = cls_logits.argmax(dim=1)
            pred_mask = seg_logits.argmax(dim=1)

            cls_correct += (cls_pred == labels).sum().item()
            total_samples += labels.shape[0]
            total_loss += batch_loss.item() * labels.shape[0]
            cls_loss_sum += cls_loss.item() * labels.shape[0]
            seg_loss_sum += seg_loss.item() * labels.shape[0]
            seg_ce_sum += seg_ce.item() * labels.shape[0]
            dice_sum += dice.item() * labels.shape[0]
            bce_sum += bce.item() * labels.shape[0]

            for class_id in (0, 1):
                pred_pixels = pred_mask == class_id
                target_pixels = seg_target == class_id
                intersection[class_id] += (pred_pixels & target_pixels).sum().item()
                union[class_id] += (pred_pixels | target_pixels).sum().item()

            if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
                print(f"  processed {batch_index}/{total_batches} batches", flush=True)

    iou_per_class = []
    for class_id in (0, 1):
        if union[class_id] == 0:
            iou_per_class.append(1.0)
        else:
            iou_per_class.append((intersection[class_id] / union[class_id]).item())

    return {
        "samples": float(total_samples),
        "accuracy": cls_correct / total_samples if total_samples else 0.0,
        "loss": total_loss / total_samples if total_samples else 0.0,
        "cls_loss": cls_loss_sum / total_samples if total_samples else 0.0,
        "seg_loss": seg_loss_sum / total_samples if total_samples else 0.0,
        "seg_ce": seg_ce_sum / total_samples if total_samples else 0.0,
        "dice_loss": dice_sum / total_samples if total_samples else 0.0,
        "bce_loss": bce_sum / total_samples if total_samples else 0.0,
        "miou": sum(iou_per_class) / len(iou_per_class),
        "no_fire_iou": iou_per_class[0],
        "fire_iou": iou_per_class[1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check overfitting on Boreal-Forest-Fire Subset-C")
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=Path(r"data/raw/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-C"),
        help="Subset-C root folder",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"models/trained/msbw-net_shared_backbone/best_dicta_model.pth"),
        help="Path to shared-backbone checkpoint",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Resize images and masks to this size")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Optional sample cap per split for a quick diagnostic; use 0 for a full sweep",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def format_metrics(name: str, metrics: Dict[str, float]) -> str:
    return (
        f"{name}: samples={int(metrics['samples'])}, "
        f"loss={metrics['loss']:.4f}, cls_loss={metrics['cls_loss']:.4f}, seg_loss={metrics['seg_loss']:.4f}, "
        f"acc={metrics['accuracy']:.4f}, mIoU={metrics['miou']:.4f}, fire_IoU={metrics['fire_iou']:.4f}, "
        f"no_fire_IoU={metrics['no_fire_iou']:.4f}"
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    max_samples = None if args.max_samples is None or args.max_samples <= 0 else args.max_samples

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    model = UltraOptimizedFireNet()
    model.load_state_dict(state_dict)
    model.to(device)

    splits = {
        "train": build_split_dataset(args.subset_root, "train", args.image_size, max_samples),
        "valid": build_split_dataset(args.subset_root, "valid", args.image_size, max_samples),
        "test": build_split_dataset(args.subset_root, "test", args.image_size, max_samples),
    }

    results = {name: evaluate_split(model, dataset, device, args.batch_size) for name, dataset in splits.items()}

    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")
    if max_samples is not None:
        print(f"sample_cap_per_split={max_samples}")
    print(format_metrics("train", results["train"]))
    print(format_metrics("valid", results["valid"]))
    print(format_metrics("test", results["test"]))

    train_loss = results["train"]["loss"]
    valid_loss = results["valid"]["loss"]
    train_miou = results["train"]["miou"]
    valid_miou = results["valid"]["miou"]
    print(f"loss_gap(valid-train)={valid_loss - train_loss:.4f}")
    print(f"miou_gap(train-valid)={train_miou - valid_miou:.4f}")

    if valid_loss > train_loss * 1.15 or train_miou - valid_miou > 0.05:
        print("overfitting_signal=strong")
    elif valid_loss > train_loss * 1.05 or train_miou - valid_miou > 0.02:
        print("overfitting_signal=moderate")
    else:
        print("overfitting_signal=low")


if __name__ == "__main__":
    main()