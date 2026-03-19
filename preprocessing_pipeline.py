"""Build and run a preprocessing/augmentation pipeline to disk.

This script defines the same resize/normalize/augment pipeline used in the notebooks,
and provides a small CLI for applying it to a folder of images and writing the
results to another folder.

Usage examples:
  python preprocessing_pipeline.py --input frames/segmentation/training/training/Fire \
      --output Output/preprocessed_fire --train
  python preprocessing_pipeline.py --input frames/segmentation/training/training/No_Fire \
      --output Output/preprocessed_no_fire --val
    python preprocessing_pipeline.py --input frames/segmentation/training/training/Fire \
            --output Output/preprocessed_fire --all-output Output/preprocessed_all --train
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(img_size: int = 224, train: bool = True) -> T.Compose:
    """Return a torchvision transform pipeline for FLAME preprocessing."""

    if train:
        return T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def apply_transform_to_dir(
    src_dir: Path,
    dst_dir: Path,
    transform: T.Compose,
    limit: Optional[int] = None,
    all_output_dir: Optional[Path] = None,
):
    """Apply the transform to all images in src_dir and save outputs to dst_dir."""

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if all_output_dir is not None:
        all_output_dir = Path(all_output_dir)
        all_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if limit is not None:
        image_paths = image_paths[:limit]

    source_tag = src_dir.name.replace(' ', '_')

    saved = 0
    skipped = 0

    try:
        for i, p in enumerate(image_paths, start=1):
            try:
                with Image.open(p) as img:
                    img = img.convert('RGB')
                    out_t = transform(img)

                # Save as PNG (normalize -> tensor -> [0,1] float -> uint8)
                out_arr = out_t.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
                out_img = Image.fromarray(out_arr)
                out_name = f'{p.stem}_preproc_{i:04d}.png'
                out_img.save(dst_dir / out_name)

                if all_output_dir is not None:
                    # Prefix with source folder to avoid collisions across class folders.
                    out_img.save(all_output_dir / f'{source_tag}_{out_name}')

                saved += 1
                if i % 250 == 0:
                    print(f'Processed {i}/{len(image_paths)} from {src_dir.name}...')
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                skipped += 1
                print(f'Skipping unreadable image: {p.name} ({exc})')
    except KeyboardInterrupt:
        print(f'Interrupted by user while processing {src_dir.name}.')

    print(f'Saved {saved} preprocessed samples to: {dst_dir.resolve()}')
    if all_output_dir is not None:
        print(f'Also appended {saved} samples to: {all_output_dir.resolve()}')
    if skipped:
        print(f'Skipped {skipped} unreadable images in: {src_dir.resolve()}')

    return saved


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Apply preprocessing pipeline to an image folder')
    parser.add_argument('--input', '-i', type=Path, required=False, default=None, help='Input image folder (manual mode)')
    parser.add_argument('--output', '-o', type=Path, required=False, default=None, help='Output folder for preprocessed images (manual mode)')
    parser.add_argument('--train', dest='train', action='store_true', help='Use training (augmented) pipeline')
    parser.add_argument('--val', dest='train', action='store_false', help='Use validation (deterministic) pipeline')
    parser.add_argument(
        '--all-output',
        type=Path,
        default=None,
        help='Optional merged output folder that receives copies of all preprocessed images',
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=None,
        help='Dataset root for auto mode (contains class subfolders like Fire/No_Fire)',
    )
    parser.add_argument('--size', type=int, default=224, help='Output image size (default: 224)')
    parser.add_argument('--limit', type=int, default=None, help='Max number of images to process')
    parser.set_defaults(train=True)
    return parser.parse_args()


def _resolve_default_dataset_root() -> Optional[Path]:
    """Return the first existing known training root path."""

    candidates = [
        Path('frames/Segmentation/Training/Training'),
        Path('frames/segmentation/training/training'),
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


def _run_auto_mode(args):
    """Process all class subfolders when no manual input/output are provided."""

    dataset_root = Path(args.dataset_root) if args.dataset_root is not None else _resolve_default_dataset_root()
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(
            'Could not find dataset root automatically. Use --dataset-root or provide --input and --output.'
        )

    transform = build_transform(img_size=args.size, train=args.train)
    by_class_root = Path('Output/preprocessed_by_class')
    all_output_dir = Path(args.all_output) if args.all_output is not None else Path('Output/preprocessed_all')

    class_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f'No class folders found under dataset root: {dataset_root.resolve()}')

    total = 0
    for class_dir in class_dirs:
        class_out = by_class_root / class_dir.name
        saved = apply_transform_to_dir(
            class_dir,
            class_out,
            transform,
            limit=args.limit,
            all_output_dir=all_output_dir,
        )
        total += saved

    print(f'Auto mode complete. Total preprocessed images: {total}')
    print(f'Class outputs: {by_class_root.resolve()}')
    print(f'Merged output: {all_output_dir.resolve()}')

    pipeline_path = all_output_dir / 'transform_pipeline.pt'
    torch.save(transform, pipeline_path)
    print(f'Saved transform pipeline to: {pipeline_path.resolve()}')


if __name__ == '__main__':
    args = _parse_args()

    # Manual mode: both --input and --output are provided.
    if args.input is not None or args.output is not None:
        if args.input is None or args.output is None:
            print('error: manual mode requires both --input/-i and --output/-o', file=sys.stderr)
            sys.exit(2)

        transform = build_transform(img_size=args.size, train=args.train)
        apply_transform_to_dir(
            args.input,
            args.output,
            transform,
            limit=args.limit,
            all_output_dir=args.all_output,
        )

        # Save the pipeline itself for reuse (torch can pickle torchvision transforms)
        pipeline_path = Path(args.output) / 'transform_pipeline.pt'
        torch.save(transform, pipeline_path)
        print(f'Saved transform pipeline to: {pipeline_path.resolve()}')
    else:
        _run_auto_mode(args)
