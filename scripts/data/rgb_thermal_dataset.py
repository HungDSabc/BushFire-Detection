"""Dataset utilities for paired RGB + thermal loading.

This module provides a PyTorch Dataset class for multimodal training with
paired RGB and thermal images. If thermal files are not available, it can
optionally synthesize a 1-channel pseudo-thermal image from RGB luminance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FlameRGBThermalDataset(Dataset):
    """Return paired RGB and thermal samples plus label.

    Expected dataframe columns by default:
    - rgb_fname: relative/absolute RGB image path
    - thermal_fname: relative/absolute thermal image path
    - label: class label as int-like value

    Parameters
    ----------
    df:
        Pandas DataFrame with file names and labels.
    rgb_root:
        Root directory for RGB images when rgb_fname is relative.
    thermal_root:
        Root directory for thermal images when thermal_fname is relative.
    transform_rgb:
        RGB transform. If both transform_rgb and transform_thermal are set,
        transform_rgb may optionally be a joint transform that accepts
        ``(rgb_image, thermal_image)`` and returns ``(rgb_image, thermal_image)``.
    transform_thermal:
        Thermal transform applied after optional joint transform.
    rgb_col:
        Dataframe column name for RGB file names/paths.
    thermal_col:
        Dataframe column name for thermal file names/paths.
    label_col:
        Dataframe column name for labels.
    generate_pseudo_thermal:
        If True and a thermal file is missing, synthesize thermal from RGB.
    """

    def __init__(
        self,
        df,
        rgb_root: str | Path,
        thermal_root: Optional[str | Path] = None,
        transform_rgb: Optional[Callable] = None,
        transform_thermal: Optional[Callable] = None,
        rgb_col: str = "rgb_fname",
        thermal_col: str = "thermal_fname",
        label_col: str = "label",
        generate_pseudo_thermal: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.rgb_root = Path(rgb_root)
        self.thermal_root = Path(thermal_root) if thermal_root is not None else None
        self.transform_rgb = transform_rgb
        self.transform_thermal = transform_thermal

        self.rgb_col = rgb_col
        self.thermal_col = thermal_col
        self.label_col = label_col
        self.generate_pseudo_thermal = generate_pseudo_thermal

        required_cols = {self.rgb_col, self.label_col}
        missing = required_cols.difference(self.df.columns)
        if missing:
            raise ValueError(f"Missing required dataframe columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, root: Optional[Path], value: str) -> Optional[Path]:
        if value is None or str(value).strip() == "":
            return None

        path = Path(str(value))
        if path.is_absolute():
            return path

        if root is None:
            return path

        return root / path

    def _load_rgb(self, rgb_path: Path) -> Image.Image:
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        return Image.open(rgb_path).convert("RGB")

    def _load_thermal_or_pseudo(self, row, rgb_image: Image.Image) -> Image.Image:
        thermal_value = row[self.thermal_col] if self.thermal_col in row.index else None
        thermal_path = self._resolve_path(self.thermal_root, thermal_value)

        if thermal_path is not None and thermal_path.exists():
            return Image.open(thermal_path).convert("L")

        if not self.generate_pseudo_thermal:
            raise FileNotFoundError(
                "Thermal image not found and pseudo-thermal generation is disabled. "
                f"Value in column '{self.thermal_col}': {thermal_value}"
            )

        # Pseudo-thermal fallback: convert RGB luminance into a 1-channel image.
        gray = np.array(rgb_image.convert("L"), dtype=np.uint8)
        return Image.fromarray(gray, mode="L")

    def _apply_transforms(
        self,
        rgb: Image.Image,
        thermal: Image.Image,
    ) -> Tuple[object, object]:
        if self.transform_rgb is not None and self.transform_thermal is not None:
            # Try joint transform first: transform_rgb(rgb, thermal) -> (rgb, thermal)
            try:
                transformed = self.transform_rgb(rgb, thermal)
                if isinstance(transformed, tuple) and len(transformed) == 2:
                    rgb, thermal = transformed
                else:
                    rgb = self.transform_rgb(rgb)
            except TypeError:
                rgb = self.transform_rgb(rgb)

            thermal = self.transform_thermal(thermal)
            return rgb, thermal

        if self.transform_rgb is not None:
            rgb = self.transform_rgb(rgb)

        if self.transform_thermal is not None:
            thermal = self.transform_thermal(thermal)

        return rgb, thermal

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        rgb_path = self._resolve_path(self.rgb_root, row[self.rgb_col])
        if rgb_path is None:
            raise ValueError(f"Empty RGB path at index {idx}")

        rgb = self._load_rgb(rgb_path)
        thermal = self._load_thermal_or_pseudo(row, rgb)

        rgb, thermal = self._apply_transforms(rgb, thermal)

        label = int(row[self.label_col])
        return rgb, thermal, label
