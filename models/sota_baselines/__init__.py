"""
SOTA Baseline Models for FLAME Wildfire Detection

This package contains implementations of state-of-the-art baseline models
for fair comparison with DICTA submission models.

Models:
- UNet-Fire 2022: Lightweight wildfire-optimized U-Net variant
"""

from .unet_fire_2022 import UNetFire2022, create_unet_fire_2022
from .thermalfusion_net_2023 import ThermalFusionNet2023, create_thermalfusion_net_2023
from .yolofm_2024 import YOLOFM2024, create_yolofm_2024, FocalSIoULoss

__all__ = [
    "UNetFire2022",
    "create_unet_fire_2022",
    "ThermalFusionNet2023",
    "create_thermalfusion_net_2023",
    "YOLOFM2024",
    "create_yolofm_2024",
    "FocalSIoULoss",
]
