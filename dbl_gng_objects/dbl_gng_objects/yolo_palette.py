"""YOLO Palette and Class Inspection Module for LTC Mapping.

Encapsulates the official Ultralytics YOLOv11 color palette and dynamic
model metadata extraction. Ensures strict color consistency across 2D mask overlays,
3D Point Cloud RGB fields, and RViz2 visualization markers.

Source of color mapping:
  ultralytics.utils.plotting.Colors().palette
  Initial hex definitions:
    #042AFF, #0BDBEB, #F3F3F3, #00DFB7, #111F68,
    #FF6FDD, #FF444F, #CCED00, #00F344, ...
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# Official Ultralytics YOLO Color Palette (RGB 0-255)
ULTRALYTICS_HEX_COLORS: Tuple[str, ...] = (
    "042AFF",  # Class 0: chair    -> RGB (4, 42, 255)
    "0BDBEB",  # Class 1: desk     -> RGB (11, 219, 235)
    "F3F3F3",  # Class 2: keyboard -> RGB (243, 243, 243)
    "00DFB7",  # Class 3: laptop   -> RGB (0, 223, 183)
    "1E64FF",  # Class 4: locker   -> RGB (30, 100, 255) (Vibrant Royal Blue)
    "FF6FDD",  # Class 5: monitor  -> RGB (255, 111, 221)
    "FF444F",  # Class 6: mouse    -> RGB (255, 68, 79)
    "CCED00",  # Class 7: person   -> RGB (204, 237, 0)
    "00F344",  # Class 8: tv       -> RGB (0, 243, 68)
    "BD00FF",  # Class 9
    "00B4FF",  # Class 10
    "DD00BA",  # Class 11
    "00FFFF",  # Class 12
    "26C000",  # Class 13
    "01FFB3",  # Class 14
    "7D24FF",  # Class 15
    "7B0068",  # Class 16
    "FF1B6C",  # Class 17
    "FC6D2F",  # Class 18
    "A2FF0B",  # Class 19
)


def _hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """Convert hex string (e.g. '042AFF') to (R, G, B) tuple."""
    hex_clean = hex_code.lstrip("#")
    return (
        int(hex_clean[0:2], 16),
        int(hex_clean[2:4], 16),
        int(hex_clean[4:6], 16),
    )


ULTRALYTICS_PALETTE_RGB: Tuple[Tuple[int, int, int], ...] = tuple(
    _hex_to_rgb(h) for h in ULTRALYTICS_HEX_COLORS
)

# Semantic Mapping Color Palette matching user reference image media_1791727282051.png
# (Desk: Red, Chair: Sky Blue, Monitor: Green, Laptop: Pink, Keyboard: Dark Green, Locker: Purple, Person: Yellow, TV: Orange)
DENGLER_HEX_COLORS: Tuple[str, ...] = (
    "0AA2F8",  # Class 0: chair    -> RGB (10, 162, 248) (Sky Blue / Cyan-Blue)
    "E11414",  # Class 1: desk     -> RGB (225, 20, 20)  (Pure Solid Red)
    "0A5000",  # Class 2: keyboard -> RGB (10, 80, 0)    (Dark Forest Green)
    "FF00B0",  # Class 3: laptop   -> RGB (255, 0, 176)  (Vibrant Pink / Magenta)
    "8C3CDC",  # Class 4: locker   -> RGB (140, 60, 220) (Vibrant Royal Purple)
    "13A221",  # Class 5: monitor  -> RGB (19, 162, 33)  (Vibrant Green)
    "141E50",  # Class 6: mouse    -> RGB (20, 30, 80)   (Deep Navy Blue)
    "F4CE10",  # Class 7: person   -> RGB (244, 206, 16) (Bright Warm Yellow)
    "FE7D00",  # Class 8: tv       -> RGB (254, 125, 0)  (Vibrant Orange)
    "BD00FF",  # Class 9
    "00B4FF",  # Class 10
    "DD00BA",  # Class 11
    "00FFFF",  # Class 12
    "26C000",  # Class 13
    "01FFB3",  # Class 14
    "7D24FF",  # Class 15
    "7B0068",  # Class 16
    "FF1B6C",  # Class 17
    "FC6D2F",  # Class 18
    "A2FF0B",  # Class 19
)

DENGLER_PALETTE_RGB: Tuple[Tuple[int, int, int], ...] = tuple(
    _hex_to_rgb(h) for h in DENGLER_HEX_COLORS
)

# Default verified classes from custom trained best_seg.pt
DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "chair",
    1: "desk",
    2: "keyboard",
    3: "laptop",
    4: "locker",
    5: "monitor",
    6: "mouse",
    7: "person",
    8: "tv",
}


def load_model_classes(model_path: str) -> Dict[int, str]:
    """Dynamically load and verify actual class names from the YOLO model metadata."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    from ultralytics import YOLO
    model = YOLO(model_path)
    if hasattr(model, "names") and isinstance(model.names, dict):
        return {int(k): str(v) for k, v in model.names.items()}
    return dict(DEFAULT_CLASS_NAMES)


class YoloPalette:
    """Manages color assignments and class metadata for LTC Mapping."""

    def __init__(
        self,
        class_names: Optional[Dict[int, str]] = None,
        palette_mode: str = "dengler"
    ):
        self.class_names: Dict[int, str] = dict(class_names or DEFAULT_CLASS_NAMES)
        self.name_to_id: Dict[str, int] = {v: k for k, v in self.class_names.items()}
        self.palette_mode = palette_mode
        if palette_mode == "ultralytics":
            self.palette_rgb = ULTRALYTICS_PALETTE_RGB
        else:
            self.palette_rgb = DENGLER_PALETTE_RGB
        self.n_colors = len(self.palette_rgb)

    def update_classes(self, new_classes: Dict[int, str]):
        """Update class names dictionary from verified model metadata."""
        self.class_names = dict(new_classes)
        self.name_to_id = {v: k for k, v in self.class_names.items()}

    def get_class_name(self, class_id: int) -> str:
        """Get class name string for class ID."""
        return self.class_names.get(int(class_id), f"class_{class_id}")

    def get_class_id(self, class_name: str) -> int:
        """Get class ID for class name string."""
        return self.name_to_id.get(class_name, 0)

    def _resolve_index(self, class_id_or_name: Union[int, str]) -> int:
        if isinstance(class_id_or_name, str):
            cid = self.name_to_id.get(class_id_or_name, 0)
        else:
            cid = int(class_id_or_name)
        return cid % self.n_colors

    def get_rgb(self, class_id_or_name: Union[int, str]) -> Tuple[int, int, int]:
        """Get RGB tuple (0-255)."""
        idx = self._resolve_index(class_id_or_name)
        return self.palette_rgb[idx]

    def get_bgr(self, class_id_or_name: Union[int, str]) -> Tuple[int, int, int]:
        """Get BGR tuple (0-255) for OpenCV."""
        r, g, b = self.get_rgb(class_id_or_name)
        return (b, g, r)

    def get_rgba_float(
        self, class_id_or_name: Union[int, str], alpha: float = 1.0
    ) -> Tuple[float, float, float, float]:
        """Get RGBA tuple (0.0-1.0) for RViz2 Marker."""
        r, g, b = self.get_rgb(class_id_or_name)
        return (r / 255.0, g / 255.0, b / 255.0, float(alpha))


# Singleton default palette instance
_DEFAULT_PALETTE = YoloPalette()


def get_rgb(class_id_or_name: Union[int, str]) -> Tuple[int, int, int]:
    return _DEFAULT_PALETTE.get_rgb(class_id_or_name)


def get_bgr(class_id_or_name: Union[int, str]) -> Tuple[int, int, int]:
    return _DEFAULT_PALETTE.get_bgr(class_id_or_name)


def get_rgba_float(
    class_id_or_name: Union[int, str], alpha: float = 1.0
) -> Tuple[float, float, float, float]:
    return _DEFAULT_PALETTE.get_rgba_float(class_id_or_name, alpha)


def pack_rgb_float(r: int, g: int, b: int) -> float:
    """Pack uint8 (R, G, B) into float32 for PointCloud2 'rgb' field."""
    rgb_u32 = (int(r) << 16) | (int(g) << 8) | int(b)
    return float(np.uint32(rgb_u32).view(np.float32))


def unpack_rgb_float(f: float) -> Tuple[int, int, int]:
    """Unpack float32 from PointCloud2 'rgb' field to (R, G, B) uint8."""
    rgb_u32 = int(np.float32(f).view(np.uint32))
    r = (rgb_u32 >> 16) & 0xFF
    g = (rgb_u32 >> 8) & 0xFF
    b = rgb_u32 & 0xFF
    return (r, g, b)
