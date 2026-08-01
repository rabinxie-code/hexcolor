from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .color import rgb24, rgb_to_hex


RGB = tuple[int, int, int]


@dataclass(frozen=True)
class ExtractionResult:
    """Normalized output shared by every extraction method."""

    method: str
    primary: RGB
    palette: tuple[RGB, ...]
    weights: tuple[float, ...]
    confidence: float
    route: str
    observed: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        palette = self.palette or (self.primary,)
        return {
            "method": self.method,
            "primary_rgb": list(self.primary),
            "rgb24": rgb24(self.primary),
            "hex": rgb_to_hex(self.primary),
            "palette_rgb": [list(color) for color in palette],
            "palette_hex": [rgb_to_hex(color) for color in palette],
            "weights": [round(float(weight), 6) for weight in self.weights],
            "confidence": round(float(self.confidence), 6),
            "route": self.route,
            "observed": self.observed,
            "diagnostics": _json_safe(self.diagnostics),
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return round(value, 6)
    return value
