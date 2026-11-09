"""Numeric LCD bounds for task09; independent of simulator state and timing."""

import math
import re


def readout_in_range(readout: str, bounds: tuple[float, float]) -> bool:
    """Accept a finite gram display within inclusive bounds; reject status text."""
    if not isinstance(readout, str):
        return False
    low, high = bounds
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        return False
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*g\s*", readout)
    if match is None:
        return False
    value = float(match.group(1))
    return math.isfinite(value) and low <= value <= high
