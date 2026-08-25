"""Lateral layout: device array, via-stack assignment, thermal-via farm.

Pure data, no dolfinx dependency.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

TILE_UM = 24.0          # 24 x 24 um tile
ARRAY_COLS, ARRAY_ROWS = 6, 5
PITCH_X_UM, PITCH_Y_UM = 2.0, 3.0
ARRAY_ORIGIN_UM = (6.0, 4.5)   # lower-left corner of the ~12x15 um array inside the tile

HOT_POWER_UW = 20.0
COLD_POWER_UW = 2.0
HOT_CLUSTER = {(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)}  # 2x3 corner block ("hot ALU")

DEVICE_W_UM, DEVICE_L_UM = 0.8, 0.15   # active island footprint
VIA_WIDTH_UM = 0.3                     # M1->M_top via/wire column width, narrower than the device

VIA_FARM_SIZE_UM = 4.0
VIA_FARM_ORIGIN_UM = (19.0, 19.0)  # far corner from the hot cluster


@dataclass(frozen=True)
class Device:
    name: str
    col: int
    row: int
    x_um: float             # center, tile-local coords
    y_um: float
    power_uw: float
    has_via_stack: bool     # True: direct M1->M4 escape path; False: unbroken low-k above

    @property
    def is_hot(self) -> bool:
        return self.power_uw == HOT_POWER_UW


@dataclass
class Layout:
    tile_um: float = TILE_UM
    devices: List[Device] = field(default_factory=list)
    via_farm_origin_um: Tuple[float, float] = VIA_FARM_ORIGIN_UM
    via_farm_size_um: float = VIA_FARM_SIZE_UM

    @property
    def total_power_uw(self) -> float:
        return sum(d.power_uw for d in self.devices)

    @property
    def flux_w_per_cm2(self) -> float:
        area_cm2 = (self.tile_um * 1e-4) ** 2
        return (self.total_power_uw * 1e-6) / area_cm2


def default_layout() -> Layout:
    """30-device array: 2x3 hot ALU corner at 20 uW, rest at 2 uW; via-stack
    assigned on a checkerboard so half the devices (in both hot and cold
    populations) get a direct M1->M4 escape and half sit under unbroken
    low-k — PLAN.md's headline asymmetric-path case.
    """
    ox, oy = ARRAY_ORIGIN_UM
    devices = []
    for row in range(ARRAY_ROWS):
        for col in range(ARRAY_COLS):
            hot = (col, row) in HOT_CLUSTER
            power = HOT_POWER_UW if hot else COLD_POWER_UW
            has_via = (col + row) % 2 == 0
            devices.append(Device(
                name=f"dev_{col}_{row}",
                col=col, row=row,
                x_um=ox + col * PITCH_X_UM,
                y_um=oy + row * PITCH_Y_UM,
                power_uw=power,
                has_via_stack=has_via,
            ))
    return Layout(devices=devices)
