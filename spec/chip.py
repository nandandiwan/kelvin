"""ChipSpec — the single declarative object mesh/build.py, physics/, and
post/ are all derived from. Pure data + validation, no dolfinx dependency.
"""

from dataclasses import dataclass, field
from typing import List

from .materials import get as get_material
from .stack import LayerStack, lod1_stack
from .layout import Layout, default_layout


@dataclass(frozen=True)
class BoundaryConditions:
    ambient_t_k: float = 300.0
    backside_h_eff: float = 20000.0   # W/m^2/K, lumped TIM + Cu spreader + sink
    top_face: str = "adiabatic"       # "adiabatic" | "c4_flip_chip"


# Fraction of a device's dissipated power attributed to the channel (hot-carrier
# scattering in the velocity-saturated, high-field region toward the drain end)
# vs. the S/D contact (I^2*R heating across the metal-silicide/Si junction).
# Channel dominates in a bulk-planar device at these power levels; contact
# resistance is the smaller but real second term the plan explicitly wants
# modeled as its own path, not folded into the channel number.
CHANNEL_POWER_FRAC = 0.7
CONTACT_POWER_FRAC = 0.3


@dataclass(frozen=True)
class SourceBox:
    """One device's volumetric power source. `kind` is 'channel' (hot-carrier
    dissipation, drain end of the channel/inversion layer) or 'contact'
    (parasitic resistance heating at the S/D silicide contact).
    """
    device: str
    kind: str
    x_um: float
    y_um: float
    z0_um: float
    w_um: float
    l_um: float
    t_um: float
    power_uw: float

    @property
    def power_density_w_per_m3(self) -> float:
        vol_m3 = (self.w_um * self.l_um * self.t_um) * 1e-18
        return (self.power_uw * 1e-6) / vol_m3


@dataclass
class ChipSpec:
    stack: LayerStack
    layout: Layout
    bcs: BoundaryConditions
    sources: List[SourceBox] = field(default_factory=list)

    def validate(self) -> "ChipSpec":
        for l in self.stack.layers:
            get_material(l.material)

        t = self.layout.tile_um
        for d in self.layout.devices:
            if not (0 <= d.x_um <= t and 0 <= d.y_um <= t):
                raise ValueError(f"device {d.name} at ({d.x_um},{d.y_um}) outside {t}x{t} tile")

        source_layer = {"channel": "STI_channel", "contact": "silicide"}
        z_by_layer = {l.name: (z0, z1) for z0, z1, l in self.stack.z_bounds()}
        device_names = {d.name for d in self.layout.devices}
        eps = 1e-9
        for s in self.sources:
            if s.device not in device_names:
                raise ValueError(f"source references unknown device {s.device!r}")
            if s.kind not in source_layer:
                raise ValueError(f"source {s.device} has unknown kind {s.kind!r}")
            z0, z1 = z_by_layer[source_layer[s.kind]]
            if not (z0 - eps <= s.z0_um and s.z0_um + s.t_um <= z1 + eps):
                raise ValueError(
                    f"{s.kind} source {s.device} z-range outside {source_layer[s.kind]} layer"
                )
        return self

    @property
    def total_power_uw(self) -> float:
        return self.layout.total_power_uw


def _device_heat_sources(stack: LayerStack, layout: Layout) -> List[SourceBox]:
    """Two source boxes per device: the channel term (majority of the power,
    hot-carrier dissipation toward the drain end of the channel) and the
    contact term (S/D silicide contact resistance heating). See
    CHANNEL_POWER_FRAC/CONTACT_POWER_FRAC above for the split and its
    justification.
    """
    channel_z0 = stack.find("STI_channel")
    z0_channel = next(z0 for z0, z1, l in stack.z_bounds() if l is channel_z0)
    t_channel = channel_z0.thickness_um

    silicide = stack.find("silicide")
    z0_silicide = next(z0 for z0, z1, l in stack.z_bounds() if l is silicide)
    t_silicide = silicide.thickness_um

    out = []
    for d in layout.devices:
        # Channel: small footprint offset toward the drain end, full channel-
        # layer thickness (the layer itself is already the ~10nm inversion film).
        out.append(SourceBox(
            device=d.name, kind="channel",
            x_um=d.x_um + 0.02, y_um=d.y_um,
            z0_um=z0_channel,
            w_um=0.15, l_um=0.02, t_um=t_channel,
            power_uw=d.power_uw * CHANNEL_POWER_FRAC,
        ))
        # Contact: sits at the S/D silicide cap, wider footprint (contact pad)
        # but lower volumetric density than the channel term.
        out.append(SourceBox(
            device=d.name, kind="contact",
            x_um=d.x_um + 0.05, y_um=d.y_um,
            z0_um=z0_silicide,
            w_um=0.10, l_um=0.10, t_um=t_silicide,
            power_uw=d.power_uw * CONTACT_POWER_FRAC,
        ))
    return out


def default_chip() -> ChipSpec:
    """LOD-1: the 24x24 um, ~30-device tile from PLAN.md, ready for mesh/build.py."""
    stack = lod1_stack()
    layout = default_layout()
    sources = _device_heat_sources(stack, layout)
    chip = ChipSpec(stack=stack, layout=layout, bcs=BoundaryConditions(), sources=sources)
    return chip.validate()
