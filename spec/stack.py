"""Vertical layer stack (LOD 1) — pure data, no dolfinx dependency.

z = 0 at the Si backside, increasing toward the die top. Each Layer is a
named z-slab with a background material; devices, vias, and wires are
carved into it laterally by layout.py + mesh/build.py, not here.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .materials import get as get_material


@dataclass(frozen=True)
class Layer:
    name: str
    thickness_um: float
    material: str            # key into MATERIALS — background fill of this layer
    mesh_size_um: float      # target element size for this layer (grading hint)
    is_metal: bool = False   # hosts lateral Cu routing (wires carved out by layout.py)
    # Lateral override materials, carved at each device's footprint by
    # mesh/section.py — kept here (not in mesh/) because which material goes
    # where is chip physics, not geometry code.
    device_fill: Optional[str] = None  # every device gets this (transistor layers)
    via_fill: Optional[str] = None     # only devices with has_via_stack (BEOL layers)


@dataclass
class LayerStack:
    layers: List[Layer] = field(default_factory=list)

    def add(self, name, thickness_um, material, mesh_size_um, is_metal=False,
            device_fill=None, via_fill=None):
        get_material(material)
        if device_fill is not None:
            get_material(device_fill)
        if via_fill is not None:
            get_material(via_fill)
        self.layers.append(Layer(name, thickness_um, material, mesh_size_um, is_metal,
                                  device_fill, via_fill))
        return self

    @property
    def total_thickness_um(self) -> float:
        return sum(l.thickness_um for l in self.layers)

    def z_bounds(self):
        """[(z0, z1, layer), ...] cumulative from z=0 at the stack base."""
        z = 0.0
        out = []
        for l in self.layers:
            out.append((z, z + l.thickness_um, l))
            z += l.thickness_um
        return out

    def layer_at_z(self, z_um: float) -> Layer:
        for z0, z1, l in self.z_bounds():
            if z0 <= z_um < z1:
                return l
        raise ValueError(f"z={z_um} outside stack (total {self.total_thickness_um} um)")

    def find(self, name: str) -> Layer:
        for l in self.layers:
            if l.name == name:
                return l
        raise KeyError(f"no layer named {name!r}")


def lod1_stack() -> LayerStack:
    """The LOD-1 vertical stack from PLAN.md, Si backside (z=0) to passivation top.

    z = 50-50.26 resolves the transistor's own vertical anatomy instead of a
    single lumped "active" slab, because the z-direction gradient through the
    device (channel -> gate stack -> silicide -> MOL) is where most of the
    interesting thermal structure actually lives. Each of these layers has a
    `material` background (continues the STI oxide field between devices) and
    a `device_fill` that mesh/section.py carves in at every device footprint:

      50.00-50.03  STI_SD       junction-depth S/D silicon (Si_SD_doped) in
                                 STI oxide.
      50.03-50.04  STI_channel  few-nm channel/inversion layer (Si_channel)
                                 in STI oxide; this is where the *channel*
                                 heat-source sub-box gets carved (see chip.py).
      50.04-50.06  gate_dielectric   HfO2 gate stack over STI oxide field.
      50.06-50.11  gate_electrode    TiN/W poly-metal gate over STI oxide field.
      50.11-50.13  silicide      NiSi contact cap over STI oxide field; this is
                                 where the *contact* heat-source sub-box lives.
      50.13-50.26  MOL_contacts  W plugs over SiO2 field, up to M1.

    This is a simplified LOD1 proxy for real S/D-channel coplanarity (they're
    actually side by side along channel length, not z-stacked); true lateral
    resolution is deferred to the LOD2 zoom model.

    M1..V3 (50.26-51.86) keep a low-k background with a Cu_fine `via_fill`
    carved only at devices with `has_via_stack` — this is the asymmetric
    escape-path structure the plan is built around. M_top (~51.86-54.91) and
    passivation (+0.5) are unbroken.

    mesh_size_um is floored at roughly half each layer's own thickness so
    every layer gets at least ~2 elements through it — finer than that buys
    little at LOD1 and blows up element count for no reason.
    """
    s = LayerStack()
    s.add("Si_substrate",   50.00, "Si_bulk", mesh_size_um=5.0)
    s.add("STI_SD",           0.03, "SiO2", mesh_size_um=0.015, device_fill="Si_SD_doped")
    s.add("STI_channel",      0.01, "SiO2", mesh_size_um=0.005, device_fill="Si_channel")
    s.add("gate_dielectric",  0.02, "SiO2", mesh_size_um=0.01,  device_fill="HfO2")
    s.add("gate_electrode",   0.05, "SiO2", mesh_size_um=0.02,  device_fill="TiN")
    s.add("silicide",         0.02, "SiO2", mesh_size_um=0.01,  device_fill="NiSi")
    s.add("MOL_contacts",     0.13, "SiO2", mesh_size_um=0.03,  device_fill="W")
    s.add("M1",  0.10, "low_k", mesh_size_um=0.03, is_metal=True, via_fill="Cu_fine")
    s.add("V1",  0.10, "low_k", mesh_size_um=0.03, is_metal=True, via_fill="Cu_fine")
    s.add("M2",  0.25, "low_k", mesh_size_um=0.05, is_metal=True, via_fill="Cu_fine")
    s.add("V2",  0.25, "low_k", mesh_size_um=0.05, is_metal=True, via_fill="Cu_fine")
    s.add("M3",  0.45, "low_k", mesh_size_um=0.08, is_metal=True, via_fill="Cu_fine")
    s.add("V3",  0.45, "low_k", mesh_size_um=0.08, is_metal=True, via_fill="Cu_fine")
    s.add("M_top", 3.05, "Cu_thick", mesh_size_um=0.3, is_metal=True)
    s.add("passivation", 0.5, "SiN", mesh_size_um=0.1)
    return s
