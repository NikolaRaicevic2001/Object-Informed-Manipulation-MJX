"""Turn the raw scans in `assets_original/` into run-ready `assets/` meshes.

Same output convention as `oim/models/xarm6_pusht_tabletop/assets`, which
`oim.objects.library` loads as `assets/{name}_centered.obj`:

    x, y = the footprint, its bounding box centred on the origin
    z    = height, the underside at exactly z = 0
    units: metres

WHAT THIS HAS TO FIX. The four sources disagree about everything:

    object      up axis   units            extent as shipped
    ----------  --------  ---------------  --------------------------
    coca_cola   +y        arbitrary        218.7 x 550.6 x 218.7
    coffee_cup  +y        arbitrary        2.78 x 1.98 x 1.98
    cup         +z        metres           0.057 x 0.057 x 0.062
    hammer      +z        metres           0.182 x 0.333 x 0.033

Each up axis was settled by RENDERING all three candidates and looking.
The obvious test -- pick the axis whose perpendicular footprint is round --
gets the coffee cup wrong: a mug whose height nearly equals its diameter
has a square side profile too, and that reading laid it on its side. The
hammer is also YAWED in its source frame -- its 0.182 x 0.333 footprint is
the diagonal of a 0.336 x 0.126 handle -- so it alone gets a yaw from the
footprint's principal axis, putting the handle on +x.

SCALES COME FROM THE MEASURED OBJECTS (2026-09-12), not from the scan: a
scan's proportions are not the proportions of the thing on the table. The
Coke bottle is 2.52 long as shipped against 3.07 measured, and the YCB
hammer is simply a different hammer (2.65 long against 3.71 measured).

Height is always scaled to the measured height. The FOOTPRINT is scaled
two different ways, because "length 7.5, width 7.5" on a mug means a
diameter, not a bounding box:

  xy_uniform  one factor for x and y, set by the NARROW footprint axis.
              A body of revolution stays circular, and a handle keeps its
              proportions and sticks out past the quoted width.
  per axis    x and y scaled independently onto the quoted length and
              width. The hammer only -- it is not round, and its bounding
              box is what FoundationPose and the collision boxes match.

`report()` prints every factor and the final extent, so neither the
distortion nor a footprint wider than quoted is ever silent.

TEXTURES. FoundationPose needs a textured mesh. The Coke bottle ships its
own UV map and photo; the other three are painted a flat colour, which is
also what makes them unambiguous to a pose estimator on a wooden table.

Requires `fast_simplification` (`uv pip install fast_simplification`) for
the coffee cup alone -- it arrives at 570k faces, ~35x every other asset
here. Run from the repo root:

    uv run python oim/models/xarm6_pusht_tabletop_real/prepare_objects.py
"""

import os
import shutil
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "assets")
# `oim.objects.library` is ONE registry every scene shares, and a scene
# resolves `assets/{name}_centered.obj` against its own directory -- so an
# entry is only loadable where its mesh exists. The mesh is mirrored here
# so `--object coca_cola` works on a sim scene too, which is also what
# `tests/test_objects_library.py` exercises (its scene list is the sim
# tabletop). Only the mesh: the material and texture serve FoundationPose,
# which only ever sees the real table.
MIRROR_DIR = os.path.normpath(
    os.path.join(HERE, "..", "xarm6_pusht_tabletop", "assets")
)


def _src(path: str) -> str:
    """A source path, relative to this scene directory.

    Not to `assets_original/`: the hammer's usable mesh is the one the SIM
    scene already ships, two directories up.
    """
    return os.path.normpath(os.path.join(HERE, path))

# Faces above this get decimated. 16384 is what every YCB asset here
# carries, so it is the budget the scene is already known to compile at.
FACE_BUDGET = 16384


@dataclass(frozen=True)
class Source:
    """One raw object and what it has to become.

    Attributes:
        path: Source mesh, relative to the scene directory, or None when
            `build` generates the geometry instead.
        build: Builds the mesh from scratch, for an object whose scan is
            not the shape on the table.
        up: Which source axis points up, from the footprint test above.
        size: Final (x, y, z) extent [m] -- (length, width, height) as
            measured on the physical object.
        color: Flat RGB to paint it, or None to keep the source's own
            texture (which `texture` then names).
        texture: Source texture file to carry over, when `color` is None.
        xy_uniform: Scale x and y by one factor, from `size[1]` over the
            narrow footprint axis -- a diameter, for anything round.
        yaw_to_x: Rotate the footprint's principal axis onto +x. Only the
            hammer needs it; on a body of revolution it is meaningless.
    """

    path: Optional[str]
    up: str
    size: Tuple[float, float, float]
    color: Optional[Tuple[int, int, int]] = None
    texture: Optional[str] = None
    xy_uniform: bool = False
    yaw_to_x: bool = False
    build: Optional[Callable[[], trimesh.Trimesh]] = None


# The cup is BUILT, not scanned. The YCB scan is a tapered tub with a
# flared lip and an interior floor 1.5 cm up -- the cup on the table is a
# straight open cylinder, rounded only where it meets the table. Revolving
# a profile gives exactly that, and gives control of the one thing a scan
# cannot express: how full it is, which is what the simulated mass follows.
CUP_RADIUS = 0.05        # 0.10 m across, as measured
CUP_HEIGHT = 0.12
CUP_WALL = 0.003         # wall thickness above the fill line
CUP_FILL = 0.06          # solid to here -- "filled to the middle"
CUP_BASE_FILLET = 0.006  # the slight curve where it stands


def _build_cup() -> trimesh.Trimesh:
    """An open cylinder, solid to mid-height, with a rounded base edge."""
    r, h = CUP_RADIUS, CUP_HEIGHT
    f, t = CUP_BASE_FILLET, CUP_WALL
    arc = [
        (r - f + f * np.sin(a), f - f * np.cos(a))
        for a in np.linspace(0.0, np.pi / 2, 8)
    ]
    # (radius, height), counter-clockwise, starting and ending on the axis
    # so `revolve` closes it into a solid.
    profile = [(0.0, 0.0), *arc, (r, h), (r - t, h),
               (r - t, CUP_FILL), (0.0, CUP_FILL)]
    mesh = trimesh.creation.revolve(np.asarray(profile), sections=96)
    mesh.fix_normals()
    return mesh


# The hammer is SCALED AND EXTRUDED, not simply fitted to a bounding box.
# Fitting the scan's 18.1 x 13.4 x 3.3 cm to a 25 cm tool squashes the head
# to half width, because this scan's head spans 74% of its length where the
# real tool's spans ~36%. So the HEAD sets the scale -- 9 cm face-to-claw,
# 4 cm thick, the measured numbers -- and the handle is then lengthened to
# make up the overall 25 cm, which is what a longer handle actually is.
HAMMER_SRC = "../xarm6_pusht_tabletop/assets/hammer_centered.obj"
HAMMER_HEAD_Y = 0.09   # striking face to claw tip, across the handle
HAMMER_THICK = 0.04    # head thickness, i.e. how tall it lies on the table
HAMMER_LENGTH = 0.25   # overall, claw end of the head to the butt


def _handle_split(v: np.ndarray) -> Tuple[float, float]:
    """Where the head begins, and where the butt cap ends, along x.

    Found from the profile rather than hard-coded: the head is the first
    band whose y span is more than twice the handle's.
    """
    xs = np.linspace(v[:, 0].min(), v[:, 0].max(), 40)
    spans = np.array([
        np.ptp(v[(v[:, 0] >= xs[i]) & (v[:, 0] < xs[i + 1]), 1])
        if ((v[:, 0] >= xs[i]) & (v[:, 0] < xs[i + 1])).sum() > 4 else 0.0
        for i in range(39)
    ])
    handle_span = np.median(spans[spans > 0][:15])
    x_head = xs[np.argmax(spans > 2.0 * handle_span)]
    # The butt's rounded cap, translated rigidly so it keeps its shape.
    x_cap = v[:, 0].min() + 0.15 * (x_head - v[:, 0].min())
    return x_head, x_cap


def _build_hammer() -> trimesh.Trimesh:
    """Scale the scan by its head, then lengthen the shaft to 25 cm."""
    mesh = trimesh.load(_src(HAMMER_SRC), force="mesh", process=False)
    v = np.asarray(mesh.vertices, dtype=np.float64).copy()

    # Handle onto +x (the scan carries it on +y), head to +x.
    xy = v[:, :2] - v[:, :2].mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(xy.T))
    angle = -np.arctan2(vecs[1, -1], vecs[0, -1])
    c, s = np.cos(angle), np.sin(angle)
    v = v @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]).T
    mid = v[:, 0].mean()
    if np.ptp(v[v[:, 0] < mid, 1]) > np.ptp(v[v[:, 0] > mid, 1]):
        v[:, :2] *= -1          # the wide end is the head; put it on +x

    # The head sets both in-plane scales, so its profile stays undistorted.
    extent = v.max(axis=0) - v.min(axis=0)
    v[:, :2] *= HAMMER_HEAD_Y / extent[1]
    v[:, 2] *= HAMMER_THICK / extent[2]

    # Lengthen the shaft to make up the rest. Linear from 0 at the head to
    # the full extension at the cap, so the shaft stretches along its own
    # axis -- a cylinder lengthened is still that cylinder -- and the head
    # and butt cap are carried along rigidly.
    x_head, x_cap = _handle_split(v)
    delta = HAMMER_LENGTH - np.ptp(v[:, 0])
    v[:, 0] -= delta * np.clip((x_head - v[:, 0]) / (x_head - x_cap), 0.0, 1.0)
    return trimesh.Trimesh(v, np.asarray(mesh.faces), process=False)


SOURCES = {
    "coca_cola": Source(
        "assets_original/coca_cola/Coca Cola Bottle.obj",
        up="y", size=(0.07, 0.07, 0.215), xy_uniform=True,
        texture="assets_original/coca_cola/CylinderSurface_Color.jpeg",
    ),
    "coffee_cup": Source(
        # Handle included: 0.075 is the BODY diameter, so the footprint
        # comes out ~0.105 along the handle. Standing on +y, not +x.
        "assets_original/coffee_cup/Coffe cupobj.obj",
        up="y", size=(0.075, 0.075, 0.10),
        color=(255, 255, 0), xy_uniform=True,
    ),
    "cup": Source(
        None, up="z", size=(0.10, 0.10, 0.12),
        color=(255, 0, 0), xy_uniform=True, build=_build_cup,
    ),
    "hammer_real": Source(
        # Named `hammer_real`, NOT `hammer`: the registry's `hammer` is
        # the sim tool, whose collision boxes describe a 9 x 18 cm shape.
        # Writing this 25 x 9 cm mesh as `hammer_centered.obj` would put it
        # on a real scene under those boxes -- the drawing and the physics
        # would be different tools.
        #
        # Built from the SIM scene's mesh, not `assets_original/hammer`:
        # that scan is a different tool. `_build_hammer` already produces
        # the final geometry, so `size` here only restates it and the
        # generic scale below comes out as 1.
        None, up="z",
        size=(HAMMER_LENGTH, HAMMER_HEAD_Y, HAMMER_THICK),
        color=(0, 0, 255), build=_build_hammer,
    ),
}


def _up_to_z(up: str) -> np.ndarray:
    """A PROPER rotation taking the source's up axis onto +z.

    A bare axis swap would mirror the mesh -- every face wound backwards,
    which a renderer shows as an inside-out object. These are rotations.
    """
    if up == "z":
        return np.eye(4)
    r = np.eye(4)
    if up == "y":          # +90 deg about x: (x, y, z) -> (x, -z, y)
        r[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    elif up == "x":        # -90 deg about y: (x, y, z) -> (-z, y, x)
        r[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    else:
        raise ValueError(f"up must be x, y or z, got {up!r}")
    assert np.isclose(np.linalg.det(r[:3, :3]), 1.0)
    return r


def _yaw_principal_to_x(vertices: np.ndarray) -> np.ndarray:
    """Rotation about z putting the footprint's long axis on +x."""
    xy = vertices[:, :2] - vertices[:, :2].mean(axis=0)
    # Principal direction of the footprint: the eigenvector of the 2D
    # covariance with the larger eigenvalue.
    _, vecs = np.linalg.eigh(np.cov(xy.T))
    long_axis = vecs[:, -1]
    angle = -np.arctan2(long_axis[1], long_axis[0])
    c, s = np.cos(angle), np.sin(angle)
    r = np.eye(4)
    r[:2, :2] = [[c, -s], [s, c]]
    return r


def _write_texture(name: str, src: Source) -> str:
    """Write the object's texture and return its file name."""
    if src.color is None:
        assert src.texture is not None
        # Re-encoded as PNG, not copied: MuJoCo's texture loader reads PNG
        # and KTX only and refuses a JPEG outright, so the shipped photo
        # could never be rendered in sim. 74 KB -> 199 KB buys that.
        out = f"{name}_texture.png"
        Image.open(_src(src.texture)).convert("RGB").save(
            os.path.join(OUT_DIR, out))
        return out
    out = f"{name}_texture.png"
    # 8x8 rather than 1x1: some renderers refuse a texture below their
    # minimum mip size, and a flat colour costs nothing either way.
    Image.new("RGB", (8, 8), src.color).save(os.path.join(OUT_DIR, out))
    return out


def _write_mtl(name: str, src: Source, texture: str) -> None:
    """One material per object, pointing at the texture beside it."""
    kd = ((1.0, 1.0, 1.0) if src.color is None
          else tuple(c / 255.0 for c in src.color))
    with open(os.path.join(OUT_DIR, f"{name}_centered.mtl"), "w") as f:
        f.write(
            f"# {name}, generated by prepare_objects.py\n"
            f"newmtl {name}\n"
            "Ka 1.000 1.000 1.000\n"
            f"Kd {kd[0]:.3f} {kd[1]:.3f} {kd[2]:.3f}\n"
            "Ks 0.000 0.000 0.000\n"
            "d 1.0\n"
            "illum 1\n"
            f"map_Kd {texture}\n"
        )


def prepare(name: str, src: Source) -> dict:
    """Reorient, scale, centre and write one object. Returns its report."""
    if src.build is not None:
        mesh = src.build()
    else:
        mesh = trimesh.load(_src(src.path), force="mesh", process=False)
    raw = mesh.bounds[1] - mesh.bounds[0]
    uv = getattr(mesh.visual, "uv", None)

    if len(mesh.faces) > FACE_BUDGET:
        mesh = mesh.simplify_quadric_decimation(face_count=FACE_BUDGET)
        uv = None                      # decimation does not carry UVs over

    mesh.apply_transform(_up_to_z(src.up))
    if src.yaw_to_x:
        mesh.apply_transform(_yaw_principal_to_x(np.asarray(mesh.vertices)))

    # Rebuilt rather than scaled in place: a non-uniform scale does not
    # preserve normals, and recomputing them from the scaled geometry is
    # the only way they stay correct.
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    extent = verts.max(axis=0) - verts.min(axis=0)
    scale = np.asarray(src.size) / extent
    if src.xy_uniform:
        # The narrow footprint axis is the diameter; the wide one carries
        # whatever sticks out of it (a mug's handle) and follows along.
        scale[:2] = src.size[1] / extent[:2].min()
    verts = verts * scale

    lo, hi = verts.min(axis=0), verts.max(axis=0)
    verts[:, :2] -= (lo[:2] + hi[:2]) / 2.0    # footprint centred on origin
    verts[:, 2] -= lo[2]                       # underside at z = 0
    out = trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.faces),
                          process=False)

    texture = _write_texture(name, src)
    _write_mtl(name, src, texture)
    _write_obj(name, out, uv)
    shutil.copyfile(os.path.join(OUT_DIR, f"{name}_centered.obj"),
                    os.path.join(MIRROR_DIR, f"{name}_centered.obj"))

    final = out.bounds[1] - out.bounds[0]
    return {"raw": raw, "oriented": extent, "scale": scale, "final": final,
            "faces": len(out.faces), "texture": texture}


def _write_obj(name: str, mesh: trimesh.Trimesh,
               uv: Optional[np.ndarray]) -> None:
    """Write the OBJ with its material and texture coordinates.

    Written by hand rather than through `trimesh.export`: the export drops
    the `mtllib`/`usemtl` pair unless a full TextureVisuals is rebuilt, and
    that pair is what makes the mesh textured for FoundationPose. A mesh
    with no UVs of its own gets one constant coordinate -- enough for a
    flat colour, which is all these carry.
    """
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.vertex_normals)
    lines = [
        f"# {name}, generated by prepare_objects.py",
        f"# Vertices: {len(verts)}  Faces: {len(faces)}",
        f"mtllib {name}_centered.mtl",
    ]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    if uv is not None and len(uv) == len(verts):
        lines += [f"vt {u:.6f} {v:.6f}" for u, v in np.asarray(uv)]
    else:
        lines.append("vt 0.500000 0.500000")
    lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in normals]
    lines.append(f"usemtl {name}")
    per_vertex_uv = uv is not None and len(uv) == len(verts)
    for a, b, c in faces + 1:
        if per_vertex_uv:
            lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
        else:
            lines.append(f"f {a}/1/{a} {b}/1/{b} {c}/1/{c}")
    with open(os.path.join(OUT_DIR, f"{name}_centered.obj"), "w") as f:
        f.write("\n".join(lines) + "\n")


def report(name: str, r: dict) -> None:
    """Print what each object was scaled by, distortion included."""
    ratio = r["scale"] / r["scale"].max()
    print(f"\n{name}")
    print(f"  source extent   {np.round(r['raw'], 4)}")
    print(f"  oriented (z up) {np.round(r['oriented'], 4)}")
    print(f"  scale per axis  {np.round(r['scale'], 4)}"
          f"   (anisotropy {1 / ratio.min():.2f}x)")
    print(f"  final extent    {np.round(r['final'], 4)} m"
          f"   faces {r['faces']}   texture {r['texture']}")


def main() -> None:
    """Prepare every object in `SOURCES`."""
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, src in SOURCES.items():
        report(name, prepare(name, src))


if __name__ == "__main__":
    main()
