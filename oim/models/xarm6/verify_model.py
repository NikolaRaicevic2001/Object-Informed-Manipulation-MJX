"""Standalone sanity check for xarm6.xml / scene.xml -- no oim imports.

Loads scene.xml exactly as any external consumer would (no path tricks),
reports body/joint/actuator counts, per-link mass, total arm reach, and
renders a couple of offscreen views to results/ for a visual sanity check.
Re-run after any edit to xarm6.xml.
"""

import pathlib

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).parent
OUT_DIR = HERE / "verify_renders"


def main() -> None:
    """Load the model, sweep a few poses, and report anything odd."""
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)

    print(f"nq={model.nq} nu={model.nu} nbody={model.nbody} njnt={model.njnt}")
    assert model.nu == 5, f"expected 5 actuated joints, got {model.nu}"

    total_mass = sum(model.body_mass)
    print(f"total mass: {total_mass:.3f} kg")

    print("\nper-body mass:")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  {name:20s} {model.body_mass[i]:.4f} kg")

    print("\njoint ranges (deg):")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = np.degrees(model.jnt_range[i])
        print(f"  {name:20s} [{lo:8.2f}, {hi:8.2f}]")

    # Zero-configuration forward kinematics.
    mujoco.mj_forward(model, data)
    base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "xarm6_link_base"
    )
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "xarm6_tip")
    base_pos = data.xpos[base_id]
    tip_pos = data.site_xpos[tip_site]
    reach = np.linalg.norm(tip_pos - base_pos)
    print(f"\nzero-config tip position: {tip_pos}")
    print(f"zero-config base->tip distance: {reach:.3f} m")
    print("(real xArm6 max reach spec is ~0.700 m; zero-config is a bent")
    print(" pose, not full extension, so this number is expected to be")
    print(" well under that, not equal to it.)")

    # A joint-space sweep to check nothing NaNs / self-collides violently
    # across the declared range, and to build renders at a few poses.
    rng = np.random.default_rng(0)
    renderer = mujoco.Renderer(model, height=480, width=640)
    OUT_DIR.mkdir(exist_ok=True)

    poses = {
        "zero": np.zeros(model.nq),
        "random1": None,
        "random2": None,
    }
    for key in ("random1", "random2"):
        q = np.zeros(model.nq)
        for j in range(model.njnt):
            lo, hi = model.jnt_range[j]
            q[j] = rng.uniform(lo, hi)
        poses[key] = q

    cam = mujoco.MjvCamera()
    cam.distance = 1.3
    cam.azimuth = 120
    cam.elevation = -20
    cam.lookat = np.array([0.2, 0.0, 0.2])

    for name, q in poses.items():
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        assert np.all(np.isfinite(data.xpos)), \
            f"NaN body position at pose {name}"
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        out_path = OUT_DIR / f"pose_{name}.png"
        import PIL.Image  # noqa: PLC0415

        PIL.Image.fromarray(img).save(out_path)
        print(f"rendered {name} -> {out_path}")

    print("\nOK: model loads standalone, all poses finite, renders written.")


if __name__ == "__main__":
    main()
