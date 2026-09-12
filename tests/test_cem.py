
from oim.algs.cem import CEM
from oim.tasks.pendulum import Pendulum


def test_explore_fraction() -> None:
    """Unit test for sampling controls with different explore_fraction values.

    This test uses the Pendulum task as a dummy task to verify:
      - The overall controls array shape is correct.
      - The split between main and exploration samples is as expected.
    """
    num_samples = 10
    num_elites = 2
    sigma_start = 1.0
    sigma_min = 0.1

    # Test different fractions: no exploration, partial exploration, full
    # exploration.
    for explore_fraction in [0.0, 0.3, 0.5, 0.75, 1.0]:
        task = Pendulum()
        opt = CEM(
            task=task,
            num_samples=num_samples,
            num_elites=num_elites,
            sigma_start=sigma_start,
            sigma_min=sigma_min,
            explore_fraction=explore_fraction,
            plan_horizon=1.0,
            spline_type="zero",
            num_knots=11,
        )
        params = opt.init_params(seed=42)
        controls, new_params = opt.sample_knots(params)

        # Check the overall shape of the controls array.
        expected_shape = (num_samples, opt.num_knots, task.model.nu)
        assert controls.shape == expected_shape, (
            f"Expected shape {expected_shape} but got {controls.shape} "
            f"for explore_fraction = {explore_fraction}"
        )

        # Calculate expected number of exploration samples.
        num_explore = int(explore_fraction * num_samples)
        num_main = num_samples - num_explore

        # The implementation concatenates main samples first and exploration
        # samples later.
        main_controls = controls[:num_main]
        explore_controls = controls[num_main:]

        # Verify that the main and exploration segments have the correct shapes.
        expected_main_shape = (num_main, opt.num_knots, task.model.nu)
        expected_explore_shape = (
            num_explore,
            opt.num_knots,
            task.model.nu,
        )
        assert main_controls.shape == expected_main_shape, (
            f"Expected main controls shape {expected_main_shape} but got "
            f"{main_controls.shape} for explore_fraction = {explore_fraction}"
        )
        assert explore_controls.shape == expected_explore_shape, (
            f"Expected explore controls shape {expected_explore_shape} but got "
            f"{explore_controls.shape} for explore_fraction = "
            f"{explore_fraction}"
        )


