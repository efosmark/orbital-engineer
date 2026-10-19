import pytest

from orbitalengineer import flags
from orbitalengineer.engine.tests import test


STARTING_POSITION_MAG = 1000.0
ORBITAL_ECCENTRICITY = 1.0
TICKS_PER_ORBIT = 100.0
NUM_POSITIONS_TO_CHECK = 20
ORBIT_RADIUS_REL_TOLERANCE = 1e-2


@pytest.fixture
def orbital_test():
    """Provide an isolated orbital simulation test fixture."""
    return test.OrbitalTest(device=(0, 0))


@pytest.fixture
def circular_orbit(orbital_test):
    """Create a fixed primary body and a secondary body in circular orbit."""
    primary = orbital_test.add_body(
        position=0 + 0j,
        velocity=0 + 0j,
        mass=1e10,
        radius=1,
        flags=flags.FIXED_POSITION | flags.FIXED_VELOCITY,
    )
    secondary = orbital_test.add_orbiting_body(
        primary,
        position=STARTING_POSITION_MAG + 0j,
        mass=1,
        radius=1,
        flags=0,
        eccentricity=ORBITAL_ECCENTRICITY,
    )
    target_orbital_period = test.orbital_period(
        primary,
        secondary,
        orbital_test.config.GRAV_CONSTANT,
    )

    return orbital_test, secondary, target_orbital_period


def test_circular_orbit_has_expected_starting_radius(circular_orbit):
    """The secondary starts at the configured orbital radius."""
    _, secondary, _ = circular_orbit

    assert abs(secondary.position) == pytest.approx(STARTING_POSITION_MAG)


def test_circular_orbit_keeps_radius_within_tolerance(circular_orbit):
    """A circular orbit remains within nominal radius tolerance over one period."""
    orbital_test, secondary, target_orbital_period = circular_orbit
    dt_step = target_orbital_period / TICKS_PER_ORBIT
    dt_sub_period = target_orbital_period / NUM_POSITIONS_TO_CHECK
    starting_position_mag = abs(secondary.position)

    for _ in range(NUM_POSITIONS_TO_CHECK):
        orbital_test.run_for(dt_sub_period, dt_step=dt_step)
        ending_position_mag = abs(secondary.position)

        assert ending_position_mag == pytest.approx(
            starting_position_mag,
            rel=ORBIT_RADIUS_REL_TOLERANCE,
        )
