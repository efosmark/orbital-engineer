import math

import pytest

from orbitalengineer.twobody import twobody


def assert_vec_close(actual, expected, **kwargs):
    """Assert both components of a 2D vector with pytest approximate matching."""
    assert actual[0] == pytest.approx(expected[0], **kwargs)
    assert actual[1] == pytest.approx(expected[1], **kwargs)


def test_basic_vector_helpers():
    """The compiled vector helpers match ordinary 2D vector arithmetic."""
    assert twobody.vec_sub((5.0, -2.0), (1.5, 3.0)) == pytest.approx((3.5, -5.0))
    assert twobody.vec_dot((2.0, 3.0), (-4.0, 5.0)) == pytest.approx(7.0)
    assert twobody.vec_norm((3.0, 4.0)) == pytest.approx(5.0)


def test_vec_unit_handles_normal_and_near_zero_vectors():
    """Unit vectors normalize non-zero inputs and clamp near-zero inputs to zero."""
    assert_vec_close(twobody._vec_unit_jit((3.0, 4.0), 1e-15), (0.6, 0.8))
    assert_vec_close(twobody._vec_unit_jit((1e-16, 0.0), 1e-15), (0.0, 0.0))


def test_universal_gravitation():
    """Universal gravitation applies the inverse-square force equation."""
    assert twobody.universal_gravitation(2.0, 3.0, 4.0, 5.0) == pytest.approx(0.96)


@pytest.mark.parametrize(
    ("prograde", "expected"),
    [
        (True, (0.0, math.sqrt(10.0))),
        (False, (0.0, -math.sqrt(10.0))),
    ],
)
def test_vis_viva_velocity_jit_returns_tangential_velocity(prograde, expected):
    """Vis-viva velocity is perpendicular to the radius and flips by orbit direction."""
    actual = twobody._vis_viva_velocity_jit(
        10.0,
        0.0,
        10.0,
        100.0,
        10.0,
        prograde,
    )

    assert_vec_close(actual, expected)


def test_relative_state():
    """Relative state is body 2 position and velocity measured from body 1."""
    r_rel, v_rel = twobody.relative_state(
        (1.0, 2.0),
        (3.0, 4.0),
        (6.0, -1.0),
        (2.0, 9.0),
    )

    assert_vec_close(r_rel, (5.0, -3.0))
    assert_vec_close(v_rel, (-1.0, 5.0))


def test_orbital_energy():
    """A circular orbit has negative specific orbital energy of -mu / 2r."""
    assert twobody.orbital_energy(100.0, 10.0, math.sqrt(10.0)) == pytest.approx(-5.0)


def test_standard_grav_param_jit():
    """The standard gravitational parameter is G times total system mass."""
    assert twobody._standard_grav_param_jit(3.0, 7.0, 2.5) == pytest.approx(25.0)


def test_ellipse_axes_for_circular_orbit():
    """A circular orbit has matching semi-major and semi-minor axes."""
    a, b = twobody._ellipse_axes_jit((10.0, 0.0), (0.0, math.sqrt(10.0)), 100.0, 0.0, 0.0)

    assert a == pytest.approx(10.0)
    assert b == pytest.approx(10.0)


def test_eccentricity_vec_for_known_periapsis_state():
    """A periapsis state on the x-axis points the eccentricity vector along +x."""
    v = math.sqrt(30.0)

    assert_vec_close(twobody.eccentricity_vec((5.0, 0.0), (0.0, v), 100.0), (0.5, 0.0))


def test_argument_of_periapsis():
    """The argument of periapsis is the angle of the eccentricity unit vector."""
    assert twobody.argument_of_periapsis((0.0, 1.0)) == pytest.approx(math.pi / 2.0)


def test_specific_angular_momentum_and_retrograde_detection():
    """The sign of 2D angular momentum determines retrograde motion."""
    assert twobody.specific_angular_momentum((2.0, 0.0), (0.0, 3.0)) == pytest.approx(6.0)
    assert twobody.is_retrograde((2.0, 0.0), (0.0, 3.0)) is False
    assert twobody.is_retrograde((2.0, 0.0), (0.0, -3.0)) is True


def test_true_anomaly_wraps_into_positive_range():
    """True anomaly is normalized to the [0, tau) interval."""
    assert twobody.true_anomaly((0.0, 1.0), 0.0) == pytest.approx(math.pi / 2.0)
    assert twobody.true_anomaly((1.0, 0.0), math.pi / 2.0) == pytest.approx(1.5 * math.pi)


def test_eccentric_and_mean_anomaly():
    """Eccentric anomaly feeds Kepler's equation for the mean anomaly."""
    e_vec = (0.5, 0.0)
    true_anomaly = math.pi / 2.0
    eccentric_anomaly = twobody.eccentric_anomaly(true_anomaly, e_vec)

    assert eccentric_anomaly == pytest.approx(math.pi / 3.0)
    assert twobody.mean_anomaly(e_vec, eccentric_anomaly) == pytest.approx(
        eccentric_anomaly - 0.5 * math.sin(eccentric_anomaly)
    )


def test_period_and_mean_motion_are_inverse_angular_rates():
    """Mean motion is tau divided by the orbital period."""
    period = twobody.orbital_period(10.0, 100.0)

    assert period == pytest.approx(2.0 * math.pi * math.sqrt(10.0))
    assert twobody.mean_motion(period) == pytest.approx(twobody.TAU / period)


def test_periapsis_direction_jit():
    """Periapsis direction shares vec_unit's normal and near-zero behavior."""
    assert_vec_close(twobody._periapsis_direction_jit((3.0, 4.0), 1e-15), (0.6, 0.8))
    assert_vec_close(twobody._periapsis_direction_jit((1e-16, 0.0), 1e-15), (0.0, 0.0))


def test_barycenter():
    """The barycenter is the mass-weighted average of both positions."""
    assert_vec_close(twobody.barycenter((0.0, 0.0), (10.0, 0.0), 3.0, 1.0), (2.5, 0.0))


def test_ellipse_center_offsets_from_focus_away_from_periapsis():
    """The ellipse center is shifted from the focus opposite the periapsis direction."""
    assert_vec_close(twobody.ellipse_center((0.0, 0.0), (0.5, 0.0), 0.5, 10.0), (-5.0, 0.0))


def test_escape_velocity():
    """Escape velocity follows sqrt(2mu / radius) for the relative radius."""
    assert twobody.escape_velocity((3.0, 4.0), 100.0) == pytest.approx(math.sqrt(40.0))
