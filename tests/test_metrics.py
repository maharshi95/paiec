import math

from paiec.metrics import ALC_WEIGHTS, BUDGETS, brier, brier_alc, ece


def test_weights_sum_to_one():
    assert math.isclose(sum(ALC_WEIGHTS), 1.0)


def test_constant_half_gives_quarter():
    assert math.isclose(brier_alc({b: 0.25 for b in BUDGETS}), 0.25)


def test_perfect_is_zero():
    assert brier([1, 0], [1, 0]) == 0 and ece([1, 0], [1, 0]) == 0
