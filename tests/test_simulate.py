from paiec.models.empirical_mean import predict
from paiec.simulate import evaluate_pair


def test_empirical_mean_pair():
    s = {"normalized_name": "m"}
    rows = [{"subject": s, "item": {"benchmark_id": "b", "item_content": str(i)},
             "item_id": str(i), "label": int(i % 3 == 0)} for i in range(100)]
    out, alc = evaluate_pair(predict, rows)
    assert set(out) == {0, 1, 3, 7, 15, 31} and 0 <= alc <= 1
