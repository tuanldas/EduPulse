"""Tìm kiếm cấu hình int8: đăng ký trước không bị sửa, luật chọn áp đúng, không chạy lại."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.train

pytest.importorskip("torch")
pytest.importorskip("onnx")

from ml import onnx_gate_search as search  # noqa: E402
from ml.export_onnx import DEPLOY_QUANT_CONFIG, GATE_MAX_DROP_POINTS, QUANT_CONFIGS  # noqa: E402


def document() -> dict:
    return json.loads(search.REGISTRATION_PATH.read_text(encoding="utf-8"))


def test_registration_block_matches_its_recorded_hash():
    loaded = search.load_registration()
    assert search.registration_digest(loaded["registration"]) == loaded["registration_sha256"]


def test_every_registered_config_is_implemented_and_there_are_at_most_four():
    names = [c["name"] for c in document()["registration"]["configs"]]
    assert 1 <= len(names) <= 4
    assert all(name in QUANT_CONFIGS for name in names)


def test_registration_states_the_gate_the_code_enforces():
    gate = document()["registration"]["gate"]
    assert gate["max_drop_points"] == GATE_MAX_DROP_POINTS == 1.0
    assert gate["both_tasks_required"] and gate["test_split_untouched"]


def test_editing_the_registration_is_detected(tmp_path):
    tampered = document()
    tampered["registration"]["configs"][0]["rationale"] = "đổi lý do sau khi xem kết quả"
    path = tmp_path / "search.json"
    path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256"):
        search.load_registration(path)


def test_changing_the_threshold_is_detected(tmp_path):
    tampered = document()
    tampered["registration"]["gate"]["max_drop_points"] = 2.0
    tampered["registration_sha256"] = search.registration_digest(tampered["registration"])
    path = tmp_path / "search.json"
    path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ngưỡng"):
        search.load_registration(path)


def completed_results() -> dict:
    results = document()["results"]
    if results is None:
        pytest.skip("tìm kiếm chưa chạy")
    return results


def test_recorded_gate_verdicts_follow_the_recorded_deltas():
    for row in completed_results()["candidates"]:
        deltas = row["delta_macro_f1_points_batch1"].values()
        assert row["passes_gate_batch1"] == all(-d <= GATE_MAX_DROP_POINTS for d in deltas)


def test_recorded_selection_is_the_smallest_passing_file():
    results = completed_results()
    passing = [c for c in results["candidates"] if c["passes_gate_batch1"]]
    assert results["passing"] == [c["name"] for c in passing]
    expected = min(passing, key=lambda c: (c["size_bytes"], c["ms_per_sentence_batch1"]["median"]))
    assert results["selected"] == expected["name"]


def test_deployed_config_is_the_selected_one():
    assert completed_results()["selected"] == DEPLOY_QUANT_CONFIG


def test_the_search_cannot_be_rerun_after_it_completed():
    completed_results()
    with pytest.raises(RuntimeError, match="đã chạy xong"):
        search.run(threads=1)


def test_registration_file_lives_in_results_metrics():
    assert (
        search.REGISTRATION_PATH == Path(search.REPO_ROOT) / "results/metrics/onnx_gate_search.json"
    )
