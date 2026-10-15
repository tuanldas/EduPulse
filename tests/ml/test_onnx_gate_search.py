"""Tìm kiếm cấu hình int8: đăng ký trước không bị sửa, luật chọn áp đúng, không chạy lại."""

import copy
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.train

pytest.importorskip("torch")
pytest.importorskip("onnx")

from ml import onnx_gate_search as search  # noqa: E402
from ml.export_onnx import DEPLOY_QUANT_CONFIG, GATE_MAX_DROP_POINTS, QUANT_CONFIGS  # noqa: E402

# Băm của khối đăng ký vòng 1, ghi cố định để một lần sửa đồng thời khối và băm cũng bị bắt.
ROUND_ONE_SHA256 = "306762f86241015a2c32f9d2ec2214691fe40058f92e4b431d801f58b0a240fa"
ROUNDS = pytest.mark.parametrize("round_number", [1, 2])


def document() -> dict:
    return json.loads(search.REGISTRATION_PATH.read_text(encoding="utf-8"))


def registration_of(round_number: int) -> dict:
    return search.round_block(document(), round_number)["registration"]


def write_tampered(tmp_path: Path, tampered: dict) -> Path:
    path = tmp_path / "search.json"
    path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    return path


# --- Đăng ký ----------------------------------------------------------------------------------


@ROUNDS
def test_registration_block_matches_its_recorded_hash(round_number):
    loaded = search.load_registration(round_number=round_number)
    block = search.round_block(loaded, round_number)
    assert search.registration_digest(block["registration"]) == block["registration_sha256"]


def test_round_one_registration_is_the_one_made_before_its_run():
    assert document()["registration_sha256"] == ROUND_ONE_SHA256


def test_every_round_one_config_is_implemented_and_there_are_at_most_four():
    names = [c["name"] for c in registration_of(1)["configs"]]
    assert 1 <= len(names) <= 4
    assert all(name in QUANT_CONFIGS for name in names)


def test_round_two_pool_is_the_round_one_passers_plus_at_most_four_new_configs():
    registration = registration_of(2)
    new = [c["name"] for c in registration["new_configs"]]
    assert 1 <= len(new) <= 4
    assert all(QUANT_CONFIGS[name].heads_fp32 for name in new)
    pool = [entry["name"] for entry in registration["pool"]]
    assert pool == ["ffn_out_fp32", "ffn_fp32", "attn_fp32", *new]
    assert document()["results"]["passing"] == pool[:3]  # đúng ba cấu hình đạt cổng ở vòng 1
    references = [entry["name"] for entry in registration["reference_configs"]]
    assert references == ["all_int8"] and not set(references) & set(pool)


@ROUNDS
def test_registration_states_the_gate_the_code_enforces(round_number):
    gate = registration_of(round_number)["gate"]
    assert gate["max_drop_points"] == GATE_MAX_DROP_POINTS == 1.0
    assert gate["both_tasks_required"] and gate["test_split_untouched"]
    assert gate["not_lowered_after_seeing_results"]


def test_round_two_registers_the_lead_rule_and_the_reason_for_changing_it():
    rule = registration_of(2)["selection_rule_lead_verbatim"]
    assert len(rule["steps"]) == 4
    assert "5%" in rule["steps"][2] and "TỔNG số nhãn đổi" in rule["steps"][2]
    assert "tiền đề sai" in rule["reason_for_changing_the_rule"]
    assert registration_of(2)["speed_tie_fraction"] == search.SPEED_TIE_FRACTION == 0.05


@ROUNDS
def test_editing_the_registration_is_detected(tmp_path, round_number):
    tampered = document()
    registration = search.round_block(tampered, round_number)["registration"]
    registration["purpose"] = "đổi mục đích sau khi xem kết quả"
    with pytest.raises(RuntimeError, match="sha256"):
        search.load_registration(write_tampered(tmp_path, tampered), round_number)


@ROUNDS
def test_changing_the_threshold_is_detected(tmp_path, round_number):
    tampered = document()
    block = search.round_block(tampered, round_number)
    block["registration"]["gate"]["max_drop_points"] = 2.0
    block["registration_sha256"] = search.registration_digest(block["registration"])
    with pytest.raises(RuntimeError, match="ngưỡng"):
        search.load_registration(write_tampered(tmp_path, tampered), round_number)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: r["measurement"]["timing"].update(rounds=3), "cách đo"),
        (lambda r: r["measurement"]["timing"].update(sentences=100), "cách đo"),
        (lambda r: r.update(speed_tie_fraction=0.1), "ngưỡng hoà"),
        (lambda r: r["pool"].append({"name": "int4", "origin": "không có"}), "mã không có"),
    ],
)
def test_round_two_registration_that_disagrees_with_the_code_is_refused(tmp_path, edit, message):
    tampered = copy.deepcopy(document())
    block = tampered["round2"]
    edit(block["registration"])
    block["registration_sha256"] = search.registration_digest(block["registration"])
    with pytest.raises(RuntimeError, match=message):
        search.load_registration(write_tampered(tmp_path, tampered), round_number=2)


def test_registration_file_lives_in_results_metrics():
    assert (
        search.REGISTRATION_PATH == Path(search.REPO_ROOT) / "results/metrics/onnx_gate_search.json"
    )


# --- Luật chọn (hàm thuần) --------------------------------------------------------------------


def candidate(name, ms, flips=(0, 0), passes=True, size_bytes=1):
    return {
        "name": name,
        "passes_gate_batch1": passes,
        "ms_per_sentence_batch1": {"median": ms},
        "label_flips_vs_fp32_batch1": {"sentiment": flips[0], "topic": flips[1]},
        "size_bytes": size_bytes,
    }


def test_the_fastest_passing_config_wins_when_the_others_are_clearly_slower():
    rows = [candidate("a", 10.0, (9, 9)), candidate("b", 11.0, (0, 0))]
    outcome = search.select_config(rows)
    assert outcome["selected"] == "a" and outcome["within_tie_band"] == ["a"]
    assert outcome["fastest"] == {"name": "a", "ms_per_sentence_batch1": 10.0}


def test_configs_within_five_percent_of_the_fastest_are_decided_by_total_label_flips():
    rows = [
        candidate("a", 10.0, (9, 9)),
        candidate("b", 10.5, (1, 1)),
        candidate("c", 10.2, (4, 4)),
    ]
    outcome = search.select_config(rows)
    assert outcome["selected"] == "b"  # đúng 5% vẫn thuộc nhóm hoà
    assert outcome["within_tie_band"] == ["a", "b", "c"]


def test_a_config_just_beyond_five_percent_is_outside_the_tie_band():
    rows = [candidate("a", 10.0, (9, 9)), candidate("b", 10.51, (0, 0))]
    assert search.select_config(rows)["selected"] == "a"


def test_total_flips_sum_both_tasks():
    rows = [candidate("a", 10.0, (1, 9)), candidate("b", 10.1, (4, 4))]
    assert search.select_config(rows)["selected"] == "b"  # 8 < 10 dù cảm xúc nhiều hơn


def test_failing_configs_are_ignored_even_when_fastest():
    rows = [candidate("fast", 6.0, passes=False), candidate("slow", 10.0, (3, 3))]
    outcome = search.select_config(rows)
    assert outcome["selected"] == "slow" and outcome["passing"] == ["slow"]


def test_no_passing_config_means_nothing_is_selected():
    outcome = search.select_config([candidate("a", 6.0, passes=False)])
    assert outcome == {"passing": [], "fastest": None, "within_tie_band": [], "selected": None}


def test_equal_flips_prefer_lower_ms_then_registration_order():
    rows = [candidate("a", 10.3, (2, 2)), candidate("b", 10.0, (2, 2))]
    assert search.select_config(rows)["selected"] == "b"
    rows = [candidate("a", 10.0, (2, 2)), candidate("b", 10.0, (2, 2))]
    assert search.select_config(rows)["selected"] == "a"


def test_file_size_plays_no_part_in_the_choice():
    rows = [candidate("big", 10.0, (1, 1), size_bytes=500), candidate("small", 12.0, size_bytes=1)]
    assert search.select_config(rows)["selected"] == "big"


# --- Đo xen kẽ --------------------------------------------------------------------------------


def test_interleaved_timing_visits_every_model_each_round_in_rotating_order(monkeypatch):
    calls = []
    monkeypatch.setattr(search, "make_session", lambda path, threads: path.stem)
    monkeypatch.setattr(
        search,
        "run_logits",
        lambda session, sequences, batch_size: calls.append((session, len(sequences))),
    )
    paths = {name: Path(f"{name}.onnx") for name in ("a", "b", "c")}
    timings, load = search.interleaved_timing(paths, [[0, 5, 2]] * 300, threads=1)

    timed = [session for session, count in calls if count == search.TIMING_SENTENCES]
    warmups = [session for session, count in calls if count == search.TIMING_WARMUP]
    assert len(timed) == len(warmups) == 3 * search.TIMING_ROUNDS
    assert timed[:9] == ["a", "b", "c", "b", "c", "a", "c", "a", "b"]
    for start in range(0, len(timed), 3):
        assert sorted(timed[start : start + 3]) == ["a", "b", "c"]
    assert set(timings) == {"a", "b", "c"}
    assert all(len(t["runs"]) == search.TIMING_ROUNDS for t in timings.values())
    assert len(load) == search.TIMING_ROUNDS


def test_reproduction_check_compares_deltas_and_flips_with_round_one():
    def row(name, delta, flips):
        return {
            "name": name,
            "delta_macro_f1_points_batch1": {"sentiment": delta, "topic": 0.1},
            "label_flips_vs_fp32_batch1": {"sentiment": flips, "topic": 2},
        }

    round_one = {"results": {"candidates": [row("x", 0.3, 9), row("y", 0.2, 5)]}}
    rows = {"x": row("x", 0.3, 9), "y": row("y", 0.2, 6), "z": row("z", 1.0, 1)}
    assert search.reproduction_of_round_one(rows, round_one) == {
        "x": {"deltas_equal": True, "flips_equal": True},
        "y": {"deltas_equal": True, "flips_equal": False},
    }


# --- Chạy trọn vòng 2 trên mô hình nhỏ -------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_outputs(tmp_path_factory):
    """Thư mục outputs giả có `onnx/mtl-s42/model.fp32.onnx` của mô hình nhỏ (2 lớp, ẩn 32)."""
    import torch

    from edupulse.text import NORMALIZER_VERSION
    from ml import export_onnx
    from ml.common import write_json
    from ml.phobert_model import MultiHeadConfig, MultiHeadPhoBERT

    root = tmp_path_factory.mktemp("outputs")
    run_dir = root / "models" / "mtl-s42"
    torch.manual_seed(0)
    config = MultiHeadConfig(
        task_heads={"sentiment": 3, "topic": 4},
        vocab_size=64001,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=132,
        type_vocab_size=1,
        pad_token_id=1,
    )
    MultiHeadPhoBERT(config).save_pretrained(run_dir)
    write_json(
        run_dir / "history.json", {"normalizer_version": NORMALIZER_VERSION, "finished": True}
    )
    export_onnx.export(run_dir, root / "onnx" / "mtl-s42", threads=2)
    return root


@pytest.mark.data
@pytest.mark.slow
def test_round_two_runs_end_to_end_on_a_tiny_model(tiny_outputs, tmp_path, monkeypatch):
    """Cả đường chạy (lượng tử hoá, chấm, đo xen kẽ, chọn, ghi) với mô hình nhỏ, tệp JSON chép."""
    from types import SimpleNamespace

    copied = document()
    copied["round2"].update(status="registered", results=None)
    copied["round2"].pop("completed_at", None)
    path = write_tampered(tmp_path, copied)
    pool = [entry["name"] for entry in copied["round2"]["registration"]["pool"]]
    monkeypatch.setattr(search, "REGISTRATION_PATH", path)
    monkeypatch.setattr(search, "settings", lambda: SimpleNamespace(outputs_dir=tiny_outputs))
    monkeypatch.setattr(search, "docker_running", lambda: False)

    results = search.run(threads=2, round_number=2)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["round2"]["status"] == "completed" and written["round2"]["results"] == results
    assert written["registration"] == copied["registration"]  # khối vòng 1 không bị động tới
    assert written["round2"]["registration"] == copied["round2"]["registration"]
    assert [row["name"] for row in results["candidates"]] == pool
    assert [row["name"] for row in results["reference_rows"]] == ["all_int8"]
    for row in [*results["candidates"], *results["reference_rows"]]:
        assert row["heads_fp32"] == QUANT_CONFIGS[row["name"]].heads_fp32
        assert set(row["ms_per_sentence_batch1"]) == {"median", "runs"}
        assert row["total_label_flips_batch1"] == search.total_flips(row)
        assert set(row["delta_macro_f1_points_batch1"]) == {"sentiment", "topic"}
    assert results["fp32_reference"]["ms_per_sentence_batch1"]["median"] > 0
    assert set(results["reproduction_of_round_one"]) == {"ffn_out_fp32", "ffn_fp32", "attn_fp32"}
    assert results["selected"] in {None, *pool}
    assert results["selection_rule_applied"] == copied["round2"]["registration"]["selection_rule"]
    assert not list((tiny_outputs / "onnx" / "mtl-s42-search").glob("*.onnx"))  # đã dọn tệp tạm
    with pytest.raises(RuntimeError, match="đã chạy xong"):
        search.run(threads=2, round_number=2)


def test_search_refuses_a_thread_count_other_than_the_registered_one(tmp_path, monkeypatch):
    copied = document()
    copied["round2"]["results"] = None
    monkeypatch.setattr(search, "REGISTRATION_PATH", write_tampered(tmp_path, copied))
    with pytest.raises(RuntimeError, match="luồng"):
        search.run(threads=7, round_number=2)


def test_round_one_can_no_longer_be_run_by_the_current_code(tmp_path, monkeypatch):
    copied = document()
    copied["results"] = None
    monkeypatch.setattr(search, "REGISTRATION_PATH", write_tampered(tmp_path, copied))
    with pytest.raises(RuntimeError, match="bản mã cũ"):
        search.run(threads=2, round_number=1)


# --- Kết quả đã ghi ---------------------------------------------------------------------------


def completed_results(round_number: int) -> dict:
    results = search.round_block(document(), round_number)["results"]
    if results is None:
        pytest.skip("tìm kiếm chưa chạy")
    return results


@ROUNDS
def test_recorded_gate_verdicts_follow_the_recorded_deltas(round_number):
    results = completed_results(round_number)
    for row in [*results["candidates"], *results.get("reference_rows", [])]:
        deltas = row["delta_macro_f1_points_batch1"].values()
        assert row["passes_gate_batch1"] == all(-d <= GATE_MAX_DROP_POINTS for d in deltas)


def test_round_one_selected_the_smallest_passing_file_as_its_rule_said():
    results = completed_results(1)
    passing = [c for c in results["candidates"] if c["passes_gate_batch1"]]
    assert results["passing"] == [c["name"] for c in passing]
    expected = min(passing, key=lambda c: (c["size_bytes"], c["ms_per_sentence_batch1"]["median"]))
    assert results["selected"] == expected["name"]


def test_round_two_selection_follows_the_registered_rule():
    results = completed_results(2)
    outcome = search.select_config(results["candidates"])
    assert {key: results[key] for key in outcome} == outcome
    assert [c["name"] for c in results["candidates"]] == [
        entry["name"] for entry in registration_of(2)["pool"]
    ]
    for row in results["candidates"]:
        assert row["total_label_flips_batch1"] == search.total_flips(row)
        assert len(row["ms_per_sentence_batch1"]["runs"]) == search.TIMING_ROUNDS


def test_round_two_remeasured_round_one_configs_reproduce_their_numbers():
    reproduced = completed_results(2)["reproduction_of_round_one"]
    assert set(reproduced) == {"ffn_out_fp32", "ffn_fp32", "attn_fp32"}
    assert all(all(checks.values()) for checks in reproduced.values()), reproduced


def test_deployed_config_is_the_selected_one():
    selected = completed_results(2)["selected"]
    assert selected == DEPLOY_QUANT_CONFIG or document().get("deploy_override")


@ROUNDS
def test_the_search_cannot_be_rerun_after_it_completed(round_number):
    completed_results(round_number)
    with pytest.raises(RuntimeError, match="đã chạy xong"):
        search.run(threads=1, round_number=round_number)
