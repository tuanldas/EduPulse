"""Logic so khớp của `scripts/check_amd64.py`: chọn câu val, golden 20/20, ngưỡng 98%, báo cáo.

Không cần Docker: phần chạy container được thay bằng dữ liệu dựng sẵn hoặc một máy chủ giả.
"""

import importlib.util
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from edupulse.config import REPO_ROOT, load_settings
from edupulse.labels import LABELS

SCRIPTS = REPO_ROOT / "scripts"


def load_script():
    sys.path.insert(0, str(SCRIPTS))  # check_amd64 dùng lại smoke_test nằm cùng thư mục
    try:
        spec = importlib.util.spec_from_file_location("check_amd64", SCRIPTS / "check_amd64.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["check_amd64"] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


check = load_script()
SENTIMENT, TOPIC = LABELS["sentiment"], LABELS["topic"]


def analysis(sentiment: str = SENTIMENT[2], topic: str = TOPIC[0], boost: float = 0.0) -> dict:
    """Phản hồi /api/analyze rút gọn: nhãn và xác suất của hai head."""

    def head(label: str, labels: list[str]) -> dict:
        probs = {name: 0.1 for name in labels}
        probs[label] = round(0.7 + boost, 6)
        return {"label": label, "confidence": probs[label], "probs": probs}

    return {"sentiment": head(sentiment, SENTIMENT), "topic": head(topic, TOPIC)}


def golden_expected(n: int = 20) -> dict[str, tuple[str, str]]:
    return {f"g{i:02d}": (SENTIMENT[i % 3], TOPIC[i % 4]) for i in range(1, n + 1)}


def responses_for(expected: dict, val_labels: list[tuple[str, str]]) -> list[dict]:
    return [analysis(*labels) for labels in [*expected.values(), *val_labels]]


def val_labels(n: int = 200) -> list[tuple[str, str]]:
    return [(SENTIMENT[i % 3], TOPIC[(i + 1) % 4]) for i in range(n)]


# --- Chọn câu val ---------------------------------------------------------------------------------


def test_sample_is_fixed_by_the_seed_sorted_and_without_repeats():
    first = check.pick_indices(1583, 200, seed=0)
    assert first == check.pick_indices(1583, 200, seed=0)
    assert first != check.pick_indices(1583, 200, seed=1)
    assert first == sorted(set(first)) and len(first) == 200
    assert 0 <= first[0] and first[-1] < 1583


def test_sample_cannot_be_empty_or_larger_than_the_validation_set():
    with pytest.raises(check.CheckError, match="chỉ có 150"):
        check.pick_indices(150, 200)
    with pytest.raises(check.CheckError, match="từ 1"):
        check.pick_indices(150, 0)
    assert check.pick_indices(5, 5) == [0, 1, 2, 3, 4]


# --- Nhãn, golden, xác suất -----------------------------------------------------------------------


def test_labels_are_the_two_heads_and_a_malformed_reply_is_an_error():
    assert check.labels_of(analysis("negative", "facility")) == ("negative", "facility")
    with pytest.raises(check.CheckError, match="thiếu nhãn"):
        check.labels_of({"sentiment": {"label": "positive"}})
    with pytest.raises(check.CheckError):
        check.labels_of({"detail": {"code": "busy"}})


def test_golden_diff_lists_wrong_and_missing_sentences_only():
    expected = {"g01": ("positive", "lecturer"), "g02": ("negative", "facility")}
    assert check.golden_diff(expected, dict(expected)) == []
    got = {"g01": ("neutral", "lecturer")}
    assert check.golden_diff(expected, got) == [
        {"id": "g01", "expected": ["positive", "lecturer"], "got": ["neutral", "lecturer"]},
        {"id": "g02", "expected": ["negative", "facility"], "got": None},
    ]


def test_probability_gap_is_the_largest_difference_over_both_heads():
    first = analysis(boost=0.0)
    second = analysis(boost=0.05)
    assert check.prob_gap(first, first) == 0.0
    assert check.prob_gap(first, second) == pytest.approx(0.05)
    far = analysis(SENTIMENT[0], TOPIC[0])  # đổi nhãn cảm xúc: 0,7 so với 0,1 ở cả hai lớp
    assert check.prob_gap(analysis(SENTIMENT[2]), far) == pytest.approx(0.6)
    assert check.prob_gap({}, first) == 0.0  # thiếu xác suất thì không có gì để so


# --- So khớp 200 câu val --------------------------------------------------------------------------


def labels_with_differences(count: int, n: int = 200, heads: str = "sentiment"):
    reference = val_labels(n)
    candidate = list(reference)
    for i in range(count):
        sentiment, topic = candidate[i]
        flipped_sentiment = SENTIMENT[(SENTIMENT.index(sentiment) + 1) % 3]
        flipped_topic = TOPIC[(TOPIC.index(topic) + 1) % 4]
        candidate[i] = (
            flipped_sentiment if heads in ("sentiment", "both") else sentiment,
            flipped_topic if heads in ("topic", "both") else topic,
        )
    return reference, candidate


def test_agreement_counts_each_head_and_both_and_keeps_the_val_row_of_each_difference():
    reference, candidate = labels_with_differences(3, n=10, heads="sentiment")
    candidate[5] = (reference[5][0], TOPIC[(TOPIC.index(reference[5][1]) + 1) % 4])
    rows = [100 + 7 * i for i in range(10)]
    result = check.compare_runs(reference, candidate, rows)
    assert (result["n"], result["same_both"]) == (10, 6)
    assert (result["same_sentiment"], result["same_topic"]) == (7, 9)
    assert [d["val_row"] for d in result["differences"]] == [100, 107, 114, 135]
    assert result["differences"][0]["arm64"] == list(reference[0])
    assert result["differences"][0]["amd64"] == list(candidate[0])
    assert result["rate_pct"] == 60.0


@pytest.mark.parametrize(
    ("differences", "go"),
    [(0, True), (4, True), (5, False), (30, False)],  # 98% của 200 câu là 196 câu
)
def test_threshold_is_98_percent_of_the_two_hundred_sentences(differences, go):
    reference, candidate = labels_with_differences(differences, heads="both")
    result = check.compare_runs(reference, candidate, list(range(200)))
    assert result["needed"] == 196
    assert result["same_both"] == 200 - differences
    expected = golden_expected()
    full = check.evaluate(
        expected,
        list(range(200)),
        responses_for(expected, reference),
        responses_for(expected, candidate),
    )
    assert check.is_go(check.verdict_rows(full)) is go


def test_needed_count_rounds_up_for_other_sample_sizes():
    for n, needed in ((201, 197), (100, 98), (1, 1)):
        reference, candidate = labels_with_differences(0, n=n)
        assert check.compare_runs(reference, candidate, list(range(n)))["needed"] == needed


def test_runs_with_different_lengths_cannot_be_compared():
    with pytest.raises(check.CheckError, match="cùng số câu"):
        check.compare_runs([("positive", "lecturer")], [], [0])


# --- Chấm cả hai lượt -----------------------------------------------------------------------------


def test_golden_must_be_all_correct_even_when_every_val_sentence_agrees():
    expected = golden_expected()
    val = val_labels()
    wrong = responses_for(expected, val)
    first_id = next(iter(expected))
    wrong[0] = analysis("negative" if expected[first_id][0] != "negative" else "neutral")
    result = check.evaluate(expected, list(range(200)), responses_for(expected, val), wrong)
    assert [m["id"] for m in result["golden"]["amd64_mismatches"]] == [first_id]
    assert result["golden"]["arm64_mismatches"] == []
    assert result["val"]["same_both"] == 200
    verdict = check.verdict_rows(result)
    assert [row["ok"] for row in verdict] == [False, True]
    assert verdict[0]["value"] == "19/20" and verdict[0]["threshold"] == "20/20"
    assert not check.is_go(verdict)


def test_identical_runs_pass_with_zero_probability_gap():
    expected = golden_expected()
    runs = responses_for(expected, val_labels())
    result = check.evaluate(expected, list(range(200)), runs, [dict(r) for r in runs])
    verdict = check.verdict_rows(result)
    assert check.is_go(verdict)
    assert verdict[0]["value"] == "20/20" and verdict[1]["value"] == "100.0% (200/200)"
    assert result["prob_gap"] == {"max": 0.0, "mean": 0.0}


def test_probability_drift_is_reported_but_does_not_fail_the_check():
    expected = golden_expected()
    arm64 = responses_for(expected, val_labels())
    amd64 = [analysis(s, t, boost=0.02) for s, t in [*expected.values(), *val_labels()]]
    result = check.evaluate(expected, list(range(200)), arm64, amd64)
    assert result["prob_gap"]["max"] == pytest.approx(0.02)
    assert check.is_go(check.verdict_rows(result))


def test_evaluate_rejects_a_wrong_number_of_replies():
    expected = golden_expected()
    runs = responses_for(expected, val_labels())
    with pytest.raises(check.CheckError, match="Số phản hồi"):
        check.evaluate(expected, list(range(200)), runs, runs[:-1])
    with pytest.raises(check.CheckError, match="Số phản hồi"):
        check.evaluate(expected, list(range(199)), runs, runs)


# --- Golden từ bundle và fixture ------------------------------------------------------------------


def test_golden_labels_come_from_the_bundle_and_texts_from_the_fixture(tmp_path):
    fixture = json.loads(check.GOLDEN_SENTENCES.read_text(encoding="utf-8"))["items"]
    stored = {
        "items": [
            {"id": item["id"], "sentiment": "positive", "topic": "lecturer"} for item in fixture
        ]
    }
    (tmp_path / "golden_predictions.json").write_text(json.dumps(stored), encoding="utf-8")
    expected, texts = check.load_golden(tmp_path)
    assert len(expected) == len(texts) == 20
    assert set(expected.values()) == {("positive", "lecturer")}
    assert texts == [item["text"] for item in fixture]


def test_golden_with_an_id_missing_from_the_fixture_is_an_error(tmp_path):
    stored = {"items": [{"id": "g99", "sentiment": "positive", "topic": "lecturer"}]}
    (tmp_path / "golden_predictions.json").write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(check.CheckError, match="g99"):
        check.load_golden(tmp_path)


# --- Hỏi một máy chủ ------------------------------------------------------------------------------


class AnalyzeStub:
    """Máy chủ giả trả nhãn theo thứ tự yêu cầu, hoặc một mã lỗi cố định."""

    def __init__(self, replies: list[dict], status: int = 200):
        self.replies, self.status, self.asked = replies, status, []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                stub.asked.append(json.loads(self.rfile.read(length))["text"])
                reply = stub.replies[len(stub.asked) - 1] if stub.status == 200 else {"detail": {}}
                body = json.dumps(reply).encode()
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "AnalyzeStub":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


def test_every_text_is_sent_in_order_and_the_replies_come_back_in_order():
    replies = [analysis(SENTIMENT[i % 3], TOPIC[i % 4]) for i in range(3)]
    with AnalyzeStub(replies) as stub:
        got = check.fetch_analyses(stub.base, ["một", "hai", "ba"], timeout=5)
    assert stub.asked == ["một", "hai", "ba"] and got == replies


def test_an_error_status_stops_the_check_with_the_status_in_the_message():
    with AnalyzeStub([], status=500) as stub, pytest.raises(check.CheckError, match="500"):
        check.fetch_analyses(stub.base, ["một"], timeout=5)


# --- Dựng image và dòng lệnh ----------------------------------------------------------------------


def test_the_amd64_build_uses_the_bundle_context_and_skips_the_demo_download(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(check.subprocess, "run", fake_run)
    settings = load_settings()
    args = SimpleNamespace(bundle_version="v1.0", amd64_image="edupulse:v1.0-amd64")
    assert check.build_amd64(args, settings) >= 0
    command = seen["command"]
    assert command[:3] == ["docker", "build", "--platform=linux/amd64"]
    assert "EDUPULSE_BUILD_DEMO=0" in command and "BUNDLE_VERSION=v1.0" in command
    assert f"artifacts={settings.artifacts_dir('v1.0')}" in command
    assert command[-3:] == ["-t", "edupulse:v1.0-amd64", "."] and seen["cwd"] == REPO_ROOT


def test_a_failed_amd64_build_is_a_check_error(monkeypatch):
    monkeypatch.setattr(check.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    args = SimpleNamespace(bundle_version="v1.0", amd64_image="edupulse:v1.0-amd64")
    with pytest.raises(check.CheckError, match="docker build"):
        check.build_amd64(args, load_settings())


def test_image_names_default_from_the_bundle_version(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_amd64.py", "--bundle-version", "v2.1"])
    args = check.parse_args(load_settings())
    assert (args.arm64_image, args.amd64_image) == ("edupulse:v2.1", "edupulse:v2.1-amd64")
    assert (args.n, args.seed, args.build) == (200, 0, False)
    monkeypatch.setattr(sys, "argv", ["check_amd64.py", "--arm64-image", "x:1", "--build"])
    args = check.parse_args(load_settings())
    assert args.amd64_image == "x:1-amd64" and args.build is True


# --- Báo cáo --------------------------------------------------------------------------------------


def report_for(result: dict, build_s: float | None = 812.5) -> dict:
    side = {
        "image": "edupulse:v1.0",
        "platform": "linux/arm64",
        "startup_s": 2.4,
        "requests_s": 9.8,
        "machine": "aarch64",
        "cpu_flags": ["asimddp"],
        "memory_peak_mb": 640.2,
    }
    verdict = check.verdict_rows(result)
    return {
        "date": "2026-10-19",
        "commit": "abc1234",
        "bundle_version": "v1.0",
        "host": {"platform": "Darwin arm64", "docker_server": "29.8.2", "docker_cpus": 10},
        "sample": {"seed": 0, "val_available": 1583, "val_n": 200},
        "build_s": build_s,
        "limits": {"cpus": 2.0, "memory": "1536m"},
        "containers": {
            "arm64": side,
            "amd64": {**side, "image": "edupulse:v1.0-amd64", "machine": "x86_64", "cpu_flags": []},
        },
        **result,
        "verdict": verdict,
        "go": check.is_go(verdict),
    }


def test_report_lists_val_rows_of_the_differences_but_never_sentence_text():
    expected = golden_expected()
    reference, candidate = labels_with_differences(2, heads="both")
    result = check.evaluate(
        expected,
        [3 * i + 11 for i in range(200)],
        responses_for(expected, reference),
        responses_for(expected, candidate),
    )
    report = report_for(result)
    markdown = check.render_markdown(report)
    assert "Kết luận: **ĐẠT**" in markdown
    assert "| 11 |" in markdown and "| 14 |" in markdown  # dòng val của hai câu lệch
    assert "198/200" in markdown and "cần ít nhất 196" in markdown
    assert "x86_64" in markdown and "giả lập" in markdown and "812.5 s" in markdown
    dumped = json.dumps(report, ensure_ascii=False)
    assert "text" not in dumped and "sentence" not in dumped  # chỉ nhãn và số dòng


def test_report_says_not_passed_and_names_the_golden_sentence_that_differs():
    expected = golden_expected()
    runs = responses_for(expected, val_labels())
    broken = list(runs)
    broken[2] = analysis("neutral" if expected["g03"][0] != "neutral" else "negative")
    result = check.evaluate(expected, list(range(200)), runs, broken)
    markdown = check.render_markdown(report_for(result))
    assert "Kết luận: **KHÔNG ĐẠT**" in markdown
    assert "## Câu golden lệch nhãn trên amd64" in markdown and "| g03 |" in markdown
    assert "## Câu golden lệch nhãn trên arm64" not in markdown  # arm64 khớp golden của chính nó


def test_report_without_a_build_in_this_run_has_no_build_line():
    expected = golden_expected()
    runs = responses_for(expected, val_labels())
    result = check.evaluate(expected, list(range(200)), runs, runs)
    markdown = check.render_markdown(report_for(result, build_s=None))
    assert "Dựng image amd64 mất" not in markdown
    assert "Câu val lệch nhãn" not in markdown
