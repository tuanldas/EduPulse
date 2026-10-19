"""Cổng go/no-go của `scripts/smoke_test.py`: chấm điểm, báo cáo và phép đo, không cần container."""

import importlib.util
import json
import math
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from edupulse.config import MAX_ROWS, REPO_ROOT, load_settings

SCRIPT = REPO_ROOT / "scripts" / "smoke_test.py"


def load_script():
    spec = importlib.util.spec_from_file_location("smoke_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["smoke_test"] = module  # dataclass cần module đã đăng ký
    spec.loader.exec_module(module)
    return module


smoke = load_script()


def gate(
    label: float | None = 0.2,
    explain: float | None = 1.0,
    *,
    cap: int | None = 25,
    upload: dict | None = None,
    batch: float | None = 20.0,
    startup: float | None = 3.0,
    peak: float | None = 600.0,
    checks: dict | None = None,
    oom: bool = False,
    compare: dict | None | str = "default",
) -> dict:
    """Một lượt đo hợp lệ ở dạng mà `verdict_rows` đọc (`compare=None`: chưa đo so sánh)."""

    def summary(p95):
        return {"n": 30, "p50_s": 0.1, "p95_s": p95, "max_s": 9.9}

    result: dict = {
        "label": "gate",
        "startup_s": startup,
        "latency_idle": {
            "analyze": summary(label),
            "explain": summary(explain),
            "analyze_explain": summary(3.0),
            "not_explained": 2,
        },
        "bench": {"total_s": batch, "cgroup_memory_peak_mb": peak, "oom_killed": oom},
        "server_memory_peak_mb": peak,
        "server_state": {"oom_killed": oom},
        "checks": {"a": True, "b": True} if checks is None else checks,
    }
    if cap is not None:
        result["limits"] = {"xai_max_words": cap, "source": "/api/config"}
    if upload is not None:
        result["upload"] = upload
    if compare == "default":
        compare = dict(GOOD_COMPARE)
    if compare is not None:
        result["compare"] = compare
    return result


GOOD_UPLOAD = {"bytes": 5 * 1048576, "rows": 2000, "upload_s": 0.4, "run_s": 31.5, "ok": True}
GOOD_COMPARE = {
    "ok": True,
    "cold_s": 1.4,
    "warm_s": 0.06,
    "predict_ms": 5.5,
    "rss_before_mb": 540.0,
    "rss_after_mb": 668.0,
    "rss_delta_mb": 128.0,
    "load_s": 1.2,
}


def rows_by_metric(verdict: list[dict]) -> dict[str, dict]:
    return {row["metric"]: row for row in verdict}


def row_starting(verdict: list[dict], prefix: str) -> dict:
    (row,) = [r for r in verdict if r["metric"].startswith(prefix)]
    return row


# --- Ngưỡng ---------------------------------------------------------------------------------------


def test_thresholds_are_the_pdr_5_numbers_and_the_old_combined_gate_is_gone():
    assert smoke.GO["label_p95_s"] == 0.5
    assert smoke.GO["explain_p95_s"] == 2.0
    assert "single_p95_s" not in smoke.GO
    assert (smoke.GO["memory_peak_mb"], smoke.GO["batch_total_s"], smoke.GO["startup_s"]) == (
        1200,
        30.0,
        20.0,
    )
    assert smoke.UPLOAD_MB == 5


# --- p95 và tóm tắt -------------------------------------------------------------------------------


def test_summary_without_samples_has_no_numbers_instead_of_crashing():
    assert smoke.summarize([]) == {"n": 0, "p50_s": None, "p95_s": None, "max_s": None}


def test_p95_is_the_nearest_rank_value():
    values = [i / 100 for i in range(1, 31)]  # 0,01 … 0,30
    summary = smoke.summarize(values)
    assert summary["n"] == 30 and summary["p95_s"] == 0.29  # lớn thứ hai
    assert summary["max_s"] == 0.3 and summary["p50_s"] == 0.155
    assert smoke.summarize([0.5])["p95_s"] == 0.5


# --- Bảng go/no-go -------------------------------------------------------------------------------


def test_label_and_explanation_are_two_separate_rows_with_their_own_p95():
    verdict = smoke.verdict_rows(gate(label=0.31, explain=1.7))
    label = row_starting(verdict, "Nhãn")
    explain = row_starting(verdict, "Giải thích")
    assert (label["value"], label["threshold"], label["ok"]) == (0.31, "≤ 0.5", True)
    assert (explain["value"], explain["threshold"], explain["ok"]) == (1.7, "≤ 2", True)
    assert "/api/analyze" in label["metric"] and "/api/explain" in explain["metric"]
    assert not any("1 câu + giải thích" in r["metric"] for r in verdict)


def test_the_explanation_row_names_the_word_limit_read_from_the_server():
    assert "câu ≤ 25 từ" in row_starting(smoke.verdict_rows(gate(cap=25)), "Giải thích")["metric"]
    assert "câu ≤ 40 từ" in row_starting(smoke.verdict_rows(gate(cap=40)), "Giải thích")["metric"]
    assert "câu ≤ ? từ" in row_starting(smoke.verdict_rows(gate(cap=None)), "Giải thích")["metric"]


@pytest.mark.parametrize(
    ("label", "explain", "label_ok", "explain_ok"),
    [
        (0.5, 2.0, True, True),  # đúng ngưỡng vẫn đạt
        (0.501, 1.0, False, True),
        (0.1, 2.001, True, False),
        (0.9, 5.0, False, False),
    ],
)
def test_each_p95_is_judged_only_against_its_own_threshold(label, explain, label_ok, explain_ok):
    verdict = smoke.verdict_rows(gate(label=label, explain=explain))
    assert row_starting(verdict, "Nhãn")["ok"] is label_ok
    assert row_starting(verdict, "Giải thích")["ok"] is explain_ok
    assert smoke.is_go(verdict) is (label_ok and explain_ok)


def test_a_slow_explanation_does_not_fail_the_label_and_vice_versa():
    slow_explain = smoke.verdict_rows(gate(label=0.05, explain=3.5))
    assert row_starting(slow_explain, "Nhãn")["ok"] is True
    assert smoke.is_go(slow_explain) is False
    slow_label = smoke.verdict_rows(gate(label=0.7, explain=0.4))
    assert row_starting(slow_label, "Giải thích")["ok"] is True
    assert smoke.is_go(slow_label) is False


def test_the_m1_result_that_failed_before_now_passes_the_split_criteria():
    # Số đo chính thức ngày 15/10: cặp 1,089 s trượt ngưỡng 0,6 s; tách ra thì nhãn 0,052 s đạt và
    # giải thích 0,994 s đạt ngưỡng 2 s.
    verdict = smoke.verdict_rows(gate(label=0.052, explain=0.994, upload=GOOD_UPLOAD))
    assert smoke.is_go(verdict)


def test_unmeasured_values_are_not_failures():
    verdict = smoke.verdict_rows(gate(label=None, explain=None, batch=None, startup=None))
    assert row_starting(verdict, "Nhãn")["ok"] is None
    assert row_starting(verdict, "Giải thích")["ok"] is None
    assert row_starting(verdict, "1.000 câu")["ok"] is None
    assert smoke.is_go(verdict)


def test_other_gates_are_kept_with_their_thresholds():
    verdict = smoke.verdict_rows(gate(peak=1201.0, batch=30.1, startup=20.5, oom=True))
    by_metric = rows_by_metric(verdict)
    assert row_starting(verdict, "memory.peak")["ok"] is False
    assert by_metric["1.000 câu end-to-end (s)"]["ok"] is False
    assert by_metric["Khởi động tới health 200 (s)"]["ok"] is False
    assert by_metric["OOM-kill"]["ok"] is False
    good_rows = smoke.verdict_rows(gate(peak=1200.0, batch=30.0, startup=20.0))
    good = rows_by_metric(good_rows)
    assert row_starting(good_rows, "memory.peak")["ok"] is True
    assert good["1.000 câu end-to-end (s)"]["ok"] and good["Khởi động tới health 200 (s)"]["ok"]
    assert good["OOM-kill"]["ok"]


def test_ram_gate_also_covers_the_process_after_the_baseline_is_loaded():
    ok = row_starting(smoke.verdict_rows(gate()), "RSS tiến trình")
    assert (ok["value"], ok["threshold"], ok["ok"]) == (668.0, "≤ 1200", True)
    at_limit = dict(GOOD_COMPARE, rss_after_mb=1200.0)
    assert row_starting(smoke.verdict_rows(gate(compare=at_limit)), "RSS tiến trình")["ok"] is True
    over = dict(GOOD_COMPARE, rss_after_mb=1200.5)
    verdict = smoke.verdict_rows(gate(compare=over))
    assert row_starting(verdict, "RSS tiến trình")["ok"] is False and not smoke.is_go(verdict)


def test_peak_memory_row_says_it_includes_the_comparison():
    assert "đã gồm so sánh" in row_starting(smoke.verdict_rows(gate()), "memory.peak")["metric"]


def test_cold_load_time_is_shown_for_reference_and_never_fails_the_run():
    row = row_starting(smoke.verdict_rows(gate()), "So sánh mô hình")
    assert (row["value"], row["threshold"], row["ok"]) == (1.4, smoke.INFO, None)
    slow = dict(GOOD_COMPARE, cold_s=60.0)
    assert smoke.is_go(smoke.verdict_rows(gate(compare=slow)))
    lines = smoke.render_markdown(report_of(gate())).splitlines()
    table_row = [line for line in lines if line.startswith("| So sánh mô hình")]
    assert table_row and table_row[0].endswith("| tham khảo |")


def test_missing_comparison_numbers_are_not_failures_but_a_broken_comparison_is():
    verdict = smoke.verdict_rows(gate(compare=None))
    assert row_starting(verdict, "RSS tiến trình")["ok"] is None and smoke.is_go(verdict)
    broken = smoke.verdict_rows(gate(checks={"compare_with_baseline_works": False}))
    assert row_starting(broken, "Hành vi")["ok"] is False and not smoke.is_go(broken)


def test_behaviour_and_log_checks_must_all_pass():
    failing = smoke.verdict_rows(gate(checks={"root_is_html": True, "logs_have_no_content": False}))
    row = row_starting(failing, "Hành vi")
    assert (row["value"], row["ok"]) == ("1/2", False)
    assert not smoke.is_go(failing)


def test_upload_row_requires_the_five_megabyte_batch_to_finish():
    ok = row_starting(smoke.verdict_rows(gate(upload=GOOD_UPLOAD)), "Upload")
    assert ok["ok"] is True and ok["value"] == "5.0 MB, 2000 dòng, chạy 31.5 s"
    assert "5MB" in ok["metric"]
    bad = dict(GOOD_UPLOAD, ok=False, problems=["batch kết thúc ở trạng thái failed"])
    failed = smoke.verdict_rows(gate(upload=bad))
    assert row_starting(failed, "Upload")["ok"] is False and not smoke.is_go(failed)
    assert row_starting(smoke.verdict_rows(gate()), "Upload")["ok"] is None  # chưa đo


def test_p95_of_reads_missing_pieces_as_unmeasured():
    assert smoke.p95_of({}, "latency_idle", "analyze") is None
    assert (
        smoke.p95_of({"latency_idle": {"analyze": {"p95_s": 0.2}}}, "latency_idle", "analyze")
        == 0.2
    )
    assert (
        smoke.p95_of({"latency_idle": {"not_explained": 3}}, "latency_idle", "not_explained")
        is None
    )


# --- Báo cáo markdown -----------------------------------------------------------------------------


def report_of(main_gate: dict, extra: list[dict] | None = None) -> dict:
    verdict = smoke.verdict_rows(main_gate)
    return {
        "date": "2026-10-19",
        "arch": "arm64",
        "commit": "abc1234",
        "host": {"platform": "Darwin arm64", "loadavg_start": [1, 1, 1], "loadavg_end": [2, 2, 2]},
        "image": None,
        "runs": [main_gate, *(extra or [])],
        "verdict": verdict,
        "go": smoke.is_go(verdict),
    }


def test_report_states_both_p95_numbers_apart_with_thresholds_and_sample_sizes():
    main = gate(label=0.052, explain=0.994, cap=25, upload=GOOD_UPLOAD)
    markdown = smoke.render_markdown(report_of(main))
    assert "Kết luận: **GO**" in markdown
    assert (
        "**Nhãn**, p95 của /api/analyze: **0.052 s** (ngưỡng ≤ 0.5), trên 30 câu thật" in markdown
    )
    assert "**Giải thích**, p95 của /api/explain: **0.994 s** (ngưỡng ≤ 2)" in markdown
    assert "trên 30 câu không quá 25 từ" in markdown
    assert "2/30 câu mẫu dài hơn giới hạn nên không được giải thích" in markdown
    assert "/api/config" in markdown
    assert "1 câu + giải thích" not in markdown


def test_report_is_no_go_when_one_of_the_two_fails_and_says_which():
    markdown = smoke.render_markdown(report_of(gate(label=0.2, explain=2.4)))
    assert "Kết luận: **NO-GO**" in markdown
    table = [line for line in markdown.splitlines() if line.startswith("| Giải thích")]
    assert table and "KHÔNG ĐẠT" in table[0]
    label_line = [line for line in markdown.splitlines() if line.startswith("| Nhãn")]
    assert label_line and "ĐẠT" in label_line[0] and "KHÔNG" not in label_line[0]


def test_latency_table_names_the_commands_and_skips_the_counter():
    main = gate()
    main["latency_under_load"] = main["latency_idle"]
    rows = smoke.latency_rows(main)
    commands = {row[1] for row in rows}
    assert commands == {
        "nhãn (/api/analyze)",
        "giải thích (/api/explain)",
        "cặp song song như giao diện (tham khảo)",
    }
    assert len(rows) == 6


def test_report_has_a_comparison_section_with_cold_time_and_memory():
    markdown = smoke.render_markdown(report_of(gate(peak=701.5)))
    assert "## So sánh với mô hình cơ sở" in markdown
    assert "(nạp lạnh): 1.4 s; lần sau 0.06 s; mô hình cơ sở tự báo 5.5 ms" in markdown
    assert "Nạp scikit-learn và hai mô hình skops: 1.2 s theo log" in markdown
    assert "RSS tiến trình: 540.0 MB → 668.0 MB (+128.0 MB)" in markdown
    assert "memory.peak của container cuối lượt đo: 701.5 MB, tính cả phần nạp" in markdown
    assert "không chạy được" not in markdown
    broken = gate(compare=dict(GOOD_COMPARE, ok=False))
    assert "so sánh với mô hình cơ sở không chạy được" in smoke.render_markdown(report_of(broken))
    assert "## So sánh với mô hình cơ sở" not in smoke.render_markdown(
        report_of(gate(compare=None))
    )


def test_report_has_an_upload_section_with_the_problems():
    upload = dict(GOOD_UPLOAD, ok=False, problems=["tiến độ bị giảm"], flagged=210)
    markdown = smoke.render_markdown(report_of(gate(upload=upload)))
    assert "## Upload 5MB" in markdown and "KHÔNG ĐẠT" in markdown
    assert "- Lỗi: tiến độ bị giảm" in markdown


def test_one_cpu_reference_lists_the_two_p95_separately():
    one_cpu = gate(label=0.4, explain=3.3)
    one_cpu.update(cpus=1.0, startup_s=2.5)
    markdown = smoke.render_markdown(report_of(gate(), [one_cpu]))
    assert "nhãn p95 0.4 s; giải thích p95 3.3 s" in markdown


def test_report_footnote_says_the_word_limit_is_settled_on_the_vps():
    markdown = smoke.render_markdown(report_of(gate()))
    assert "XAI_MAX_WORDS" in markdown and "--url" in markdown


# --- Giờ từng pha và việc nặng của người khác -----------------------------------------------------


def with_phases(run: dict, *phases: tuple[str, list[str] | None]) -> dict:
    run["phases"] = [
        {"name": name, "start": "14:40:00", "end": "14:40:30", "seconds": 30.0, "noisy_with": seen}
        for name, seen in phases
    ]
    return run


NOISE_WINDOW = {"start": "14:42:30", "end": "14:42:36", "kinds": ["node", "pnpm-native"]}


def test_report_lists_every_phase_with_its_times_and_marks_the_noisy_ones():
    main = with_phases(gate(), ("Khởi động", []), ("Bench 1000 câu", ["node", "pnpm-native"]))
    report = report_of(main)
    report["host"].update(
        started="14:39:12",
        ended="14:51:00",
        heavy_watch={"samples": 100, "busy_samples": 3, "windows": [NOISE_WINDOW]},
    )
    markdown = smoke.render_markdown(report)
    assert "## Giờ từng pha đo" in markdown and "Lượt `gate`:" in markdown
    assert "| Khởi động | 14:40:00 | 14:40:30 | 30.0 | không |" in markdown
    assert "| Bench 1000 câu | 14:40:00 | 14:40:30 | 30.0 | CÓ: node, pnpm-native |" in markdown
    assert "Giờ đo 14:39:12 → 14:51:00" in markdown
    assert "có việc nặng khác chạy cùng** lúc 14:42:30–14:42:36 (node, pnpm-native)" in markdown
    assert "3/100 lần lấy mẫu" in markdown
    warning = [line for line in markdown.splitlines() if "Cảnh báo nhiễu" in line]
    assert len(warning) == 1 and "Bench 1000 câu" in warning[0] and "Khởi động" not in warning[0]
    assert "Kết luận: **GO**" in markdown  # nhiễu chỉ cảnh báo, không đổi kết luận


def test_report_names_each_run_in_its_own_phase_table():
    main = with_phases(gate(), ("Khởi động", []))
    main["label"] = "gate"
    one_cpu = with_phases(gate(label=0.4, explain=3.3), ("Khởi động", ["pytest"]))
    one_cpu.update(label="1cpu", cpus=1.0)
    markdown = smoke.render_markdown(report_of(main, [one_cpu]))
    assert "Lượt `gate`:" in markdown and "Lượt `1cpu`:" in markdown
    assert "CÓ: pytest" in markdown
    assert "Cảnh báo nhiễu" not in markdown  # chỉ lượt đo chính được cảnh báo


def test_report_says_when_the_machine_was_clean_and_when_it_was_not_watched():
    clean = report_of(with_phases(gate(), ("Khởi động", [])))
    clean["host"]["heavy_watch"] = {"samples": 150, "busy_samples": 0, "windows": []}
    markdown = smoke.render_markdown(clean)
    assert "không thấy việc nặng khác chạy cùng (150 lần lấy mẫu)" in markdown
    assert "Cảnh báo nhiễu" not in markdown
    unwatched = smoke.render_markdown(report_of(with_phases(gate(), ("Khởi động", None))))
    assert "chưa theo dõi việc nặng khác" in unwatched
    assert "| Khởi động | 14:40:00 | 14:40:30 | 30.0 | - |" in unwatched
    assert "## Giờ từng pha đo" not in smoke.render_markdown(report_of(gate()))  # chưa có pha nào


def test_the_heavy_pattern_matches_the_four_tools_and_never_its_own_command_line():
    for command in (
        "node /x/vitest run",
        "pnpm exec playwright test",
        "pytest -q tests",
        "node /x/next build",
    ):
        assert re.search(smoke.HEAVY_PATTERN, command), command
    assert not re.search(smoke.HEAVY_PATTERN, f"pgrep -f {smoke.HEAVY_PATTERN}")
    assert not re.search(smoke.HEAVY_PATTERN, "uv run python scripts/smoke_test.py --also-1cpu")


def test_heavy_processes_names_the_executables_once_each(monkeypatch):
    answers = iter(
        [
            SimpleNamespace(stdout="11\n12\n13\n"),
            SimpleNamespace(stdout="/usr/bin/node\n  pytest \nnode\n"),
        ]
    )
    monkeypatch.setattr(smoke.subprocess, "run", lambda *args, **kwargs: next(answers))
    assert smoke.heavy_processes() == ["node", "pytest"]


def test_heavy_processes_is_empty_when_nothing_matches_or_pgrep_is_missing(monkeypatch):
    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    assert smoke.heavy_processes() == []

    def missing(*args, **kwargs):
        raise FileNotFoundError("pgrep")

    monkeypatch.setattr(smoke.subprocess, "run", missing)
    assert smoke.heavy_processes() == []


def watch_with(samples: list[tuple[float, list[str]]]) -> "smoke.BusyWatch":
    watch = smoke.BusyWatch(period=2.0)
    watch.samples = samples
    return watch


def test_a_phase_sees_the_tasks_running_inside_it_and_the_ones_already_running_at_its_start():
    watch = watch_with([(100.0, []), (102.0, ["node"]), (104.0, ["node", "vitest"]), (106.0, [])])
    assert watch.seen_between(103.0, 105.0) == ["node", "vitest"]
    assert watch.seen_between(103.9, 103.95) == [
        "node"
    ]  # chạy từ trước và chưa xong lúc pha bắt đầu
    assert watch.seen_between(100.0, 101.0) == []
    assert watch.seen_between(106.5, 120.0) == []


def test_busy_windows_merge_neighbouring_samples_and_split_on_a_real_gap():
    watch = watch_with(
        [
            (100.0, []),
            (102.0, ["node"]),
            (104.0, ["vitest"]),
            (106.0, []),
            (108.0, []),
            (110.0, ["node"]),
            (118.0, ["node"]),
        ]
    )
    assert watch.busy_windows() == [
        {"start": smoke.clock(102.0), "end": smoke.clock(104.0), "kinds": ["node", "vitest"]},
        {"start": smoke.clock(110.0), "end": smoke.clock(110.0), "kinds": ["node"]},
        {"start": smoke.clock(118.0), "end": smoke.clock(118.0), "kinds": ["node"]},
    ]
    summary = watch.summary()
    assert (summary["samples"], summary["busy_samples"]) == (7, 4)
    assert watch_with([]).busy_windows() == [] and watch_with([(1.0, [])]).busy_windows() == []


def test_a_running_watch_keeps_sampling_until_it_is_closed():
    with smoke.BusyWatch(probe=lambda: ["node"], period=0.01) as watch:
        time.sleep(0.15)
    taken = len(watch.samples)
    assert taken >= 3 and watch.summary()["busy_samples"] == taken
    time.sleep(0.05)
    assert len(watch.samples) == taken  # đã dừng


class FakeWatch:
    """Trả lần lượt các câu trả lời đã định cho từng pha; giờ bắt đầu không sau giờ kết thúc."""

    def __init__(self, *answers: list[str]):
        self.answers = list(answers)

    def seen_between(self, start: float, end: float) -> list[str]:
        assert start <= end
        return self.answers.pop(0)


def test_timeline_records_each_phase_in_order_with_clock_times_and_the_noise_seen():
    timeline = smoke.Timeline(FakeWatch([], ["node"]))
    with timeline.phase("một"):
        pass
    with timeline.phase("hai"):
        time.sleep(0.15)
    first, second = timeline.phases
    assert [p["name"] for p in timeline.phases] == ["một", "hai"]
    assert (first["noisy_with"], second["noisy_with"]) == ([], ["node"])
    for phase in timeline.phases:
        assert re.fullmatch(r"\d\d:\d\d:\d\d", phase["start"])
        assert re.fullmatch(r"\d\d:\d\d:\d\d", phase["end"])
    assert second["seconds"] >= 0.1 and first["end"] <= second["start"]


def test_timeline_without_a_watch_leaves_the_noise_unknown_and_records_failed_phases():
    timeline = smoke.Timeline()
    with pytest.raises(RuntimeError), timeline.phase("lỗi"):
        raise RuntimeError("máy chủ không lên")
    (phase,) = timeline.phases
    assert phase["name"] == "lỗi" and phase["noisy_with"] is None


def test_service_measurement_records_idle_long_and_loaded_phases_in_order():
    near = [" ".join(["từ"] * 22) + f" {i}" for i in range(8)]
    timeline = smoke.Timeline()
    with StubService() as service:
        smoke.measure_service(service.base, [*near, SHORT], repeat=4, timeline=timeline)
    phases = timeline.phases
    assert len(phases) == 3 and "câu dài" in phases[1]["name"] and "tải" in phases[2]["name"]
    assert all(a["end"] <= b["start"] for a, b in zip(phases, phases[1:], strict=False))
    plain = smoke.Timeline()
    with StubService() as service:
        smoke.measure_service(service.base, [SHORT] * 4, repeat=4, timeline=plain)
    assert len(plain.phases) == 2  # không có câu dài sát giới hạn thì bỏ pha đó


# --- Tệp upload thử -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_sentences() -> list[str]:
    return [
        f"Thầy dạy số {i} rất nhiệt tình, dễ hiểu và có nhiều ví dụ thực tế." for i in range(50)
    ]


def test_upload_csv_has_the_exact_size_row_count_and_markers(sample_sentences):
    import csv
    import io

    size = 5 * 1024 * 1024
    data = smoke.build_upload_csv(sample_sentences, size, MAX_ROWS, smoke.MARKERS[3])
    assert len(data) == size and data.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
    assert rows[0] == ["Góp ý", "Nhóm", "Ghi chú"] and len(rows) == MAX_ROWS + 1
    assert all(r[1].startswith(smoke.MARKERS[3]) for r in rows[1:])
    assert max(len(r[2]) for r in rows[1:]) < 20_000  # dưới trần ký tự mỗi ô của parser
    assert size < 10 * 1024 * 1024


def test_upload_csv_is_deterministic_and_never_smaller_than_its_sentences(sample_sentences):
    first = smoke.build_upload_csv(sample_sentences, 1_000_000, 100, "NHOM")
    assert first == smoke.build_upload_csv(sample_sentences, 1_000_000, 100, "NHOM")
    tiny = smoke.build_upload_csv(sample_sentences, 10, 100, "NHOM")  # trần nhỏ hơn phần câu
    assert len(tiny) > 10


def test_multipart_body_frames_the_file_and_the_marker_filename():
    body, content_type = smoke.multipart_file("file", "ten.csv", b"a,b\r\n1,2\r\n")
    boundary = content_type.split("boundary=")[1]
    assert content_type.startswith("multipart/form-data; ")
    assert body.startswith(f"--{boundary}\r\n".encode()) and body.endswith(
        f"--{boundary}--\r\n".encode()
    )
    assert b'name="file"; filename="ten.csv"' in body and b"a,b\r\n1,2\r\n" in body
    assert smoke.multipart_file("file", "x.csv", b"")[1] != content_type  # ranh giới ngẫu nhiên


def test_log_check_catches_every_marker_including_the_filename_and_the_batch_id():
    clean = "POST /api/analyze 200 12ms\nPOST /api/batches 201 5ms\n"
    assert smoke.check_logs(clean, ("ID-THAT",)) == {
        "logs_have_no_request_content": True,
        "logs_have_route_template": True,
    }
    for leak in (*smoke.MARKERS, smoke.FILE_MARKER, "ID-THAT"):
        checks = smoke.check_logs(clean + f"lỗi với {leak}\n", ("ID-THAT",))
        assert checks["logs_have_no_request_content"] is False, leak


def test_sentences_come_from_a_file_one_per_line_without_blank_lines(tmp_path: Path):
    file = tmp_path / "sentences.txt"
    file.write_text("Câu một.\n\n  Câu hai.  \n", encoding="utf-8")
    assert smoke.load_sentences(str(file), SimpleNamespace(data_dir=tmp_path)) == [
        "Câu một.",
        "Câu hai.",
    ]


def test_missing_validation_set_is_a_clear_error_not_a_crash(tmp_path: Path):
    with pytest.raises(smoke.SmokeError, match="validation.parquet"):
        smoke.load_sentences(None, SimpleNamespace(data_dir=tmp_path))


def test_container_asks_for_a_platform_only_when_emulating(monkeypatch):
    commands: list[tuple] = []
    monkeypatch.setattr(smoke, "docker", lambda *args, **kwargs: commands.append(args) or "")
    smoke.Container("c", "img:1", 2.0, "1536m", 8000, platform="linux/amd64").start()
    smoke.Container("c", "img:1", 2.0, "1536m", 8000).start()
    emulated, native = [c for c in commands if c[0] == "run"]  # theo thứ tự đã gọi
    assert "--platform" not in native
    assert emulated[emulated.index("--platform") + 1] == "linux/amd64"
    assert emulated[-1] == "img:1" and "--memory=1536m" in emulated


# --- Phép đo với một máy chủ giả để thử công cụ đo ------------------------------------------------


class StubService:
    """HTTP server nhỏ giả lập hợp đồng của API để thử logic đo (không phải để lấy số đo)."""

    def __init__(
        self,
        cap: int | None = 25,
        config_status: int = 200,
        analyze_delay: float = 0.0,
        compare_status: int = 200,
        baseline_mb: float = 130.0,
    ):
        self.cap, self.config_status, self.analyze_delay = cap, config_status, analyze_delay
        self.compare_status, self.baseline_mb = compare_status, baseline_mb
        self.rss_mb = 520.0
        self.baseline_loaded = False
        self.calls: list[tuple[str, str]] = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status: int, payload: dict):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                service.calls.append(("GET", self.path))
                if self.path == "/api/config" and service.config_status == 200:
                    limits = {} if service.cap is None else {"xai_max_words": service.cap}
                    self._reply(200, {"limits": limits})
                elif self.path == "/api/health":
                    self._reply(200, {"status": "ok", "rss_mb": service.rss_mb})
                else:
                    self._reply(404, {"detail": {"code": "not_found", "message": "-"}})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length) or b"{}")
                text = request.get("text", "")
                service.calls.append(("POST", self.path))
                words = len(text.split())
                limit = service.cap if service.cap is not None else 25
                if self.path == "/api/analyze" and request.get("compare"):
                    if service.compare_status != 200:
                        self._reply(service.compare_status, {"detail": {"code": "unavailable"}})
                        return
                    if not service.baseline_loaded:  # nạp lười ở lần đầu: chậm hơn và tốn RAM
                        time.sleep(0.05)
                        service.baseline_loaded = True
                        service.rss_mb += service.baseline_mb
                    self._reply(
                        200, {"xai_available": words <= limit, "baseline": {"elapsed_ms": 4.2}}
                    )
                elif self.path == "/api/analyze":
                    time.sleep(service.analyze_delay)
                    self._reply(200, {"xai_available": words <= limit})
                elif self.path == "/api/explain" and words <= limit:
                    self._reply(200, {})
                else:
                    self._reply(422, {"detail": {"code": "xai_too_long", "message": "-"}})

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "StubService":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    def count(self, path: str) -> int:
        return sum(1 for _, p in self.calls if p == path)


SHORT = "Thầy dạy rất hay"  # 4 từ
LONG = " ".join(["từ"] * 30)  # 30 từ, quá giới hạn mặc định 25


def test_first_comparison_reports_cold_and_warm_time_and_the_memory_it_added():
    with StubService() as service:
        result = smoke.measure_compare(service.base, SHORT)
    assert result["ok"] is True
    assert result["cold_s"] >= 0.05 > result["warm_s"]  # lần đầu nạp lười nên chậm hơn hẳn
    assert result["predict_ms"] == 4.2
    assert (result["rss_before_mb"], result["rss_after_mb"], result["rss_delta_mb"]) == (
        520.0,
        650.0,
        130.0,
    )


def test_the_comparison_asks_twice_for_the_baseline_and_reads_memory_before_and_after():
    with StubService() as service:
        smoke.measure_compare(service.base, SHORT)
    assert service.calls == [
        ("GET", "/api/health"),
        ("POST", "/api/analyze"),
        ("POST", "/api/analyze"),
        ("GET", "/api/health"),
    ]


def test_a_server_without_the_baseline_makes_the_comparison_check_fail_not_crash():
    with StubService(compare_status=503) as service:
        result = smoke.measure_compare(service.base, SHORT)
    assert result["ok"] is False and result["predict_ms"] is None
    assert result["rss_delta_mb"] == 0.0  # không nạp được nên bộ nhớ không đổi


def test_baseline_load_time_is_read_from_the_server_log():
    logs = "INFO edupulse.engine: Đã nạp mô hình cơ sở trong 1.7 s\nPOST /api/analyze 200 9ms"
    assert smoke.baseline_load_seconds(logs) == 1.7
    assert smoke.baseline_load_seconds("POST /api/analyze 200 9ms") is None


def test_server_limits_come_from_the_config_of_the_service_being_measured():
    with StubService(cap=40) as service:
        assert smoke.server_limits(service.base) == {"xai_max_words": 40, "source": "/api/config"}


def test_server_limits_fall_back_to_the_local_settings_when_config_is_missing():
    with StubService(config_status=404) as service:
        found = smoke.server_limits(service.base)
    assert found == {"xai_max_words": load_settings().xai_max_words, "source": "cấu hình máy đo"}
    with StubService(cap=None) as service:  # có /api/config nhưng thiếu trường
        assert smoke.server_limits(service.base)["source"] == "cấu hình máy đo"


def test_explain_is_measured_only_on_sentences_the_server_says_it_can_explain():
    with StubService() as service:
        texts = [SHORT, LONG, SHORT, LONG, SHORT, SHORT]
        result = smoke.measure_latency(service.base, texts, repeat=6)
    assert result["analyze"]["n"] == 6  # nhãn đo trên mọi câu, kể cả câu dài
    assert result["explain"]["n"] == 4 and result["analyze_explain"]["n"] == 4
    assert result["not_explained"] == 2
    assert service.count("/api/explain") == 4 + 4  # lần lượt rồi lại song song, chỉ câu ngắn


def test_the_label_sample_is_larger_than_the_explanation_sample():
    texts = [SHORT + f" {i}" for i in range(120)] + [LONG + f" {i}" for i in range(30)]
    with StubService() as service:
        result = smoke.measure_latency(service.base, texts, repeat=8)
    assert result["analyze"]["n"] == smoke.LABEL_SAMPLES  # nhãn rẻ nên đo nhiều câu hơn
    assert result["explain"]["n"] == 8 and result["analyze_explain"]["n"] == 8
    assert result["not_explained"] == smoke.LABEL_SAMPLES - sum(
        1 for text in texts[: smoke.LABEL_SAMPLES] if len(text.split()) <= 25
    )


def test_label_and_explanation_get_their_own_p95():
    with StubService(analyze_delay=0.02) as service:
        result = smoke.measure_latency(service.base, [SHORT, LONG], repeat=6)
    assert result["analyze"]["p50_s"] >= 0.02
    assert result["explain"]["p95_s"] is not None and result["analyze"]["p95_s"] is not None
    assert result["explain"]["n"] != result["analyze"]["n"]  # hai mẫu khác nhau, hai p95 khác nhau


def test_a_service_that_explains_nothing_stops_the_measurement():
    texts = [LONG] * 40
    with StubService() as service, pytest.raises(smoke.SmokeError, match="≤ 25 từ"):
        smoke.measure_service(service.base, texts, repeat=5)


def test_measure_service_reports_the_limit_and_both_latencies_per_situation():
    texts = [SHORT + f" {i}" for i in range(28)] + [LONG + f" {i}" for i in range(2)]
    with StubService(cap=25) as service:
        result = smoke.measure_service(service.base, texts, repeat=6)
    assert result["limits"] == {"xai_max_words": 25, "source": "/api/config"}
    for situation in ("latency_idle", "latency_under_load"):
        assert result[situation]["analyze"]["n"] == 30  # nhãn đo trên cả 30 câu mẫu
        assert result[situation]["explain"]["n"] <= 6
    assert result["load"]["workers"] == 2 and result["load"]["errors"] == 0
    verdict = smoke.verdict_rows(result)  # đọc được đúng dạng dữ liệu mà phép đo sinh ra
    assert row_starting(verdict, "Nhãn")["value"] == result["latency_idle"]["analyze"]["p95_s"]
    assert (
        row_starting(verdict, "Giải thích")["value"] == result["latency_idle"]["explain"]["p95_s"]
    )


def test_near_limit_sentences_are_measured_separately_when_there_are_enough():
    near = [" ".join(["từ"] * 22) + f" {i}" for i in range(8)]
    with StubService() as service:
        result = smoke.measure_service(service.base, [*near, SHORT], repeat=4)
    assert "latency_idle_long" in result and result["latency_idle_long"]["explain"]["n"] > 0
    with StubService() as service:
        assert "latency_idle_long" not in smoke.measure_service(service.base, [SHORT] * 4, repeat=4)


# --- Chạy thật trên API (không Docker) ------------------------------------------------------------


@pytest.fixture(scope="module")
def live_base():
    """API thật (mô hình thật) chạy bằng uvicorn trong một luồng, để chạy đúng mã đo qua HTTP."""
    import uvicorn

    from edupulse.api.main import create_app
    from edupulse.text.segment import SegmenterError

    if not (load_settings().bundle_dir / "manifest.json").is_file():
        pytest.skip("chưa có artifacts — chạy `make artifacts`")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    try:
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(), host="127.0.0.1", port=port, log_level="warning", access_log=False
            )
        )
    except SegmenterError as exc:
        pytest.skip(f"không khởi động được bộ tách từ: {exc}")
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not thread.is_alive():
            pytest.skip("API không khởi động được (thiếu JRE hay bundle hỏng)")
        try:
            if smoke.call(base, "GET", "/api/health", timeout=2).status == 200:
                break
        except OSError:
            pass
        time.sleep(0.2)
    else:
        server.should_exit = True
        pytest.fail("API chưa sẵn sàng sau 120 s")
    yield base
    server.should_exit = True
    thread.join(30)


REAL_TEXTS = [
    "[đã gỡ câu UIT-VSFC].",
    "Phòng máy nóng, máy chiếu mờ, quạt kêu to.",
    "Giáo trình cần được cập nhật thêm ví dụ thực tế.",
    "Em không có ý kiến gì thêm về môn học này.",
    "Cô đến lớp đúng giờ và hỗ trợ sinh viên tận tình.",
    "Lịch thi dày đặc, mong nhà trường dàn đều hơn.",
    "Wifi trong giảng đường yếu nên không tra cứu tài liệu được.",
    "Môn học hữu ích và gắn với thực tế công việc.",
    "[đã gỡ câu UIT-VSFC] "
    "thuyết suông [đã gỡ câu UIT-VSFC] "
    "lý thuyết.",
]


@pytest.mark.model
def test_measuring_the_real_api_yields_a_verdict_with_both_latencies(live_base):
    result = smoke.measure_service(live_base, REAL_TEXTS, repeat=len(REAL_TEXTS))
    assert result["limits"]["source"] == "/api/config"
    assert result["limits"]["xai_max_words"] == load_settings().xai_max_words
    idle = result["latency_idle"]
    assert idle["analyze"]["n"] == len(REAL_TEXTS)
    assert idle["explain"]["n"] == len(REAL_TEXTS) - idle["not_explained"] >= 1
    assert idle["not_explained"] >= 1  # câu 40+ từ bị máy chủ từ chối giải thích
    verdict = smoke.verdict_rows(result)
    assert row_starting(verdict, "Nhãn")["ok"] is not None
    assert row_starting(verdict, "Giải thích")["ok"] is not None


@pytest.mark.model
def test_the_real_baseline_answers_the_first_comparison_and_the_memory_is_reported(live_base):
    result = smoke.measure_compare(live_base, REAL_TEXTS[0])
    assert result["ok"] is True, result
    assert result["cold_s"] > 0 and result["warm_s"] > 0 and result["predict_ms"] >= 0
    assert result["rss_before_mb"] > 0 and result["rss_after_mb"] > 0
    assert result["rss_delta_mb"] == round(result["rss_after_mb"] - result["rss_before_mb"], 1)


@pytest.mark.model
@pytest.mark.slow
def test_the_five_megabyte_upload_check_runs_the_whole_batch_path(live_base):
    result, batch_id = smoke.check_upload(live_base, REAL_TEXTS, timeout=600)
    assert result["problems"] == [] and result["ok"] is True
    assert result["bytes"] == 5 * 1024 * 1024 and result["rows"] == MAX_ROWS
    assert result["flagged"] >= math.ceil(0.10 * MAX_ROWS)
    assert isinstance(batch_id, str) and result["run_s"] > 0
    # Batch đã được đóng: không còn giữ dữ liệu của người thử.
    assert smoke.call(live_base, "GET", f"/api/batches/{batch_id}").status == 404


def test_a_dropped_connection_becomes_a_problem_not_a_crash():
    result, batch_id = smoke.check_upload("http://127.0.0.1:9", REAL_TEXTS, timeout=5)
    assert result["ok"] is False and batch_id is None
    assert result["problems"] and "mất kết nối" in result["problems"][0]


@pytest.mark.model
@pytest.mark.slow
def test_remote_mode_end_to_end_writes_a_coherent_report(live_base, tmp_path: Path):
    args = SimpleNamespace(url=live_base, repeat=len(REAL_TEXTS))
    run = smoke.run_remote(args, REAL_TEXTS)
    report = report_of(run)
    markdown = smoke.render_markdown(report)
    assert "**Nhãn**, p95 của /api/analyze" in markdown and "**Giải thích**, p95" in markdown
    assert "## Upload 5MB" in markdown and "## So sánh với mô hình cơ sở" in markdown
    assert run["upload"]["ok"] is True
    assert run["compare"]["ok"] is True and run["checks"]["compare_with_baseline_works"] is True
    assert len(run["phases"]) >= 4 and all(p["noisy_with"] is None for p in run["phases"])
    # Bản web chưa chắc đã build ở máy chạy test, nên chỉ kiểm các hành vi của API.
    assert run["checks"]["api_unknown_is_json_404"] and run["checks"]["wrong_method_is_405"]
    (tmp_path / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
