"""CLI đo hiệu năng: một dòng JSON với đủ khoá mà `scripts/smoke_test.py` đọc."""

import json
from pathlib import Path

import pytest

from edupulse.config import AUTO_XAI_TOP, DEFAULT_BUDGET
from edupulse.engine.bench import main, measure
from edupulse.engine.service import Analyzer

pytestmark = pytest.mark.model

GOLDEN_SENTENCES = Path(__file__).resolve().parents[1] / "fixtures" / "golden_sentences.json"
STEPS = {"prepare", "predict", "flags", "explain"}


@pytest.fixture(scope="module")
def sentences() -> list[str]:
    items = json.loads(GOLDEN_SENTENCES.read_text(encoding="utf-8"))["items"]
    return [item["text"] for item in items if "no_diacritics" not in item["covers"]]


def test_measure_reports_each_step_and_a_total_that_adds_up(analyzer: Analyzer, sentences):
    result = measure(analyzer, sentences, xai_rows=3, budget_pct=20)
    assert set(result["steps_s"]) == STEPS
    assert result["total_s"] == pytest.approx(sum(result["steps_s"].values()), abs=0.005)
    assert result["n"] == len(sentences)
    assert result["flagged"] == -(-20 * len(sentences) // 100)  # ⌈20% · n⌉
    assert 0 < result["xai_rows"] <= 3
    assert result["xai_forwards"] > result["xai_rows"]


def test_measure_skips_sentences_with_no_content_instead_of_failing(analyzer: Analyzer, sentences):
    result = measure(analyzer, [*sentences, "0912345678"], xai_rows=2, budget_pct=10)
    assert result["skipped_no_content"] == 1


def test_cli_prints_one_json_line_with_the_keys_the_smoke_test_reads(
    analyzer: Analyzer, sentences, tmp_path, capsys
):
    path = tmp_path / "cau.txt"
    path.write_text("\n".join(sentences) + "\n", encoding="utf-8")

    assert main(["--input", str(path), "--n", str(len(sentences)), "--xai", "2", "--json"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    result = json.loads(out[0])
    for key in ("total_s", "steps_s", "rss_mb", "cgroup_memory_peak_mb", "bundle_version"):
        assert key in result
    assert result["budget_pct"] == DEFAULT_BUDGET
    assert AUTO_XAI_TOP > 2  # --xai ghi đè mặc định
    assert result["xai_rows"] <= 2


def test_cli_refuses_a_file_with_fewer_sentences_than_requested(tmp_path, capsys):
    path = tmp_path / "ngan.txt"
    path.write_text("một câu thôi\n", encoding="utf-8")
    assert main(["--input", str(path), "--n", "5", "--json"]) == 2
    assert "chỉ có 1" in capsys.readouterr().err
