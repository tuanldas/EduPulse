"""Chốt chặn câu gốc trước khi đẩy: tìm trong lịch sử, dạng tách từ, \\uXXXX, commit message."""

import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml import check_raw_sentences as check

GIT_IDENTITY = ["-c", "user.email=dev@example.org", "-c", "user.name=dev"]
# Câu dựng riêng cho test, không thuộc bộ dữ liệu.
SENTENCE = "con mèo màu cam ngồi trên nóc tủ lạnh suốt buổi chiều thứ bảy"


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *GIT_IDENTITY, *args], cwd=root, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def commit_all(root: Path, message: str) -> str:
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def raw(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    pq.write_table(pa.table({"sentence": [SENTENCE]}), directory / "train.parquet")
    return directory


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    (root / "README.md").write_text("EduPulse\n", "utf-8")
    commit_all(root, "khởi tạo")
    return root


def run(repo: Path, raw: Path, *args: str) -> int:
    return check.main(["--repo", str(repo), "--raw-dir", str(raw), *args])


def test_clean_history_passes(repo, raw, capsys):
    assert run(repo, raw) == 0
    assert "sạch" in capsys.readouterr().out


def test_sentence_removed_from_head_is_still_found_in_history(repo, raw, capsys):
    sample = repo / "test_sample.py"
    sample.write_text(f'TEXT = "{SENTENCE.capitalize()} ."\n', "utf-8")
    commit_all(repo, "thêm câu mẫu")
    sample.write_text('TEXT = "câu viết lại"\n', "utf-8")
    commit_all(repo, "viết lại câu mẫu")

    assert run(repo, raw, "--head") == 0
    capsys.readouterr()
    assert run(repo, raw) == 1
    assert "câu gốc: test_sample.py (1 dòng)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "line",
    [
        SENTENCE.replace("tủ lạnh", "tủ_lạnh").replace("buổi chiều", "buổi_chiều"),
        json.dumps({"text": SENTENCE}),
    ],
    ids=["tach-tu", "unicode-escape"],
)
def test_segmented_and_escaped_forms_are_found(repo, raw, capsys, line):
    (repo / "fixture.json").write_text(line + "\n", "utf-8")
    commit_all(repo, "thêm fixture")
    assert run(repo, raw, "--head") == 1
    assert "fixture.json" in capsys.readouterr().out


def test_commit_message_is_checked(repo, raw, capsys):
    git(repo, "commit", "-q", "--allow-empty", "-m", f"sửa lỗi với câu {SENTENCE}")
    assert run(repo, raw) == 1
    assert "commit message: 1 dòng" in capsys.readouterr().out


def test_secret_is_reported_by_kind_without_its_value(repo, raw, capsys):
    # Dựng lúc chạy để chính tệp test này không chứa chuỗi giống token.
    token = "hf_" + "A1b2C3d4" * 5
    (repo / "settings.py").write_text(f'TOKEN = "{token}"\n', "utf-8")
    commit_all(repo, "cấu hình")
    assert run(repo, raw, "--head") == 1
    out = capsys.readouterr().out
    assert "bí mật: settings.py (token Hugging Face)" in out
    assert token not in out


def test_a_range_only_scans_commits_not_yet_pushed(repo, raw):
    (repo / "old.txt").write_text(SENTENCE + "\n", "utf-8")
    pushed = commit_all(repo, "đã có trên remote")
    (repo / "new.txt").write_text("một câu hoàn toàn khác\n", "utf-8")
    head = commit_all(repo, "mới")
    assert run(repo, raw, head, "--not", pushed) == 0
    assert run(repo, raw, head) == 1


def test_missing_raw_data_exits_2(repo, tmp_path):
    assert check.main(["--repo", str(repo), "--raw-dir", str(tmp_path / "trong")]) == 2
