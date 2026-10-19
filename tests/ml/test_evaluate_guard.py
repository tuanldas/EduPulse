"""Chốt chặn của lần chấm test cuối: thiếu freeze, freeze chưa commit, cây bẩn, đã chấm rồi."""

import subprocess
from pathlib import Path

import pytest

from ml import evaluate, freeze, preds
from ml.common import read_json, write_json

GIT_IDENTITY = ["-c", "user.email=dev@example.org", "-c", "user.name=dev"]


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
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    git(root, "init", "-q", "-b", "main")
    (root / "README.md").write_text("mã đang kiểm\n", encoding="utf-8")
    commit_all(root, "init")
    monkeypatch.setattr(freeze, "REPO_ROOT", root)
    monkeypatch.setattr(freeze, "FREEZE_PATH", root / "results" / "freeze.json")
    return root


def freeze_document(commit: str) -> dict:
    return {
        "git": {"commit": commit, "dirty": []},
        "deploy": {"run": "mtl-s42"},
        "files": {},
        "data": {"test_sha256": "x", "normalizer_version": "1"},
        "reruns": [],
    }


def committed_freeze(repo: Path) -> dict:
    """Ghi freeze ghi nhận HEAD hiện tại, rồi commit nó (đúng trình tự khi chạy thật)."""
    document = freeze_document(git(repo, "rev-parse", "HEAD"))
    write_json(freeze.FREEZE_PATH, document)
    commit_all(repo, "chore: freeze")
    return document


def allowed(repo: Path, preds_dir: Path):
    return freeze.assert_final_allowed(preds_dir=preds_dir, check_files=False)


def test_refuses_without_a_freeze_file(repo, tmp_path):
    with pytest.raises(freeze.GuardError, match="chưa có .*freeze.json"):
        allowed(repo, tmp_path / "test_preds")


def test_refuses_a_freeze_that_was_never_committed(repo, tmp_path):
    write_json(freeze.FREEZE_PATH, freeze_document(git(repo, "rev-parse", "HEAD")))
    with pytest.raises(freeze.GuardError, match="chưa được commit"):
        allowed(repo, tmp_path / "test_preds")


def test_refuses_when_other_changes_are_uncommitted(repo, tmp_path):
    committed_freeze(repo)
    (repo / "README.md").write_text("đã sửa sau khi đóng băng\n", encoding="utf-8")
    with pytest.raises(freeze.GuardError, match="thay đổi chưa commit"):
        allowed(repo, tmp_path / "test_preds")


def test_refuses_untracked_files_too(repo, tmp_path):
    committed_freeze(repo)
    (repo / "scratch.py").write_text("print(1)\n", encoding="utf-8")
    with pytest.raises(freeze.GuardError, match="thay đổi chưa commit"):
        allowed(repo, tmp_path / "test_preds")


def test_refuses_a_freeze_edited_after_it_was_committed(repo, tmp_path):
    document = committed_freeze(repo)
    document["deploy"]["run"] = "mtl-s43"
    write_json(freeze.FREEZE_PATH, document)
    with pytest.raises(freeze.GuardError, match="thay đổi chưa commit"):
        allowed(repo, tmp_path / "test_preds")


def test_allows_a_clean_tree_with_the_freeze_committed_before_the_run(repo, tmp_path):
    document = committed_freeze(repo)
    assert allowed(repo, tmp_path / "test_preds") == document


def test_allows_later_commits_on_top_of_the_freeze_commit(repo, tmp_path):
    committed_freeze(repo)
    (repo / "README.md").write_text("báo cáo thêm\n", encoding="utf-8")
    commit_all(repo, "docs: more")
    assert allowed(repo, tmp_path / "test_preds")["deploy"]["run"] == "mtl-s42"


def test_refuses_when_the_frozen_commit_is_not_an_ancestor_of_head(repo, tmp_path):
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "side.txt").write_text("nhánh khác\n", encoding="utf-8")
    side = commit_all(repo, "side")
    git(repo, "checkout", "-q", "main")
    write_json(freeze.FREEZE_PATH, freeze_document(side))
    commit_all(repo, "chore: freeze from another branch")
    with pytest.raises(freeze.GuardError, match="không phải tổ tiên của HEAD"):
        allowed(repo, tmp_path / "test_preds")


def test_refuses_when_the_test_was_already_evaluated(repo, tmp_path):
    committed_freeze(repo)
    done = tmp_path / "test_preds"
    done.mkdir()
    with pytest.raises(freeze.GuardError, match="đã tồn tại.*--record-rerun"):
        allowed(repo, done)


def test_a_rerun_needs_a_recorded_reason_and_a_new_committed_freeze(repo, tmp_path):
    committed_freeze(repo)
    done = tmp_path / "test_preds"
    done.mkdir()
    (done / "marker").write_text("dự đoán lần đầu", encoding="utf-8")

    archive = freeze.record_rerun("sửa lỗi ghép nhãn khi đọc dự đoán", done)
    assert archive.name == "test_preds.superseded-1" and (archive / "marker").is_file()
    assert not done.exists()
    recorded = read_json(freeze.FREEZE_PATH)["reruns"]
    assert recorded[0]["n"] == 1 and "ghép nhãn" in recorded[0]["reason"]
    with pytest.raises(freeze.GuardError, match="thay đổi chưa commit"):
        allowed(repo, done)  # freeze vừa sửa chưa commit
    commit_all(repo, "chore: record rerun")
    assert allowed(repo, done)["reruns"][0]["n"] == 1


def test_second_rerun_gets_the_next_number_and_keeps_both_reasons(repo, tmp_path):
    committed_freeze(repo)
    done = tmp_path / "test_preds"
    for number, reason in enumerate(("lý do thứ nhất, đủ dài", "lý do thứ hai, đủ dài"), 1):
        done.mkdir()
        assert freeze.record_rerun(reason, done).name == f"test_preds.superseded-{number}"
    reruns = read_json(freeze.FREEZE_PATH)["reruns"]
    assert [r["n"] for r in reruns] == [1, 2]
    assert [r["reason"] for r in reruns] == ["lý do thứ nhất, đủ dài", "lý do thứ hai, đủ dài"]


def test_rerun_reason_must_be_meaningful_and_test_must_have_been_run(repo, tmp_path):
    committed_freeze(repo)
    done = tmp_path / "test_preds"
    done.mkdir()
    with pytest.raises(freeze.GuardError, match="lý do chạy lại"):
        freeze.record_rerun("lỗi", done)
    assert done.exists()  # lý do hỏng thì không cất gì
    with pytest.raises(freeze.GuardError, match="chưa chấm test"):
        freeze.record_rerun("sửa lỗi ghép nhãn khi đọc dự đoán", tmp_path / "chua-co")


def test_final_command_never_opens_the_test_split_when_a_guard_fails(
    repo, tmp_path, monkeypatch, capsys
):
    def explode(*args, **kwargs):
        raise AssertionError("test không được mở khi chốt chặn chưa thoả")

    ctx = preds.EvalContext("test", tmp_path / "test_preds", tmp_path / "results", rehearsal=False)
    monkeypatch.setattr(evaluate, "context", lambda rehearsal=False: ctx)
    monkeypatch.setattr(preds, "load_processed", explode)
    assert evaluate.cmd_final(rehearsal=False, threads=1) == evaluate.EXIT_GUARD
    assert "không được chấm test" in capsys.readouterr().err
    assert not ctx.preds_dir.exists()


def test_freeze_command_refuses_once_the_test_was_evaluated(tmp_path, monkeypatch, capsys):
    done = tmp_path / "test_preds"
    done.mkdir()
    ctx = preds.EvalContext("test", done, tmp_path / "results", rehearsal=False)
    monkeypatch.setattr(evaluate, "context", lambda rehearsal=False: ctx)
    assert evaluate.cmd_freeze() == evaluate.EXIT_GUARD
    assert "không đóng băng lại" in capsys.readouterr().err


def test_record_rerun_command_reports_guard_errors(repo, tmp_path, monkeypatch, capsys):
    ctx = preds.EvalContext("test", tmp_path / "test_preds", tmp_path / "results", rehearsal=False)
    monkeypatch.setattr(evaluate, "context", lambda rehearsal=False: ctx)
    assert evaluate.cmd_record_rerun("lý do đủ dài để hợp lệ") == evaluate.EXIT_GUARD
    assert "chưa có" in capsys.readouterr().err


def test_tables_command_needs_a_finished_evaluation(tmp_path, monkeypatch, capsys):
    ctx = preds.EvalContext("test", tmp_path / "test_preds", tmp_path / "results", rehearsal=False)
    monkeypatch.setattr(evaluate, "context", lambda rehearsal=False: ctx)
    assert evaluate.cmd_tables(rehearsal=False) == evaluate.EXIT_GUARD
    assert "chưa chấm xong" in capsys.readouterr().err


def test_command_line_requires_a_mode(capsys):
    with pytest.raises(SystemExit) as stop:
        evaluate.main([])
    assert stop.value.code == 2
    with pytest.raises(SystemExit):
        evaluate.main(["--split", "test"])  # test cần --final hoặc --tables
    with pytest.raises(SystemExit):
        evaluate.main(["--split", "val"])  # không còn lệnh val
