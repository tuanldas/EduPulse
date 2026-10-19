"""Hàng đợi run PhoBERT: thứ tự, chạy lại được, một khoá MPS cho cả hàng, từ chối khi Docker bật."""

import json
import subprocess
from pathlib import Path

import pytest

from edupulse.text import NORMALIZER_VERSION
from ml import experiment_queue as queue
from ml.common import mps_lock, mps_lock_is_held, write_json

ALL_RUNS = [
    "mtl-s42",
    "mtl-s43",
    "mtl-s44",
    "stl-sentiment-s42",
    "stl-topic-s42",
    "stl-sentiment-s43",
    "stl-topic-s43",
    "stl-sentiment-s44",
    "stl-topic-s44",
]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    return tmp_path


def write_history(home: Path, name: str, *, finished: bool = True, **args) -> None:
    history = {
        "args": {"epochs": 4, **queue.EXPECTED_ARGS, **args},
        "normalizer_version": NORMALIZER_VERSION,
        "finished": finished,
        "best_epoch": 4,
        "total_s": 700.0,
    }
    write_json(home / "models" / "phobert" / name / "history.json", history)


class Recorder:
    """Thay cho tiến trình huấn luyện: ghi lại lời gọi và xem khoá MPS có đang được giữ không."""

    def __init__(self, codes: dict[str, int] | None = None, raises: BaseException | None = None):
        self.calls: list[dict] = []
        self.codes = codes or {}
        self.raises = raises

    def __call__(self, spec, *, epochs, force, allow_docker, log_path):
        self.calls.append(
            {
                "run": spec.run_name,
                "epochs": epochs,
                "force": force,
                "allow_docker": allow_docker,
                "log": log_path.name,
                "lock_held": mps_lock_is_held(),
            }
        )
        if self.raises:
            raise self.raises
        return self.codes.get(spec.run_name, 0)

    @property
    def runs(self) -> list[str]:
        return [call["run"] for call in self.calls]


def run(home, recorder, *, docker=False, **kwargs):
    lines: list[str] = []
    code = queue.run_queue(
        queue.default_queue(),
        runner=recorder,
        docker_check=lambda: docker,
        out=lines.append,
        **kwargs,
    )
    return code, "\n".join(lines)


def test_default_queue_has_nine_runs_with_mtl_first_then_paired_stl():
    assert [spec.run_name for spec in queue.default_queue()] == ALL_RUNS


def test_exploration_run_is_kept_apart_from_the_comparison_runs():
    spec = queue.explore_run()
    assert (spec.heads, spec.seed, spec.epochs) == ("mtl", 42, 6)
    assert spec.run_name not in ALL_RUNS and not spec.run_name.startswith("mtl-s")


def test_missing_runs_are_all_pending_and_run_in_queue_order(home):
    recorder = Recorder()
    code, text = run(home, recorder)
    assert code == 0 and recorder.runs == ALL_RUNS
    assert "Còn 9 run" in text


def test_finished_run_with_the_same_settings_is_skipped(home):
    write_history(home, "mtl-s42")
    recorder = Recorder()
    code, text = run(home, recorder)
    assert code == 0 and recorder.runs == ALL_RUNS[1:]
    assert "Còn 8 run" in text


def test_everything_finished_means_nothing_to_do(home):
    for name in ALL_RUNS:
        write_history(home, name)
    recorder = Recorder()
    code, text = run(home, recorder)
    assert code == 0 and not recorder.calls
    assert "Không còn run nào" in text


def test_unfinished_run_starts_again_without_force(home):
    write_history(home, "mtl-s42", finished=False)
    recorder = Recorder()
    run(home, recorder)
    assert recorder.calls[0]["run"] == "mtl-s42" and recorder.calls[0]["force"] is False


def test_finished_run_trained_with_other_epochs_blocks_the_whole_queue(home):
    write_history(home, "mtl-s42", epochs=6)
    recorder = Recorder()
    code, text = run(home, recorder)
    assert code == queue.EXIT_USAGE and not recorder.calls
    assert "epochs: đã chạy với 6" in text and "--retrain-mismatched" in text


def test_retrain_mismatched_reruns_that_run_with_force(home):
    write_history(home, "mtl-s42", epochs=6)
    recorder = Recorder()
    code, _ = run(home, recorder, retrain_mismatched=True)
    assert code == 0
    forced = {call["run"]: call["force"] for call in recorder.calls}
    assert forced["mtl-s42"] is True and forced["mtl-s43"] is False


@pytest.mark.parametrize(
    ("key", "value"), [("lr", 5e-5), ("batch_size", 16), ("limit", 64), ("max_len", 64)]
)
def test_any_changed_training_setting_counts_as_a_different_configuration(home, key, value):
    write_history(home, "mtl-s42", **{key: value})
    recorder = Recorder()
    code, text = run(home, recorder)
    assert code == queue.EXIT_USAGE and key in text


def test_run_made_by_another_normalizer_counts_as_a_different_configuration(home):
    write_history(home, "mtl-s42")
    path = home / "models" / "phobert" / "mtl-s42" / "history.json"
    history = json.loads(path.read_text("utf-8"))
    history["normalizer_version"] = "0"
    write_json(path, history)
    code, text = run(home, Recorder())
    assert code == queue.EXIT_USAGE and "normalizer_version" in text


def test_run_from_an_equivalent_normalizer_is_not_a_different_configuration(home, monkeypatch):
    write_history(home, "mtl-s42")
    path = home / "models" / "phobert" / "mtl-s42" / "history.json"
    history = json.loads(path.read_text("utf-8"))
    history["normalizer_version"] = "0"
    write_json(path, history)
    monkeypatch.setattr(queue, "normalizer_compatible", lambda found: found == "0")
    assert queue.mismatches(history, epochs=4) == []


def test_dry_run_prints_the_plan_but_runs_and_locks_nothing(home):
    recorder = Recorder()
    code, text = run(home, recorder, dry_run=True, docker=True)
    assert code == 0 and not recorder.calls
    assert "Docker Desktop: ĐANG CHẠY" in text and "ước tính" in text
    assert not mps_lock_is_held()


def test_estimate_scales_with_the_number_of_epochs():
    specs = queue.default_queue()[:2]
    assert queue.estimate_minutes(specs, 4) == pytest.approx(2 * 4 * queue.MINUTES_PER_EPOCH)
    assert queue.estimate_minutes(specs, 6) == pytest.approx(2 * 6 * queue.MINUTES_PER_EPOCH)
    explore = [queue.explore_run()]
    assert queue.estimate_minutes(explore, 4) == pytest.approx(6 * queue.MINUTES_PER_EPOCH)


def test_machine_lock_is_held_across_every_run_and_released_at_the_end(home):
    recorder = Recorder()
    run(home, recorder)
    assert recorder.calls and all(call["lock_held"] for call in recorder.calls)
    assert not mps_lock_is_held()


def test_docker_running_refuses_the_queue_before_taking_the_lock(home):
    recorder = Recorder()
    code, text = run(home, recorder, docker=True)
    assert code == queue.EXIT_DOCKER and not recorder.calls
    assert "Tắt Docker Desktop" in text and not mps_lock_is_held()


def test_allow_docker_goes_through_and_is_passed_to_each_run(home):
    recorder = Recorder()
    code, _ = run(home, recorder, docker=True, allow_docker=True)
    assert code == 0 and all(call["allow_docker"] for call in recorder.calls)


def test_lock_held_by_another_job_refuses_the_queue(home):
    recorder = Recorder()
    with mps_lock(note="job khác"):
        code, text = run(home, recorder)
    assert code == queue.EXIT_LOCK and not recorder.calls
    assert "job khác" in text


def test_a_failed_run_does_not_stop_the_others_but_the_queue_reports_failure(home):
    recorder = Recorder(codes={"mtl-s43": 1})
    code, text = run(home, recorder)
    assert code == queue.EXIT_FAILED and recorder.runs == ALL_RUNS
    assert "1 run lỗi: mtl-s43" in text
    status = json.loads((home / "outputs" / "logs" / "experiments.status.json").read_text("utf-8"))
    assert status["failed"] == ["mtl-s43"] and status["finished"] is True


@pytest.mark.parametrize("environment_code", [queue.EXIT_DOCKER, queue.EXIT_LOCK])
def test_docker_or_lock_failure_of_a_child_stops_the_queue_at_once(home, environment_code):
    recorder = Recorder(codes={"mtl-s43": environment_code})
    code, _ = run(home, recorder)
    assert code == environment_code and recorder.runs == ["mtl-s42", "mtl-s43"]
    assert not mps_lock_is_held()


def test_status_file_lists_each_run_with_exit_code_and_log_name(home):
    recorder = Recorder()
    run(home, recorder)
    status = json.loads((home / "outputs" / "logs" / "experiments.status.json").read_text("utf-8"))
    assert [r["run"] for r in status["runs"]] == ALL_RUNS
    assert all(r["exit_code"] == 0 for r in status["runs"])
    assert recorder.calls[0]["log"] == "mtl-s42.log"


def test_interrupt_stops_the_queue_and_releases_the_lock(home):
    recorder = Recorder(raises=KeyboardInterrupt())
    code, text = run(home, recorder)
    assert code == queue.EXIT_INTERRUPTED and len(recorder.calls) == 1
    assert "khoá MPS đã nhả" in text and not mps_lock_is_held()


def test_main_filters_runs_and_rejects_unknown_names(home, capsys):
    assert queue.main(["--only", "mtl-s43,nope", "--dry-run"]) == queue.EXIT_USAGE
    assert "nope" in capsys.readouterr().err
    assert queue.main(["--only", "mtl-s43,stl-topic-s42", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "mtl-s43" in printed and "stl-topic-s42" in printed and "mtl-s44" not in printed


def test_main_rejects_non_positive_epochs(capsys):
    assert queue.main(["--epochs", "0"]) == queue.EXIT_USAGE


def test_main_explore_adds_the_exploration_run_first(home, capsys):
    assert queue.main(["--explore", "--dry-run"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("  ")]
    assert lines[0].split()[0] == "explore-mtl-e6-s42" and len(lines) == 10


class FakeChild:
    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0


def captured_command(monkeypatch, home, spec, **kwargs) -> list[str]:
    started: list[FakeChild] = []

    def fake_popen(command, **popen_kwargs):
        child = FakeChild(command, **popen_kwargs)
        started.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    log = home / "outputs" / "logs" / "x.log"
    options = {"epochs": 4, "force": False, "allow_docker": False, "log_path": log} | kwargs
    assert queue.run_training(spec, **options) == 0
    return started[0].command


def test_child_command_uses_the_shared_lock_and_the_queue_epochs(monkeypatch, home):
    command = captured_command(monkeypatch, home, queue.RunSpec("sentiment", 43))
    assert command[1:3] == ["-m", "ml.train_phobert"]
    pairs = dict(zip(command[3::2], command[4::2], strict=False))
    assert pairs["--heads"] == "sentiment" and pairs["--seed"] == "43" and pairs["--epochs"] == "4"
    assert "--lock-held" in command and "--force" not in command and "--out" not in command


def test_child_command_forces_and_passes_docker_flag_when_asked(monkeypatch, home):
    command = captured_command(
        monkeypatch, home, queue.RunSpec("mtl", 42), force=True, allow_docker=True
    )
    assert "--force" in command and "--allow-docker" in command


def test_exploration_child_writes_to_its_own_directory_with_six_epochs(monkeypatch, home):
    command = captured_command(monkeypatch, home, queue.explore_run())
    assert command[command.index("--epochs") + 1] == "6"
    assert command[command.index("--out") + 1] == str(
        home / "models" / "phobert" / "explore-mtl-e6-s42"
    )


def test_child_output_is_appended_to_the_run_log(monkeypatch, home):
    captured_command(monkeypatch, home, queue.RunSpec("mtl", 43))
    text = (home / "outputs" / "logs" / "x.log").read_text("utf-8")
    assert "ml.train_phobert" in text and text.startswith("\n=== ")


def test_expected_arguments_match_the_training_defaults():
    pytest.importorskip("torch")
    from ml import train_phobert

    defaults = vars(train_phobert.build_parser().parse_args([]))
    for key, value in queue.EXPECTED_ARGS.items():
        assert defaults[key] == value, key
    assert defaults["epochs"] == queue.DEFAULT_EPOCHS


def test_wrapper_script_is_executable_bash_that_loads_dotenv():
    script = Path(queue.REPO_ROOT) / "ml" / "run-experiments.sh"
    assert script.is_file() and script.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text("utf-8")
    assert "ml.experiment_queue" in text and "--env-file" in text
