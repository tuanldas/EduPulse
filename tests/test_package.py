"""Khung package: thứ tự nhãn, cấu hình qua env, phụ thuộc runtime và hợp đồng Makefile."""

import importlib
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from edupulse import config, labels
from edupulse.config import REPO_ROOT, load_settings

ROOT = Path(__file__).resolve().parents[1]

# Target mà các phase sau gọi; Makefile khai báo trước làm hợp đồng.
CONTRACT_TARGETS = set(
    "help setup data eda baseline train export calibrate experiments evaluate report artifacts "
    "demo-data api web gen-api test e2e lint docker-build smoke up fetch-artifacts".split()
)


def test_label_order_is_uit_vsfc_original_ids():
    assert labels.SENTIMENT == ("negative", "neutral", "positive")
    assert labels.TOPIC == ("lecturer", "training_program", "facility", "others")
    assert labels.TASKS == ("sentiment", "topic")
    assert labels.LABELS == {"sentiment": labels.SENTIMENT, "topic": labels.TOPIC}


def test_vietnamese_names_cover_each_label_once():
    for task, names in labels.LABELS.items():
        display = labels.LABELS_VI[task]
        assert set(display) == set(names)
        assert len(set(display.values())) == len(names)


def test_default_budget_is_one_of_the_budgets():
    assert config.DEFAULT_BUDGET in config.BUDGETS


def test_ml_package_is_importable_from_repo_root():
    assert importlib.import_module("ml").__doc__


def test_defaults_without_edupulse_home():
    s = load_settings({})
    assert s.home == REPO_ROOT
    assert s.data_dir == REPO_ROOT / "data"
    assert s.models_dir == REPO_ROOT / "models"
    assert s.outputs_dir == REPO_ROOT / "outputs"
    assert s.demo_dir == REPO_ROOT / "demo-data"
    assert s.mps_lock_path == REPO_ROOT / ".mps.lock"
    assert s.artifacts_dir("v1.0") == REPO_ROOT / "artifacts" / "v1.0"
    assert s.bundle_dir == REPO_ROOT / "artifacts" / "v1.0"
    assert (s.ort_threads, s.ner, s.cors_origins, s.source_url) == (2, False, (), "")
    assert s.job_ttl_min == 30


def test_blank_values_count_as_unset():
    blank = {
        "EDUPULSE_HOME": "  ",
        "EDUPULSE_ORT_THREADS": "",
        "EDUPULSE_NER": "",
        "EDUPULSE_SOURCE_URL": "",
        "BUNDLE_VERSION": "",
    }
    assert load_settings(blank) == load_settings({})


def test_paths_follow_edupulse_home(tmp_path):
    home = tmp_path.resolve()
    s = load_settings({"EDUPULSE_HOME": str(tmp_path)})
    assert s.home == home
    assert s.data_dir == home / "data"
    assert s.models_dir == home / "models"
    assert s.outputs_dir == home / "outputs"
    assert s.demo_dir == home / "demo-data"
    assert s.mps_lock_path == home / ".mps.lock"
    assert s.artifacts_dir("v0-smoke") == home / "artifacts" / "v0-smoke"
    assert s.bundle_dir == home / "artifacts" / "v1.0"
    # Mã nguồn đi kèm repo nên không theo EDUPULSE_HOME.
    assert s.vncorenlp_dir == REPO_ROOT / "vendor" / "vncorenlp"
    assert s.web_dir == REPO_ROOT / "apps" / "web" / "out"


def test_specific_variables_override_home(tmp_path):
    bundle, demo, java_dir = tmp_path / "b", tmp_path / "d", tmp_path / "j"
    s = load_settings(
        {
            "EDUPULSE_HOME": str(tmp_path / "home"),
            "BUNDLE_VERSION": "v2",
            "EDUPULSE_ARTIFACTS_DIR": str(bundle),
            "EDUPULSE_DEMO_DIR": str(demo),
            "EDUPULSE_VNCORENLP_DIR": str(java_dir),
        }
    )
    assert (s.bundle_dir, s.demo_dir, s.vncorenlp_dir) == (bundle, demo, java_dir)
    # Nơi build bundle vẫn theo EDUPULSE_HOME và phiên bản.
    assert s.artifacts_dir() == (tmp_path / "home").resolve() / "artifacts" / "v2"


def test_reads_process_environment_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    monkeypatch.setenv("EDUPULSE_NER", "1")
    s = load_settings()
    assert s.home == tmp_path.resolve()
    assert s.ner is True


def test_parses_typed_values():
    s = load_settings(
        {
            "EDUPULSE_ORT_THREADS": "4",
            "EDUPULSE_NER": "true",
            "EDUPULSE_CORS_ORIGINS": "http://localhost:3000/, https://edupulse.example ,",
            "EDUPULSE_SOURCE_URL": "https://github.com/acme/edupulse",
            "EDUPULSE_JOB_TTL_MIN": "0.05",
        }
    )
    assert s.ort_threads == 4
    assert s.ner is True
    assert s.cors_origins == ("http://localhost:3000", "https://edupulse.example")
    assert s.source_url == "https://github.com/acme/edupulse"
    assert s.job_ttl_s == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("EDUPULSE_ORT_THREADS", "0"),
        ("EDUPULSE_ORT_THREADS", "hai"),
        ("EDUPULSE_NER", "maybe"),
        ("EDUPULSE_JOB_TTL_MIN", "0"),
        ("EDUPULSE_JOB_TTL_MIN", "inf"),
        ("EDUPULSE_CORS_ORIGINS", "localhost:3000"),
        ("EDUPULSE_SOURCE_URL", "ftp://example.com"),
        ("BUNDLE_VERSION", "../etc"),
    ],
)
def test_invalid_values_raise_and_name_the_variable(key, value):
    with pytest.raises(ValueError, match=key):
        load_settings({key: value})


def test_artifacts_dir_rejects_path_traversal():
    with pytest.raises(ValueError, match="version"):
        load_settings({}).artifacts_dir("../outside")


def _group_closure(lock: dict, group: str) -> set[str]:
    """Mọi gói mà một dependency-group kéo vào, tính cả extras (bỏ qua marker nên là cận trên)."""
    packages = {p["name"]: p for p in lock["package"]}
    pending = [d["name"] for d in packages["edupulse"]["dev-dependencies"][group]]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for dep in packages[name].get("dependencies", []):
            pending.append(dep["name"])
            extras = packages[dep["name"]].get("optional-dependencies", {})
            for extra in dep.get("extra", []):
                pending.extend(d["name"] for d in extras.get(extra, []))
    return seen


def test_app_group_excludes_training_stack():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    closure = _group_closure(lock, "app")
    assert {"fastapi", "onnxruntime", "tokenizers"} <= closure, "đọc sai uv.lock"
    assert closure.isdisjoint({"torch", "pandas", "pyarrow"}), sorted(closure)


@pytest.mark.skipif(shutil.which("make") is None, reason="thiếu `make`")
def test_make_help_lists_every_contract_target():
    result = subprocess.run(
        ["make", "-s", "help"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    listed = {line.split()[0] for line in result.stdout.splitlines() if line.strip()}
    assert CONTRACT_TARGETS <= listed, sorted(CONTRACT_TARGETS - listed)


def _run_marked_tests(pytester, markers: list[str]):
    """Chạy trong một phiên pytest con dùng đúng tests/conftest.py, mỗi marker một test rỗng."""
    pytester.makeini("[pytest]\nmarkers =\n    data\n    model\n    java\n    train\n")
    pytester.makeconftest((ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    body = "import pytest\n" + "".join(
        f"\n@pytest.mark.{m}\ndef test_{m}():\n    pass\n" for m in markers
    )
    pytester.makepyfile(body + "\ndef test_plain():\n    pass\n")
    return pytester


def test_data_and_model_tests_skip_with_reason_until_files_exist(pytester, monkeypatch, tmp_path):
    for var in ("EDUPULSE_ARTIFACTS_DIR", "BUNDLE_VERSION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    _run_marked_tests(pytester, ["data", "model"])

    result = pytester.runpytest_inprocess("-rs")
    result.assert_outcomes(passed=1, skipped=2)
    result.stdout.fnmatch_lines(["*chạy `make data`*", "*chạy `make artifacts`*"])

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "MANIFEST.json").write_text("{}", encoding="utf-8")
    bundle = tmp_path / "artifacts" / "v1.0"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    pytester.runpytest_inprocess().assert_outcomes(passed=3)


def test_java_and_train_tests_skip_with_reason_when_missing(pytester, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))  # thư mục rỗng nên không có `java`
    monkeypatch.setitem(sys.modules, "torch", None)  # find_spec("torch") trả None
    _run_marked_tests(pytester, ["java", "train"])

    result = pytester.runpytest_inprocess("-rs")
    result.assert_outcomes(passed=1, skipped=2)
    result.stdout.fnmatch_lines(["*thiếu JRE*", "*thiếu group train*"])
