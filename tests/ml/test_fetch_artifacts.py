"""scripts/fetch-artifacts.sh: tải bundle theo revision ghim, kiểm sha256, không để lại bản dở.

"Hub" là một thư mục dựng đúng đường dẫn `<repo>/resolve/<revision>/` và script gọi nó qua file://
(HF_ENDPOINT), nên chạy được khi không có mạng. Manifest do `write_json` ghi (mỗi tệp một dòng) đúng
như bản thật, nên test cũng khoá luôn hợp đồng định dạng giữa bên ghi và bên đọc của manifest.
"""

import json
import shutil

import pytest
from bundle_factory import (
    BUNDLE_VERSION,
    HUB_REPO,
    REVISION,
    SCRIPT,
    fetch_with_script,
    publish_to_hub,
    write_bundle,
)

from edupulse.config import REPO_ROOT
from edupulse.engine.bundle import load_bundle
from ml.common import write_json

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("curl"))
    or not (shutil.which("sha256sum") or shutil.which("shasum")),
    reason="cần bash, curl và sha256sum (hoặc shasum)",
)


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    root = tmp_path_factory.mktemp("pristine") / "bundle"
    return write_bundle(root, stage="release", git_commit="6c19cdb3cf")


@pytest.fixture()
def hub(pristine, tmp_path):
    """Hub giả chứa bundle cộng một README mà manifest không liệt kê."""
    upload = tmp_path / "upload"
    shutil.copytree(pristine, upload)
    (upload / "README.md").write_text("# README không nằm trong manifest\n", encoding="utf-8")
    return publish_to_hub(upload, tmp_path / "hub")


def revision_dir(hub):
    return hub / HUB_REPO / "resolve" / REVISION


def rewrite_manifest(hub, mutate):
    path = revision_dir(hub) / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    mutate(manifest)
    write_json(path, manifest)


def files_under(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_the_make_target_runs_this_script():
    assert SCRIPT.is_file()
    assert "bash scripts/fetch-artifacts.sh" in (REPO_ROOT / "Makefile").read_text("utf-8")


def test_fetch_reproduces_the_bundle_and_skips_files_the_manifest_does_not_list(
    pristine, hub, tmp_path
):
    target = tmp_path / "out" / BUNDLE_VERSION
    result = fetch_with_script(hub, target)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((pristine / "manifest.json").read_text("utf-8"))
    assert files_under(target) == {"manifest.json", *manifest["files"]}  # không có README.md
    for name in files_under(target):
        assert (target / name).read_bytes() == (pristine / name).read_bytes()
    assert "khớp sha256" in result.stderr


def test_the_runtime_loader_accepts_what_was_fetched(hub, tmp_path):
    target = tmp_path / "out" / BUNDLE_VERSION
    assert fetch_with_script(hub, target).returncode == 0
    bundle = load_bundle(target)  # kiểm sha256 từng tệp và schema, như API lúc khởi động
    assert bundle.manifest.bundle_version == BUNDLE_VERSION


def test_a_corrupted_file_is_rejected_and_nothing_is_left_behind(hub, tmp_path):
    (revision_dir(hub) / "tokenizer.json").write_text('{"x": 1}', encoding="utf-8")
    target = tmp_path / "out" / BUNDLE_VERSION
    result = fetch_with_script(hub, target)
    assert result.returncode != 0
    assert "sha256 không khớp" in result.stderr
    assert not target.exists()
    assert not list(target.parent.glob(".fetch-*"))  # thư mục tạm đã được dọn


def test_a_missing_file_listed_in_the_manifest_fails_the_download(hub, tmp_path):
    (revision_dir(hub) / "tokenizer.json").unlink()
    result = fetch_with_script(hub, tmp_path / "out" / BUNDLE_VERSION)
    assert result.returncode != 0
    assert "không tải được tokenizer.json" in result.stderr


def test_a_revision_missing_on_the_hub_reports_the_failed_download(hub, tmp_path):
    result = fetch_with_script(hub, tmp_path / "o", {"EDUPULSE_HF_REVISION": "f" * 40})
    assert result.returncode != 0
    assert "không tải được manifest.json" in result.stderr


def test_replacing_a_bundle_needs_force(hub, tmp_path):
    target = tmp_path / "out" / BUNDLE_VERSION
    assert fetch_with_script(hub, target).returncode == 0
    again = fetch_with_script(hub, target)
    assert again.returncode != 0
    assert "FORCE=1" in again.stderr
    assert fetch_with_script(hub, target, {"FORCE": "1"}).returncode == 0


def test_a_failed_replacement_keeps_the_existing_bundle(hub, tmp_path):
    target = tmp_path / "out" / BUNDLE_VERSION
    assert fetch_with_script(hub, target).returncode == 0
    before = (target / "manifest.json").read_bytes()
    (revision_dir(hub) / "tokenizer.json").write_text('{"x": 1}', encoding="utf-8")
    assert fetch_with_script(hub, target, {"FORCE": "1"}).returncode != 0
    assert (target / "manifest.json").read_bytes() == before
    assert files_under(target) >= {"model.int8.onnx", "tokenizer.json"}


def test_force_never_replaces_a_folder_that_is_not_a_bundle(hub, tmp_path):
    target = tmp_path / "ghi-chu"
    target.mkdir()
    (target / "quan-trong.txt").write_text("đừng xoá", encoding="utf-8")
    result = fetch_with_script(hub, target, {"FORCE": "1"})
    assert result.returncode != 0
    assert "không phải bundle" in result.stderr
    assert (target / "quan-trong.txt").read_text("utf-8") == "đừng xoá"


@pytest.mark.parametrize(
    ("revision", "fragment"),
    [
        ("main", "40 ký tự"),
        ("0123456789ABCDEF0123456789abcdef01234567", "40 ký tự"),
        ("0123456789abcdef", "40 ký tự"),
        ("", "EDUPULSE_HF_REVISION"),
    ],
)
def test_the_revision_must_be_a_full_commit_sha(hub, tmp_path, revision, fragment):
    result = fetch_with_script(hub, tmp_path / "o", {"EDUPULSE_HF_REVISION": revision})
    assert result.returncode != 0
    assert fragment in result.stderr
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize(
    ("repo", "fragment"),
    [
        ("your-user/edupulse-phobert-mtl", "giá trị mẫu"),
        ("", "EDUPULSE_HF_REPO"),
        ("khong-co-chu", "chủ/tên"),
        ("a/b/c", "chủ/tên"),
    ],
)
def test_the_repo_must_be_a_real_owner_and_name(hub, tmp_path, repo, fragment):
    result = fetch_with_script(hub, tmp_path / "o", {"EDUPULSE_HF_REPO": repo})
    assert result.returncode != 0
    assert fragment in result.stderr


def test_a_manifest_for_another_version_is_refused(hub, tmp_path):
    result = fetch_with_script(hub, tmp_path / "o", {"BUNDLE_VERSION": "v9.9"})
    assert result.returncode != 0
    assert "bundle_version" in result.stderr
    assert not (tmp_path / "o").exists()


def test_unsafe_names_in_the_manifest_are_refused_before_any_download(hub, tmp_path):
    rewrite_manifest(hub, lambda m: m["files"].update({"../thoat.txt": "a" * 64}))
    target = tmp_path / "out" / BUNDLE_VERSION
    result = fetch_with_script(hub, target)
    assert result.returncode != 0
    assert "không an toàn" in result.stderr
    assert not (tmp_path / "out" / "thoat.txt").exists()


def test_a_manifest_that_is_not_one_entry_per_line_is_refused(hub, tmp_path):
    path = revision_dir(hub) / "manifest.json"
    path.write_text(json.dumps(json.loads(path.read_text("utf-8"))), encoding="utf-8")  # 1 dòng
    result = fetch_with_script(hub, tmp_path / "o")
    assert result.returncode != 0
    assert "không đọc được danh sách tệp" in result.stderr


def test_settings_come_from_the_env_file_and_the_environment_wins(hub, tmp_path):
    target = tmp_path / "out" / BUNDLE_VERSION
    env_file = tmp_path / "fetch.env"
    # CRLF và khoảng trắng cuối dòng (file sửa trên Windows) không được lọt vào giá trị
    env_file.write_bytes(
        f"EDUPULSE_HF_REVISION={REVISION}  \r\nEDUPULSE_ARTIFACTS_DIR={target}\r\n".encode()
    )
    silent = {
        "EDUPULSE_HF_REVISION": "",
        "EDUPULSE_ARTIFACTS_DIR": "",
        "EDUPULSE_ENV_FILE": str(env_file),
    }
    result = fetch_with_script(hub, target, silent)
    assert result.returncode == 0, result.stderr
    assert (target / "manifest.json").is_file()

    other = tmp_path / "other"
    overridden = {**silent, "EDUPULSE_HF_REVISION": "f" * 40, "EDUPULSE_ARTIFACTS_DIR": str(other)}
    result = fetch_with_script(hub, other, overridden)
    assert result.returncode != 0
    assert "không tải được manifest.json" in result.stderr  # revision của môi trường thắng file
    assert not other.exists()


def test_the_target_defaults_to_home_artifacts_version(hub, tmp_path):
    home = tmp_path / "home"
    result = fetch_with_script(
        hub, tmp_path / "unused", {"EDUPULSE_ARTIFACTS_DIR": "", "EDUPULSE_HOME": str(home)}
    )
    assert result.returncode == 0, result.stderr
    assert (home / "artifacts" / BUNDLE_VERSION / "manifest.json").is_file()
