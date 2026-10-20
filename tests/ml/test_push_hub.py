"""Đẩy bundle lên Hub: dry-run không mạng, chốt chặn, đối chiếu sau khi đẩy, khớp với script tải.

Không test nào gọi Hugging Face: phần mạng được thay bằng `FakeApi` trả lời đúng kiểu Hub (tệp lớn
qua LFS có sha256, tệp nhỏ là git blob), còn dry-run chạy dưới một bộ chặn mọi kết nối socket.
"""

import json
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from bundle_factory import (
    BUNDLE_VERSION,
    fetch_with_script,
    frozen_document,
    publish_to_hub,
    refresh_manifest,
    write_bundle,
)

from edupulse.config import REPO_ROOT
from edupulse.engine.bundle import load_bundle
from ml import build_artifacts as builder
from ml import push_hub
from ml.common import sha256_file, write_json

REPO = "owner/edupulse-phobert-mtl"
SOURCE = "https://github.com/owner/edupulse"
OID = "a" * 40
LONG = "thầy giảng bài rất dễ hiểu và luôn nhiệt tình giúp đỡ sinh viên ."
LONG_SEGMENTED = LONG.replace("giúp đỡ", "giúp_đỡ")
NEEDS_FETCH_TOOLS = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("curl"))
    or not (shutil.which("sha256sum") or shutil.which("shasum")),
    reason="cần bash, curl và sha256sum (hoặc shasum)",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Biến của máy người chạy (repo, token, đường dẫn) không được lọt vào test."""
    variables = ("EDUPULSE_ARTIFACTS_DIR", "EDUPULSE_SOURCE_URL", "EDUPULSE_HF_REPO", "HF_TOKEN")
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    root = tmp_path_factory.mktemp("pristine") / "bundle"
    return write_bundle(root, stage="release", git_commit="6c19cdb3cf")


@pytest.fixture()
def tiny_corpus(monkeypatch):
    """Nguồn câu thu nhỏ cho guard (logic guard vẫn chạy thật); dữ liệu thật dùng marker data."""
    full = pd.DataFrame({"sentence": [LONG], "text_rdr": [LONG_SEGMENTED]})
    empty = pd.DataFrame({"sentence": [], "text_rdr": []})
    frames = {"train": full, "validation": empty, "test": empty}
    monkeypatch.setattr(builder, "load_processed", lambda split, allow_test=False: frames[split])


@dataclass
class Release:
    bundle: Path
    frozen: dict
    out: Path

    def prepare(self, **overrides):
        options = {"repo_id": REPO, "source_url": SOURCE, "frozen": self.frozen} | overrides
        return push_hub.prepare_upload(self.bundle, self.out, **options)


@pytest.fixture()
def release(pristine, tmp_path, tiny_corpus):
    bundle = tmp_path / "bundle"
    shutil.copytree(pristine, bundle)
    return Release(bundle, frozen_document(bundle), tmp_path / "upload")


def edit_json(bundle: Path, name: str, mutate) -> None:
    """Sửa một tệp JSON của bundle rồi ghi lại sha256 trong manifest, để chỉ chốt đang thử nổ."""
    path = bundle / name
    data = json.loads(path.read_text("utf-8"))
    mutate(data)
    write_json(path, data)
    refresh_manifest(bundle)


def forbidden(what):
    def refuse(*args, **kwargs):
        raise AssertionError(f"{what} không được gọi ở đường này")

    return refuse


@pytest.fixture()
def no_network(monkeypatch):
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, forbidden("kết nối mạng"))
    monkeypatch.setattr(socket, "getaddrinfo", forbidden("tra DNS"))
    monkeypatch.setattr(socket, "create_connection", forbidden("kết nối mạng"))


class FakeApi:
    """Phần của HfApi mà push_hub dùng, ghi lại lời gọi và trả lời như Hub."""

    def __init__(self, *, private=False, oid=OID):
        self.private, self.oid = private, oid
        self.calls = []
        self.folder = None
        self.tamper = {}  # tên tệp -> "lfs" | "blob" | "size" | "xet" | "missing"
        self.extra = []  # tệp thừa đã có sẵn trên Hub

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        self.calls.append(("create_repo", repo_id, repo_type, private, exist_ok))

    def model_info(self, repo_id, *, revision=None, files_metadata=False):
        self.calls.append(("model_info", repo_id, revision, files_metadata))
        return SimpleNamespace(private=self.private, siblings=self.siblings() if revision else None)

    def upload_folder(self, *, repo_id, repo_type, folder_path, commit_message, commit_description):
        self.folder = Path(folder_path)
        self.calls.append(
            ("upload_folder", repo_id, repo_type, self.folder, commit_message, commit_description)
        )
        return SimpleNamespace(oid=self.oid)

    def siblings(self):
        def sibling(name, size, lfs, blob_id):
            return SimpleNamespace(rfilename=name, size=size, lfs=lfs, blob_id=blob_id)

        items = [sibling(".gitattributes", 100, None, "9" * 40)]
        for path in sorted(p for p in self.folder.rglob("*") if p.is_file()):
            name = path.relative_to(self.folder).as_posix()
            kind = self.tamper.get(name)
            if kind == "missing":
                continue
            size = path.stat().st_size + (1 if kind == "size" else 0)
            if path.suffix == ".onnx":  # tệp lớn: Hub lưu qua LFS và báo sha256
                digest = "0" * 64 if kind == "lfs" else sha256_file(path)
                items.append(sibling(name, size, SimpleNamespace(sha256=digest), "p" * 40))
            elif kind == "xet":  # tệp lớn lưu kiểu khác: không có thông tin LFS, blob_id là con trỏ
                items.append(sibling(name, size, None, "3" * 40))
            else:  # tệp nhỏ: git blob
                blob = "1" * 40 if kind == "blob" else push_hub.git_blob_sha1(path)
                items.append(sibling(name, size, None, blob))
        items += [sibling(name, 1, None, "2" * 40) for name in self.extra]
        return items


# --- Dựng thư mục upload và các chốt chặn ------------------------------------------------------


def test_the_upload_folder_holds_the_bundle_files_the_manifest_lists_plus_the_readme(release):
    plan = release.prepare()
    manifest = plan.manifest
    assert {item.name for item in plan.files} == {"manifest.json", "README.md", *manifest.files}
    for item in plan.files:
        if item.name in manifest.files:
            assert item.sha256 == manifest.files[item.name]
    uploaded = (plan.directory / "manifest.json").read_bytes()
    assert uploaded == (release.bundle / "manifest.json").read_bytes()
    assert plan.leak_pool == 2  # câu thu nhỏ ở hai dạng


def test_the_readme_comes_from_the_bundle_and_the_given_repo(release):
    text = (release.prepare().directory / "README.md").read_text("utf-8")
    assert text.startswith("---\n")
    assert "license: agpl-3.0" in text
    assert f"EDUPULSE_HF_REPO={REPO}" in text
    assert SOURCE in text


def test_preparing_the_upload_never_changes_the_bundle(release):
    before = {p.name: sha256_file(p) for p in release.bundle.rglob("*") if p.is_file()}
    release.prepare()
    assert {p.name: sha256_file(p) for p in release.bundle.rglob("*") if p.is_file()} == before


def test_a_smoke_bundle_is_refused(tmp_path, tiny_corpus):
    smoke = write_bundle(tmp_path / "smoke")
    with pytest.raises(push_hub.PushError, match="chỉ đẩy bản release"):
        push_hub.prepare_upload(
            smoke, tmp_path / "up", repo_id=REPO, source_url="", frozen=frozen_document(smoke)
        )
    assert not (tmp_path / "up").exists()


@pytest.mark.parametrize("commit", ["6c19cdb3cf+dirty", "unknown"])
def test_a_bundle_built_from_a_dirty_or_unknown_commit_is_refused(release, commit):
    edit_json(release.bundle, "manifest.json", lambda manifest: manifest.update(git_commit=commit))
    with pytest.raises(push_hub.PushError, match="commit không rõ hoặc cây git bẩn"):
        release.prepare()


def test_a_card_and_a_manifest_that_disagree_on_the_version_are_refused(release):
    edit_json(release.bundle, "model_card.json", lambda card: card.update(model_version="khac"))
    with pytest.raises(push_hub.PushError, match="phiên bản khác nhau"):
        release.prepare()


def test_a_tampered_file_is_caught_by_the_sha256_check(release):
    (release.bundle / "tokenizer.json").write_text('{"x": 1}', encoding="utf-8")
    with pytest.raises(builder.BundleError, match=r"tokenizer\.json.*sha256"):
        release.prepare()
    assert not release.out.exists()


def test_a_stray_file_is_refused_but_finder_litter_is_ignored(release):
    (release.bundle / ".DS_Store").write_bytes(b"x")
    release.prepare()
    (release.bundle / "ghi-chu.txt").write_text("x", encoding="utf-8")
    with pytest.raises(push_hub.PushError, match=r"tệp lạ.*ghi-chu\.txt"):
        release.prepare()


def test_files_other_than_the_ones_scored_on_test_are_refused(release):
    frozen = json.loads(json.dumps(release.frozen))
    frozen["files"]["tokenizer.json"] = "0" * 64
    with pytest.raises(builder.BundleError, match="khác bản đã đóng băng"):
        release.prepare(frozen=frozen)
    assert not release.out.exists()


@pytest.mark.parametrize("sentence", [LONG, LONG_SEGMENTED], ids=["original", "word-segmented"])
def test_a_source_sentence_in_the_model_card_blocks_the_upload_and_removes_the_folder(
    release, sentence
):
    # Ghi chú CheckList không vào README, nên chỉ model_card.json chứa câu này.
    edit_json(
        release.bundle, "model_card.json", lambda card: card["checklist"][0].update(note=sentence)
    )
    with pytest.raises(builder.BundleError, match=r"model_card\.json"):
        release.prepare()
    assert not release.out.exists()


def test_a_source_sentence_in_the_readme_blocks_the_upload(release, monkeypatch):
    monkeypatch.setattr(push_hub, "render_readme", lambda *args, **kwargs: f"# x\n{LONG}\n")
    with pytest.raises(builder.BundleError, match=r"README\.md"):
        release.prepare()
    assert not release.out.exists()


def test_a_chosen_folder_with_content_is_never_overwritten(release):
    release.out.mkdir()
    (release.out / "giu-lai.txt").write_text("giữ lại", encoding="utf-8")
    with pytest.raises(push_hub.ConfigError, match="đã có nội dung"):
        release.prepare(owned_directory=False)
    assert (release.out / "giu-lai.txt").read_text("utf-8") == "giữ lại"
    (release.out / "giu-lai.txt").unlink()
    assert release.prepare(owned_directory=False).directory == release.out  # trống thì dùng được


def test_the_default_folder_is_rebuilt_from_scratch(release):
    release.out.mkdir()
    (release.out / "cu.txt").write_text("bản cũ", encoding="utf-8")
    plan = release.prepare()
    assert not (plan.directory / "cu.txt").exists()


# --- Repo đích và liên kết mã nguồn ------------------------------------------------------------


@pytest.mark.parametrize("value", ["owner/name", "Owner-1/edu.pulse_2", "a/b", "x" * 96 + "/y"])
def test_valid_repo_ids_are_accepted(value):
    assert push_hub.parse_repo_id(f"  {value} ") == value


@pytest.mark.parametrize(
    "value",
    ["", "name", "a/b/c", "-a/b", "a/b-", "a/.b", "a/b.", "a/b..c", "a/b--c", "a b/c", "a/", "/b"]
    + ["x" * 97 + "/y"],
)
def test_invalid_repo_ids_are_refused(value):
    with pytest.raises(push_hub.ConfigError):
        push_hub.parse_repo_id(value)


def test_sample_values_from_the_env_example_are_recognised_as_placeholders():
    assert push_hub.is_placeholder("your-user/edupulse-phobert-mtl")
    assert not push_hub.is_placeholder("owner/edupulse-phobert-mtl")
    assert push_hub.clean_source_url("https://github.com/your-user/edupulse") == ""
    assert push_hub.clean_source_url("   ") == ""
    assert push_hub.clean_source_url(f" {SOURCE} ") == SOURCE


def test_the_env_example_declares_the_pin_with_a_sample_that_both_tools_refuse():
    text = (REPO_ROOT / ".env.example").read_text("utf-8")
    values = dict(
        line.split("=", 1) for line in text.splitlines() if "=" in line and line[0] not in "# "
    )
    assert values["EDUPULSE_HF_REVISION"] == ""  # chỉ điền sau khi đẩy
    assert push_hub.is_placeholder(values["EDUPULSE_HF_REPO"])
    push_hub.parse_repo_id(values["EDUPULSE_HF_REPO"])  # đúng dạng chủ/tên
    assert "EDUPULSE_MODEL_URL" in values


def test_git_blob_ids_match_what_git_computes(tmp_path):
    (tmp_path / "empty").write_bytes(b"")
    (tmp_path / "hello").write_bytes(b"hello\n")
    assert push_hub.git_blob_sha1(tmp_path / "empty") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert push_hub.git_blob_sha1(tmp_path / "hello") == "ce013625030ba8dba906f756967f9e9ca394464a"


# --- Dòng lệnh: dry-run, từ chối cấu hình sai ----------------------------------------------------


@pytest.fixture()
def cli(release, tmp_path, monkeypatch):
    """EDUPULSE_HOME tạm có bundle ở artifacts/<phiên bản>; freeze lấy từ bản khớp bundle."""
    home = tmp_path / "home"
    (home / "artifacts").mkdir(parents=True)
    shutil.move(release.bundle, home / "artifacts" / BUNDLE_VERSION)
    monkeypatch.setenv("EDUPULSE_HOME", str(home))
    monkeypatch.setenv("BUNDLE_VERSION", BUNDLE_VERSION)
    monkeypatch.setattr(push_hub.freeze_lib, "load_freeze", lambda: release.frozen)
    return home


def test_dry_run_builds_and_checks_everything_without_touching_the_network(
    cli, no_network, monkeypatch, capsys
):
    monkeypatch.setenv("EDUPULSE_HF_REPO", REPO)
    monkeypatch.setenv("EDUPULSE_SOURCE_URL", SOURCE)
    monkeypatch.setattr(push_hub, "hub_api", forbidden("hub_api"))
    assert push_hub.main(["--dry-run"]) == 0
    captured = capsys.readouterr()
    folder = cli / "outputs" / "hub_upload" / BUNDLE_VERSION
    assert (folder / "README.md").is_file()
    assert (folder / "manifest.json").is_file()
    assert "DRY-RUN xong" in captured.out
    for name in ("model.int8.onnx", "README.md", "manifest.json"):
        assert name in captured.out
    assert "CẢNH BÁO" not in captured.err


def test_dry_run_is_the_default_mode(cli, no_network, monkeypatch, capsys):
    monkeypatch.setattr(push_hub, "hub_api", forbidden("hub_api"))
    assert push_hub.main([]) == 0
    assert "DRY-RUN xong" in capsys.readouterr().out


def test_dry_run_warns_when_the_repo_and_the_source_link_are_not_set(cli, capsys):
    assert push_hub.main(["--dry-run"]) == 0
    warnings = capsys.readouterr().err
    assert "EDUPULSE_HF_REPO chưa đặt" in warnings
    assert "liên kết mã nguồn" in warnings


def test_dry_run_and_upload_cannot_be_combined():
    with pytest.raises(SystemExit) as stop:
        push_hub.main(["--dry-run", "--upload"])
    assert stop.value.code == 2


def test_a_missing_bundle_is_reported_not_crashed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    monkeypatch.setenv("BUNDLE_VERSION", BUNDLE_VERSION)
    monkeypatch.setattr(push_hub.freeze_lib, "load_freeze", lambda: {"files": {}})
    assert push_hub.main(["--dry-run"]) == 1
    assert "không đọc được bundle" in capsys.readouterr().err


def test_upload_needs_a_real_repo_and_stops_before_building_anything(cli, monkeypatch, capsys):
    monkeypatch.setattr(push_hub, "hub_api", forbidden("hub_api"))
    assert push_hub.main(["--upload"]) == 2
    assert "repo đích" in capsys.readouterr().err
    assert push_hub.main(["--upload", "--repo", "your-user/edupulse-phobert-mtl"]) == 2
    assert push_hub.main(["--upload", "--repo", "khong hop le"]) == 2
    assert not (cli / "outputs" / "hub_upload").exists()


def test_upload_without_a_login_is_refused_before_any_request(cli, no_network, monkeypatch, capsys):
    pytest.importorskip("huggingface_hub")
    monkeypatch.setenv("EDUPULSE_HF_REPO", REPO)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    assert push_hub.main(["--upload"]) == 2
    assert "chưa đăng nhập" in capsys.readouterr().err


# --- Đẩy lên Hub (FakeApi) ---------------------------------------------------------------------


def test_upload_creates_a_public_repo_pushes_the_staged_folder_and_verifies_it(release):
    plan = release.prepare()
    api = FakeApi()
    revision, warnings = push_hub.upload(plan, api)
    assert (revision, warnings) == (OID, [])
    assert [call[0] for call in api.calls] == [
        "create_repo",
        "model_info",
        "upload_folder",
        "model_info",
    ]
    assert api.calls[0] == ("create_repo", REPO, "model", False, True)
    assert api.calls[2][1:5] == (REPO, "model", plan.directory, "m: bundle v-test")
    assert api.calls[2][5].startswith("Commit mã nguồn 6c19cdb3cf.")
    assert plan.manifest.files["model.int8.onnx"] in api.calls[2][5]
    assert api.calls[3] == ("model_info", REPO, OID, True)


def test_a_private_repo_is_refused_before_anything_is_pushed(release):
    api = FakeApi(private=True)
    with pytest.raises(push_hub.PushError, match="riêng tư"):
        push_hub.upload(release.prepare(), api)
    assert "upload_folder" not in [call[0] for call in api.calls]


@pytest.mark.parametrize(
    ("name", "kind", "fragment"),
    [
        ("model.int8.onnx", "lfs", "sha256 trên Hub khác"),
        ("model.int8.onnx", "size", "kích thước trên Hub khác"),
        ("manifest.json", "blob", "nội dung trên Hub khác"),
        ("manifest.json", "size", "kích thước trên Hub khác"),
        ("README.md", "missing", "thiếu trên Hub"),
    ],
)
def test_what_landed_on_the_hub_must_match_the_staged_files(release, name, kind, fragment):
    api = FakeApi()
    api.tamper = {name: kind}
    with pytest.raises(push_hub.PushError, match=fragment) as failure:
        push_hub.upload(release.prepare(), api)
    assert OID in str(failure.value)  # nêu revision đã lên để người chạy biết cần dọn gì


def test_a_large_file_without_lfs_info_is_only_a_warning(release, monkeypatch):
    monkeypatch.setattr(push_hub, "LFS_MIN_SIZE", 1)  # mọi tệp của bundle thu nhỏ coi như lớn
    api = FakeApi()
    api.tamper = {"tokenizer.json": "xet"}
    revision, warnings = push_hub.upload(release.prepare(), api)
    assert revision == OID
    assert any("tokenizer.json" in w and "chưa đối chiếu" in w for w in warnings)


def test_files_already_on_the_hub_that_the_manifest_does_not_list_only_warn(release):
    api = FakeApi()
    api.extra = ["cu.bin"]
    revision, warnings = push_hub.upload(release.prepare(), api)
    assert revision == OID
    assert any("cu.bin" in warning for warning in warnings)
    assert not any(".gitattributes" in warning for warning in warnings)


def test_a_revision_that_is_not_a_full_commit_sha_is_refused(release):
    with pytest.raises(push_hub.PushError, match="commit sha"):
        push_hub.upload(release.prepare(), FakeApi(oid="main"))


def test_a_successful_upload_prints_the_lines_to_pin_in_the_env_file(cli, monkeypatch, capsys):
    api = FakeApi()
    monkeypatch.setenv("EDUPULSE_HF_REPO", REPO)
    monkeypatch.setattr(push_hub, "hub_api", lambda: (api, "hf_geheim"))
    assert push_hub.main(["--upload"]) == 0
    out = capsys.readouterr().out
    assert f"EDUPULSE_HF_REPO={REPO}" in out
    assert f"EDUPULSE_HF_REVISION={OID}" in out
    assert f"EDUPULSE_MODEL_URL=https://huggingface.co/{REPO}" in out
    assert api.folder == cli / "outputs" / "hub_upload" / BUNDLE_VERSION


def test_errors_from_the_hub_never_echo_the_token(cli, monkeypatch, capsys):
    class Broken(FakeApi):
        def upload_folder(self, **kwargs):
            raise RuntimeError("401 Unauthorized cho token hf_geheim")

    monkeypatch.setenv("EDUPULSE_HF_REPO", REPO)
    monkeypatch.setattr(push_hub, "hub_api", lambda: (Broken(), "hf_geheim"))
    assert push_hub.main(["--upload"]) == 1
    captured = capsys.readouterr()
    assert "hf_geheim" not in captured.out + captured.err
    assert "***" in captured.err
    assert "RuntimeError" in captured.err


# --- Cái đẩy lên đúng là cái script tải về -------------------------------------------------------


@NEEDS_FETCH_TOOLS
def test_the_staged_folder_is_what_the_fetch_script_downloads(release, tmp_path):
    plan = release.prepare()
    hub = publish_to_hub(plan.directory, tmp_path / "hub")
    target = tmp_path / "fetched" / BUNDLE_VERSION
    result = fetch_with_script(hub, target)
    assert result.returncode == 0, result.stderr
    fetched = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
    assert fetched == {"manifest.json", *plan.manifest.files}  # README.md ở lại trên Hub
    assert load_bundle(target).manifest.model_version == plan.manifest.model_version
    builder.verify_bundle(target)  # cùng kiểm tra lúc đóng gói: ONNX và baseline nạp được
