"""Đẩy bundle phát hành lên repo model công khai trên Hugging Face Hub (mốc M3, chạy tay).

    python -m ml.push_hub                 # = --dry-run: dựng và kiểm thư mục upload, KHÔNG gọi mạng
    python -m ml.push_hub --upload        # đẩy thật, rồi in các dòng cần ghi vào .env

Repo đích lấy từ `--repo` hoặc biến EDUPULSE_HF_REPO (dạng chủ/tên). Token chỉ được đọc khi
`--upload`, từ HF_TOKEN hoặc phiên `hf auth login` của người chạy; không bao giờ được in hay ghi.

Thư mục upload (mặc định `$EDUPULSE_HOME/outputs/hub_upload/<phiên bản>/`) gồm đúng các tệp mà
`scripts/fetch-artifacts.sh` tải về (manifest.json và mọi tệp nó liệt kê) cộng README.md, thẻ mô
hình của Hub. Trước khi dựng, bundle phải qua lại các kiểm tra của lúc đóng gói và thêm:

- stage release, dựng từ cây git sạch, thẻ mô hình là số trên TEST;
- tập tệp đúng danh sách cho phép, không tệp lạ, sha256 khớp manifest, ONNX và baseline nạp được;
- mô hình, cấu hình và baseline đúng thứ đã đóng băng và chấm test (`results/freeze.json`);
- sau khi dựng: sha256 của chính thư mục upload khớp manifest, và không câu nào của UIT-VSFC (gốc
  hoặc đã tách từ, từ 8 tiếng trở lên) xuất hiện trong bất kỳ tệp văn bản nào, kể cả README.

`--upload` còn đối chiếu lại thứ đã lên Hub (kích thước, sha256 của tệp LFS, git blob của tệp
nhỏ) với bản địa phương và in commit sha để ghim vào EDUPULSE_HF_REVISION.

Mã thoát: 0 xong, 1 bị từ chối hoặc lỗi, 2 cấu hình sai (repo, đăng nhập, biến môi trường).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from edupulse.bundle_schema import Manifest, ModelCard
from ml import freeze as freeze_lib
from ml.build_artifacts import (
    BUNDLE_FILES,
    BundleError,
    assert_matches_freeze,
    assert_no_source_sentences_anywhere,
    verify_bundle,
)
from ml.common import settings, sha256_file
from ml.hub_readme import MANIFEST_NAME, README_NAME, render_readme
from ml.plotting import vn

LOG_PREFIX = "push_hub"
HUB_URL = "https://huggingface.co"
PLACEHOLDER_OWNER = "your-user"  # giá trị mẫu trong .env.example
IGNORED_NAMES = frozenset({".DS_Store"})
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
LFS_MIN_SIZE = 10_000_000  # từ cỡ này Hub bắt buộc lưu tệp qua LFS/Xet; blob_id chỉ là con trỏ
# Quy tắc tên repo của Hub: chữ, số, '_', '-', '.'; không bắt đầu/kết thúc bằng '-' hay '.'.
_REPO_PART = r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?"
REPO_ID = re.compile(rf"{_REPO_PART}/{_REPO_PART}")


class PushError(RuntimeError):
    """Bundle không đủ điều kiện để đẩy, hoặc việc đẩy thất bại."""


class ConfigError(PushError):
    """Thiếu hoặc sai cấu hình do người chạy đặt (repo, đăng nhập, biến môi trường)."""


@dataclass(frozen=True)
class UploadFile:
    name: str  # đường dẫn posix trong repo
    size: int
    sha256: str


@dataclass(frozen=True)
class UploadPlan:
    directory: Path
    repo_id: str
    manifest: Manifest
    files: tuple[UploadFile, ...]
    leak_pool: int  # số câu dài của dữ liệu đã đối chiếu


class HubApi(Protocol):
    """Phần của `huggingface_hub.HfApi` mà việc đẩy dùng (để thử mà không cần mạng)."""

    def create_repo(
        self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool
    ) -> Any: ...

    def model_info(
        self, repo_id: str, *, revision: str | None = None, files_metadata: bool = False
    ) -> Any: ...

    def upload_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: Path,
        commit_message: str,
        commit_description: str,
    ) -> Any: ...


# --- Cấu hình đầu vào -------------------------------------------------------------------------


def parse_repo_id(value: str) -> str:
    text = value.strip()
    if not REPO_ID.fullmatch(text) or ".." in text or "--" in text:
        raise ConfigError(f"repo Hugging Face phải dạng chủ/tên (chữ, số, _ - .): {value!r}")
    return text


def is_placeholder(repo_id: str) -> bool:
    return repo_id.split("/", 1)[0] == PLACEHOLDER_OWNER


def clean_source_url(value: str) -> str:
    """Liên kết mã nguồn cho README; rỗng hoặc còn là giá trị mẫu thì bỏ (không có liên kết giả)."""
    url = value.strip()
    return "" if PLACEHOLDER_OWNER in url else url


# --- Đọc và kiểm bundle -----------------------------------------------------------------------


def describe(error: ValidationError) -> str:
    """Vị trí và loại lỗi, không chép giá trị."""
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['type']}" for e in error.errors()[:5])


def read_release_bundle(bundle_dir: Path) -> tuple[Manifest, ModelCard]:
    """Manifest và thẻ mô hình của bundle; chỉ nhận bản release dựng từ cây git sạch."""
    try:
        manifest = Manifest.model_validate_json((bundle_dir / MANIFEST_NAME).read_text("utf-8"))
        card = ModelCard.model_validate_json((bundle_dir / "model_card.json").read_text("utf-8"))
    except OSError as error:
        raise PushError(
            f"không đọc được bundle {bundle_dir} ({error.strerror}) — chạy `make artifacts`"
        ) from error
    except ValidationError as error:
        raise PushError(f"manifest hoặc thẻ mô hình sai schema: {describe(error)}") from error
    if manifest.stage != "release" or card.stage != "release":
        raise PushError(
            f"chỉ đẩy bản release (manifest: {manifest.stage}, thẻ mô hình: {card.stage}); "
            "bản smoke chỉ có số validation"
        )
    versions = (card.bundle_version, card.model_version)
    if versions != (manifest.bundle_version, manifest.model_version):
        raise PushError("thẻ mô hình và manifest khai phiên bản khác nhau")
    if manifest.git_commit.endswith("+dirty") or manifest.git_commit == "unknown":
        raise PushError(
            f"bundle dựng từ commit không rõ hoặc cây git bẩn ({manifest.git_commit}): "
            "dựng lại từ cây sạch bằng `make artifacts`"
        )
    if set(manifest.files) != set(BUNDLE_FILES):
        raise PushError(f"tập tệp trong manifest lệch danh sách cho phép: {sorted(manifest.files)}")
    return manifest, card


def assert_no_stray_files(bundle_dir: Path, manifest: Manifest) -> None:
    """Thư mục bundle không có gì ngoài manifest và tệp nó liệt kê (image đóng cả thư mục)."""
    present = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name not in IGNORED_NAMES
    }
    stray = sorted(present - {MANIFEST_NAME, *manifest.files})
    if stray:
        raise PushError(f"tệp lạ trong bundle (không có trong manifest): {stray}")


def verify_bundle_contents(bundle_dir: Path) -> None:
    try:
        verify_bundle(bundle_dir)
    except ValidationError as error:
        raise PushError(f"một tệp JSON của bundle sai schema: {describe(error)}") from error


# --- Dựng và kiểm thư mục upload --------------------------------------------------------------


def fresh_directory(path: Path, *, owned: bool) -> None:
    """Thư mục trống để dựng. Thư mục do công cụ sở hữu thì xoá dựng lại; thư mục người dùng chọn
    phải chưa có hoặc trống, để một `--out` gõ nhầm không xoá mất dữ liệu."""
    if path.exists():
        if owned:
            shutil.rmtree(path)
        elif not path.is_dir() or any(path.iterdir()):
            raise ConfigError(f"--out {path} đã có nội dung: chọn thư mục chưa có hoặc để trống")
    path.mkdir(parents=True, exist_ok=True)


def copy_bundle(bundle_dir: Path, directory: Path, manifest: Manifest) -> None:
    for name in (MANIFEST_NAME, *manifest.files):
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle_dir / name, target)


def verify_upload_dir(
    directory: Path, bundle_dir: Path, manifest: Manifest
) -> tuple[UploadFile, ...]:
    """Thư mục upload có đúng tập tệp, sha256 từng tệp khớp manifest, manifest giống bản gốc."""
    expected = {MANIFEST_NAME, README_NAME, *manifest.files}
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if present != expected:
        raise PushError(
            f"thư mục upload lệch danh sách: thừa {sorted(present - expected)}, "
            f"thiếu {sorted(expected - present)}"
        )
    digests = {name: sha256_file(directory / name) for name in sorted(expected)}
    for name, digest in manifest.files.items():
        if digests[name] != digest:
            raise PushError(f"{name} trong thư mục upload không khớp sha256 của manifest")
    if (directory / MANIFEST_NAME).read_bytes() != (bundle_dir / MANIFEST_NAME).read_bytes():
        raise PushError("manifest.json trong thư mục upload khác bản của bundle")
    return tuple(
        UploadFile(name, (directory / name).stat().st_size, digest)
        for name, digest in digests.items()
    )


def prepare_upload(
    bundle_dir: Path,
    directory: Path,
    *,
    repo_id: str,
    source_url: str,
    frozen: dict[str, Any],
    owned_directory: bool = True,
) -> UploadPlan:
    """Mọi kiểm tra và việc dựng thư mục upload; không chạm mạng.

    Kiểm thất bại thì thư mục upload bị xoá, để bản bị từ chối không nằm lại chờ ai đó đẩy tay.
    """
    manifest, card = read_release_bundle(bundle_dir)
    assert_no_stray_files(bundle_dir, manifest)
    verify_bundle_contents(bundle_dir)
    assert_matches_freeze(bundle_dir, frozen)

    fresh_directory(directory, owned=owned_directory)
    try:
        copy_bundle(bundle_dir, directory, manifest)
        readme = render_readme(manifest, card, repo_id=repo_id, source_url=source_url)
        (directory / README_NAME).write_text(readme, encoding="utf-8")
        files = verify_upload_dir(directory, bundle_dir, manifest)
        pool = assert_no_source_sentences_anywhere(directory)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return UploadPlan(directory, repo_id, manifest, files, pool)


# --- Đẩy lên Hub ------------------------------------------------------------------------------


def git_blob_sha1(path: Path) -> str:
    """Id blob của git (sha1 của "blob <kích thước>\\0<nội dung>"): Hub gọi nó là blob_id."""
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def remote_differences(info: Any, plan: UploadPlan) -> tuple[list[str], list[str]]:
    """(lỗi, cảnh báo) khi so danh sách tệp ở revision vừa đẩy với thư mục upload."""
    remote = {sibling.rfilename: sibling for sibling in info.siblings or []}
    errors, warnings = [], []
    for item in plan.files:
        sibling = remote.get(item.name)
        if sibling is None:
            errors.append(f"thiếu trên Hub: {item.name}")
        elif sibling.size is not None and sibling.size != item.size:
            errors.append(f"{item.name}: kích thước trên Hub khác bản địa phương")
        elif sibling.lfs is not None:
            if sibling.lfs.sha256 != item.sha256:
                errors.append(f"{item.name}: sha256 trên Hub khác bản địa phương")
        elif item.size < LFS_MIN_SIZE and sibling.blob_id is not None:
            if sibling.blob_id != git_blob_sha1(plan.directory / item.name):
                errors.append(f"{item.name}: nội dung trên Hub khác bản địa phương")
        else:
            # Tệp lớn không có thông tin LFS: blob_id chỉ là con trỏ, không so được với nội dung.
            warnings.append(f"{item.name}: Hub không trả sha256, chưa đối chiếu được nội dung")
    known = {item.name for item in plan.files} | {".gitattributes"}
    extra = sorted(set(remote) - known)
    warnings += [f"tệp thừa trên Hub (fetch-artifacts bỏ qua): {name}" for name in extra]
    return errors, warnings


def upload(plan: UploadPlan, api: HubApi) -> tuple[str, list[str]]:
    """Tạo repo công khai nếu chưa có, đẩy thư mục upload và đối chiếu; trả về (sha, cảnh báo)."""
    api.create_repo(plan.repo_id, repo_type="model", private=False, exist_ok=True)
    if api.model_info(plan.repo_id).private is not False:
        raise PushError(
            f"repo {plan.repo_id} đang riêng tư: đặt công khai trong Settings của repo rồi chạy lại"
        )
    manifest = plan.manifest
    commit = api.upload_folder(
        repo_id=plan.repo_id,
        repo_type="model",
        folder_path=plan.directory,
        commit_message=f"{manifest.model_version}: bundle {manifest.bundle_version}",
        commit_description=(
            f"Commit mã nguồn {manifest.git_commit}. "
            f"model.int8.onnx sha256 {manifest.files['model.int8.onnx']}."
        ),
    )
    revision = str(commit.oid)
    if not COMMIT_SHA.fullmatch(revision):
        raise PushError(f"Hub trả về revision không phải commit sha đầy đủ: {revision!r}")
    info = api.model_info(plan.repo_id, revision=revision, files_metadata=True)
    errors, warnings = remote_differences(info, plan)
    if errors:
        detail = "; ".join(errors)
        raise PushError(f"đối chiếu sau khi đẩy thất bại ở revision {revision}: {detail}")
    return revision, warnings


def hub_api() -> tuple[HubApi, str]:
    """HfApi kèm token của người chạy (HF_TOKEN hoặc `hf auth login`); chưa đăng nhập: từ chối."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError as error:
        raise ConfigError("thiếu huggingface_hub — chạy `uv sync --all-groups`") from error
    token = get_token()
    if not token:
        raise ConfigError(
            "chưa đăng nhập Hugging Face: chạy `uv run hf auth login` hoặc đặt biến HF_TOKEN"
        )
    return HfApi(token=token), token


# --- Dòng lệnh --------------------------------------------------------------------------------


def print_plan(plan: UploadPlan) -> None:
    total = sum(item.size for item in plan.files)
    print(
        f"{LOG_PREFIX}: thư mục upload {plan.directory}: {len(plan.files)} tệp, "
        f"{vn(total / 1e6, 1)} MB"
    )
    for item in plan.files:
        size = f"{vn(item.size / 1e6, 2)} MB"
        print(f"  {item.name:<36} {size:>11}  sha256 {item.sha256[:12]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", help="phiên bản bundle (mặc định BUNDLE_VERSION)")
    parser.add_argument("--repo", help="repo Hub dạng chủ/tên (mặc định EDUPULSE_HF_REPO)")
    parser.add_argument("--out", type=Path, help="thư mục dựng bản upload (chưa có hoặc trống)")
    parser.add_argument(
        "--source-url", help="liên kết mã nguồn cho README (mặc định EDUPULSE_SOURCE_URL)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="dựng và kiểm, không gọi mạng (mặc định)"
    )
    mode.add_argument("--upload", action="store_true", help="đẩy thật lên Hub")
    return parser


def resolve_target(args: argparse.Namespace) -> tuple[str, str, list[str]]:
    """(repo, liên kết mã nguồn, cảnh báo). Dry-run chấp nhận thiếu repo; --upload thì không."""
    repo_text = args.repo if args.repo is not None else os.environ.get("EDUPULSE_HF_REPO", "")
    repo_id = parse_repo_id(repo_text) if repo_text.strip() else ""
    warnings = []
    if not repo_id or is_placeholder(repo_id):
        if args.upload:
            raise ConfigError("chưa có repo đích thật: dùng --repo hoặc EDUPULSE_HF_REPO=chủ/tên")
        repo_id = f"{PLACEHOLDER_OWNER}/edupulse-phobert-mtl"
        warnings.append("EDUPULSE_HF_REPO chưa đặt (hoặc là mẫu): README dựng với repo mẫu")
    raw_url = args.source_url if args.source_url is not None else settings().source_url
    source_url = clean_source_url(raw_url)
    if not source_url:
        warnings.append("README chưa có liên kết mã nguồn (EDUPULSE_SOURCE_URL trống hoặc là mẫu)")
    return repo_id, source_url, warnings


def push(plan: UploadPlan) -> None:
    api, token = hub_api()
    print(f"{LOG_PREFIX}: đẩy lên {HUB_URL}/{plan.repo_id} (công khai)")
    try:
        revision, warnings = upload(plan, api)
    except PushError:
        raise
    except Exception as error:  # huggingface_hub và httpx ném nhiều kiểu lỗi riêng; đều dừng ở đây
        detail = str(error).replace(token, "***")
        raise PushError(f"đẩy lên Hub thất bại ({type(error).__name__}): {detail}") from error
    for warning in warnings:
        print(f"{LOG_PREFIX}: CẢNH BÁO: {warning}", file=sys.stderr)
    print(f"{LOG_PREFIX}: xong: {HUB_URL}/{plan.repo_id}/tree/{revision}")
    print("Ghi vào .env (và .env.example) để `make fetch-artifacts` tải đúng bản này:")
    print(f"EDUPULSE_HF_REPO={plan.repo_id}")
    print(f"EDUPULSE_HF_REVISION={revision}")
    print(f"EDUPULSE_MODEL_URL={HUB_URL}/{plan.repo_id}")
    print(
        "Kiểm đầu cuối: make fetch-artifacts EDUPULSE_ARTIFACTS_DIR=/đường/dẫn/tạm/bundle "
        "(sha256 từng tệp được kiểm theo manifest)"
    )


def run(args: argparse.Namespace) -> int:
    version = args.version or settings().bundle_version
    bundle_dir = settings().artifacts_dir(version)
    repo_id, source_url, warnings = resolve_target(args)
    try:
        frozen = freeze_lib.load_freeze()
    except freeze_lib.GuardError as error:
        raise PushError(str(error)) from error

    print(f"{LOG_PREFIX}: kiểm bundle {bundle_dir}")
    plan = prepare_upload(
        bundle_dir,
        args.out or settings().outputs_dir / "hub_upload" / version,
        repo_id=repo_id,
        source_url=source_url,
        frozen=frozen,
        owned_directory=args.out is None,
    )
    manifest = plan.manifest
    print(
        f"{LOG_PREFIX}: {manifest.model_version} · stage={manifest.stage} · commit "
        f"{manifest.git_commit}: schema, sha256 ({len(manifest.files)} tệp) và ONNX hợp lệ, "
        "khớp freeze.json"
    )
    print_plan(plan)
    print(f"{LOG_PREFIX}: guard câu gốc: đối chiếu {vn(plan.leak_pool, 0)} câu dài, 0 xuất hiện")
    for warning in warnings:
        print(f"{LOG_PREFIX}: CẢNH BÁO: {warning}", file=sys.stderr)

    if args.upload:
        push(plan)
    else:
        print(
            f"{LOG_PREFIX}: DRY-RUN xong, chưa gọi mạng. Đẩy thật: đăng nhập (`uv run hf auth "
            "login`), đặt EDUPULSE_HF_REPO rồi chạy `uv run --env-file .env python -m "
            "ml.push_hub --upload`"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except ConfigError as error:
        print(f"{LOG_PREFIX}: cấu hình sai: {error}", file=sys.stderr)
        return 2
    except (PushError, BundleError) as error:
        print(f"{LOG_PREFIX}: LỖI: {error}", file=sys.stderr)
        return 1
    except ValueError as error:  # giá trị sai trong biến môi trường (edupulse.config)
        print(f"{LOG_PREFIX}: cấu hình sai: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
