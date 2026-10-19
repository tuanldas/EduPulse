#!/usr/bin/env bash
# Tải bundle mô hình từ Hugging Face Hub (repo công khai, revision ghim) vào artifacts/<phiên bản>/
# rồi kiểm sha256 từng tệp theo manifest.json:  make fetch-artifacts  (hoặc bash scripts/fetch-artifacts.sh)
#
# Chỉ cần bash, curl và sha256sum (Linux) hoặc shasum (macOS); không cần Python. Cấu hình lấy từ
# biến môi trường, thiếu thì đọc .env ở gốc repo (biến môi trường đã đặt thì thắng):
#   EDUPULSE_HF_REPO       repo model dạng chủ/tên (`python -m ml.push_hub --upload` in ra)
#   EDUPULSE_HF_REVISION   commit sha đầy đủ (40 ký tự hex); không nhận tên nhánh hay tag
#   BUNDLE_VERSION         phiên bản bundle, mặc định v1.1
#   EDUPULSE_ARTIFACTS_DIR thư mục đích; mặc định $EDUPULSE_HOME/artifacts/<phiên bản>, còn
#                          EDUPULSE_HOME mặc định là gốc repo (cùng quy tắc với Makefile và config.py)
#   HF_ENDPOINT            máy chủ Hub, mặc định https://huggingface.co (file:// để thử không cần mạng)
#   EDUPULSE_ENV_FILE      file .env khác thay cho .env ở gốc repo
#   FORCE=1                thay bundle đã có ở thư mục đích (mặc định từ chối)
#
# Chỉ tải manifest.json và các tệp nó liệt kê (README.md và tệp lạ trên Hub bị bỏ qua), nên thư mục
# đích đúng bằng thứ `make docker-build` đóng vào image: bundle ml.push_hub đẩy lên là bundle tải về.
# Tải vào thư mục tạm, kiểm hết rồi mới đổi tên sang thư mục đích, nên không bao giờ để lại bundle dở.
set -euo pipefail

cd "$(dirname "$0")/.."
repo_root=$(pwd)

die() {
  echo "fetch-artifacts: $*" >&2
  exit 1
}

# Nạp khoá còn trống từ file .env (cú pháp KEY=value của .env.example: không dấu nháy).
env_file=${EDUPULSE_ENV_FILE:-.env}
read_env() {
  local key=$1 value
  if [ -n "${!key:-}" ] || [ ! -f "$env_file" ]; then
    return 0
  fi
  value=$(sed -n -E "s/^${key}=(.*)$/\\1/p" "$env_file" | tail -n 1 | tr -d '\r' |
    sed -E 's/[[:space:]]+$//')
  if [ -n "$value" ]; then
    export "$key=$value"
  fi
}
for key in EDUPULSE_HF_REPO EDUPULSE_HF_REVISION BUNDLE_VERSION EDUPULSE_HOME \
  EDUPULSE_ARTIFACTS_DIR HF_ENDPOINT; do
  read_env "$key"
done

repo=${EDUPULSE_HF_REPO:-}
revision=${EDUPULSE_HF_REVISION:-}
version=${BUNDLE_VERSION:-v1.1}
endpoint=${HF_ENDPOINT:-https://huggingface.co}
endpoint=${endpoint%/}
target=${EDUPULSE_ARTIFACTS_DIR:-${EDUPULSE_HOME:-$repo_root}/artifacts/$version}

repo_re='^[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*$'
sha1_re='^[0-9a-f]{40}$'
version_re='^[A-Za-z0-9][A-Za-z0-9._-]*$'
name_re='^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$'

[ -n "$repo" ] || die "chưa đặt EDUPULSE_HF_REPO (xem .env.example)"
case $repo in
  your-user/*) die "EDUPULSE_HF_REPO còn là giá trị mẫu ($repo): đặt repo thật trong .env" ;;
esac
[[ $repo =~ $repo_re ]] || die "EDUPULSE_HF_REPO phải dạng chủ/tên, nhận '$repo'"
[ -n "$revision" ] || die "chưa đặt EDUPULSE_HF_REVISION: ghim commit sha mà ml.push_hub in ra"
[[ $revision =~ $sha1_re ]] || die "EDUPULSE_HF_REVISION phải là commit sha đầy đủ (40 ký tự hex), nhận '$revision'"
[[ $version =~ $version_re ]] || die "BUNDLE_VERSION chỉ gồm chữ, số, '.', '_', '-': '$version'"
if [ -e "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
  [ "${FORCE:-0}" = 1 ] || die "$target đã có nội dung: xoá nó hoặc chạy lại với FORCE=1 để thay"
  [ -f "$target/manifest.json" ] || die "$target không phải bundle (thiếu manifest.json) nên không thay"
fi

if command -v sha256sum >/dev/null 2>&1; then
  verify=(sha256sum -c --quiet)
elif command -v shasum >/dev/null 2>&1; then
  verify=(shasum -a 256 -c --quiet)
else
  die "cần sha256sum (Linux) hoặc shasum (macOS) để kiểm tệp tải về"
fi
if [ -t 2 ]; then
  progress=(--progress-bar)
else
  progress=(--silent --show-error)
fi

base="$endpoint/$repo/resolve/$revision"
fetch() { # fetch <đường dẫn trong repo> <tệp đích>
  curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 20 \
    --proto '=https,file' --proto-redir '=https' "${progress[@]}" \
    --output "$2" "$base/$1" </dev/null || die "không tải được $1 từ $base"
}

mkdir -p "$(dirname "$target")"
work=$(mktemp -d "$(dirname "$target")/.fetch-$version.XXXXXX")
trap 'rm -rf "$work"' EXIT

echo "fetch-artifacts: $repo @ ${revision:0:12} -> $target" >&2
fetch manifest.json "$work/manifest.json"

# Mỗi tệp một dòng "tên": "sha256" trong khối "files" (ml.common.write_json ghi đúng dạng này);
# đổi thành "sha256  tên" để sha256sum/shasum -c đọc thẳng.
sums=$(sed -n -E '/"files"[[:space:]]*:[[:space:]]*\{/,/^[[:space:]]*\}/{s/^[[:space:]]*"([^"]+)"[[:space:]]*:[[:space:]]*"([0-9a-f]{64})"[[:space:]]*,?[[:space:]]*$/\2  \1/p;}' \
  "$work/manifest.json")
listed=$(printf '%s\n' "$sums" | grep -c . || true)
digests=$(grep -c -E '"[0-9a-f]{64}"' "$work/manifest.json" || true)
if [ "$listed" -eq 0 ] || [ "$listed" -ne "$digests" ]; then
  die "không đọc được danh sách tệp trong manifest.json (cần mỗi tệp một dòng \"tên\": \"sha256\")"
fi

declared=$(sed -n -E 's/^[[:space:]]*"bundle_version"[[:space:]]*:[[:space:]]*"([^"]+)".*$/\1/p' \
  "$work/manifest.json" | head -n 1)
[ "$declared" = "$version" ] || die "manifest khai bundle_version '$declared', khác BUNDLE_VERSION '$version'"

# Kiểm hết tên tệp trước khi tải bất kỳ tệp nào: tên như ../x không được chạm tới đĩa hay mạng.
while read -r _ name; do
  [[ $name =~ $name_re ]] || die "tên tệp không an toàn trong manifest: '$name'"
done <<<"$sums"

while read -r _ name; do
  mkdir -p "$work/$(dirname "$name")"
  echo "fetch-artifacts: tải $name" >&2
  fetch "$name" "$work/$name"
done <<<"$sums"

(cd "$work" && printf '%s\n' "$sums" | "${verify[@]}" -) || die "sha256 không khớp manifest.json: bundle tải về bị hỏng hoặc bị sửa"

rm -rf "$target"
chmod 755 "$work"
mv "$work" "$target"
trap - EXIT
echo "fetch-artifacts: xong, $listed tệp khớp sha256 trong $target" >&2
