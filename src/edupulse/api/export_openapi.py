"""Xuất hợp đồng OpenAPI của API ra `apps/web/openapi.json`, nguồn để sinh kiểu TypeScript.

    python -m edupulse.api.export_openapi            # ghi apps/web/openapi.json
    python -m edupulse.api.export_openapi --check    # thoát 1 nếu file đã commit không khớp mã

`make gen-api` chạy lệnh này rồi sinh `apps/web/src/lib/api/schema.d.ts`. Không cần bundle mô hình:
lược đồ dựng từ định nghĩa route, không chạy lifespan.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from edupulse.api.main import create_app
from edupulse.config import REPO_ROOT, load_settings

DEFAULT_OUT = REPO_ROOT / "apps" / "web" / "openapi.json"


def build_openapi() -> dict[str, Any]:
    return create_app(load_settings()).openapi()


def render(spec: dict[str, Any]) -> str:
    """JSON ổn định (cùng mã -> cùng byte) để diff trong git đọc được."""
    return json.dumps(spec, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="đường dẫn file đầu ra")
    parser.add_argument("--check", action="store_true", help="chỉ so sánh, không ghi")
    args = parser.parse_args(argv)

    text = render(build_openapi())
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.is_file() else None
        if current != text:
            print(f"{args.out} không khớp mã nguồn: chạy `make gen-api`.", file=sys.stderr)
            return 1
        print(f"{args.out} khớp.")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"đã ghi {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
