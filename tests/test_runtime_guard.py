"""Image chạy chỉ cài group `app`: nạp API không được kéo theo gói của group `train` hay `demo`.

Mỗi lần kiểm chạy trong một tiến trình Python mới, vì phiên pytest có thể đã import torch hay
pandas qua test khác. Kiểm bằng `sys.modules` nên đúng cả khi máy dev đã cài đủ mọi group.
Để kiểm trên môi trường chỉ có group `app`:
    uv run --isolated --no-default-groups --group app --group dev pytest tests/test_runtime_guard.py
"""

import json
import subprocess
import sys

import pytest

HEAVY_MODULES = ("torch", "transformers", "pandas", "pyarrow", "huggingface_hub")


def _heavy_modules_after(statement: str) -> list[str]:
    probe = (
        f"{statement}\n"
        "import json, sys\n"
        f"print(json.dumps([m for m in {HEAVY_MODULES!r} if m in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize(
    "statement",
    [
        "import edupulse.api.main",
        "from edupulse.api.main import create_app; create_app()",
    ],
)
def test_importing_the_api_does_not_pull_in_training_or_demo_packages(statement: str):
    assert _heavy_modules_after(statement) == []
