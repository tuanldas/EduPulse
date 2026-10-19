"""Image chạy chỉ cài group `app`: nạp API và suy luận không được kéo theo gói của `train`/`demo`.

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
# scikit-learn và skops chỉ phục vụ nút so sánh với mô hình cơ sở: nạp lười, tốn ~100MB RSS.
LAZY_MODULES = ("sklearn", "skops", "scipy")

SERVE_ONE_REQUEST = """
from fastapi.testclient import TestClient
from edupulse.api.main import create_app

with TestClient(create_app()) as client:
    assert client.post("/api/analyze", json={"text": "Thầy dạy rất hay"}).status_code == 200
    assert client.post("/api/explain", json={"text": "Thầy dạy rất hay"}).status_code == 200
"""

SERVE_COMPARISON = """
import json, sys
from fastapi.testclient import TestClient
from edupulse.api.main import create_app

LAZY = %(lazy)r

with TestClient(create_app()) as client:
    plain = client.post("/api/analyze", json={"text": "Thầy dạy rất hay"})
    assert plain.status_code == 200
    before = [m for m in LAZY if m in sys.modules]
    compared = client.post("/api/analyze", json={"text": "Thầy dạy rất hay", "compare": True})
    assert compared.status_code == 200 and compared.json()["baseline"] is not None
    after = [m for m in LAZY if m in sys.modules]
    print(json.dumps([before, after]))
"""


def _modules_after(statement: str, modules: tuple[str, ...]) -> list[str]:
    """Trong tiến trình mới: chạy `statement` rồi liệt kê những module trong `modules` đã bị nạp."""
    probe = (
        f"{statement}\n"
        "import json, sys\n"
        f"print(json.dumps([m for m in {modules!r} if m in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def _heavy_modules_after(statement: str) -> list[str]:
    return _modules_after(statement, HEAVY_MODULES)


@pytest.mark.parametrize(
    "statement",
    [
        "import edupulse.api.main",
        "from edupulse.api.main import create_app; create_app()",
        "import edupulse.engine.service, edupulse.engine.bench",
    ],
)
def test_importing_the_api_does_not_pull_in_training_or_demo_packages(statement: str):
    assert _heavy_modules_after(statement) == []


@pytest.mark.model
def test_serving_a_request_does_not_pull_in_training_or_demo_packages():
    assert _heavy_modules_after(SERVE_ONE_REQUEST) == []


@pytest.mark.parametrize(
    "statement",
    ["import edupulse.api.main", "from edupulse.api.main import create_app; create_app()"],
)
def test_scikit_learn_is_not_imported_just_by_loading_the_api(statement: str):
    assert _modules_after(statement, LAZY_MODULES) == []


@pytest.mark.model
def test_scikit_learn_is_loaded_only_when_a_comparison_is_asked_for():
    probe = SERVE_COMPARISON % {"lazy": LAZY_MODULES}
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr
    before, after = json.loads(result.stdout.splitlines()[-1])
    assert before == []  # phân tích và giải thích thường không đụng tới scikit-learn
    assert "sklearn" in after and "skops" in after  # nạp lười ở lần so sánh đầu tiên
