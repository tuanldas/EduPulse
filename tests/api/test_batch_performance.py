"""Batch 1.000 câu thật qua HTTP: thời gian end-to-end và độ trễ /api/analyze giữa chừng."""

import csv
import io
import threading
import time

import pytest
from batch_helpers import run, upload, wait_for
from fastapi.testclient import TestClient

from edupulse.config import load_settings

pytestmark = [pytest.mark.model, pytest.mark.slow, pytest.mark.data]

N = 1000
# Tiêu chí gồm cả giải thích tự động 20 câu; đo chính thức trong Docker 2 CPU.
BATCH_BUDGET_S = 30.0
ANALYZE_EXTRA_S = 1.0  # /api/analyze chen vào lúc batch chạy được chậm thêm tối đa ngần này


@pytest.fixture(scope="module")
def validation_csv() -> bytes:
    path = load_settings().data_dir / "raw" / "validation.parquet"
    if not path.is_file():
        pytest.skip("chưa có data/raw — chạy `make data`")
    import pyarrow.parquet as pq

    sentences = pq.read_table(path).column("sentence").to_pylist()[:N]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow(["Góp ý"])
    writer.writerows([s] for s in sentences)
    return out.getvalue().encode("utf-8-sig")


def test_a_thousand_real_sentences_finish_within_the_budget_while_analyze_stays_fast(
    live_client: TestClient, validation_csv: bytes
):
    batch_id = upload(live_client, validation_csv).json()["batch_id"]
    latencies: list[float] = []
    stop = threading.Event()

    def single_sentences() -> None:
        while not stop.is_set():
            started = time.perf_counter()
            response = live_client.post("/api/analyze", json={"text": "Thầy dạy rất nhiệt tình."})
            latencies.append(time.perf_counter() - started)
            assert response.status_code == 200
            time.sleep(0.2)

    caller = threading.Thread(target=single_sentences)
    started = time.perf_counter()
    assert run(live_client, batch_id).status_code == 202
    caller.start()
    try:
        status = wait_for(live_client, batch_id, timeout=300)
    finally:
        stop.set()
        caller.join()
    elapsed = time.perf_counter() - started

    assert status["state"] == "done"
    assert elapsed < BATCH_BUDGET_S, f"{elapsed:.1f}s cho {N} câu"
    assert latencies and max(latencies) < ANALYZE_EXTRA_S, max(latencies)
    results = live_client.get(f"/api/batches/{batch_id}/results").json()
    summary = results["summary"]
    assert summary["n_analyzed"] + summary["n_skipped"] == N
    assert summary["n_explained"] == 20
    print(
        f"\n{N} câu: {elapsed:.1f}s, dựng kết quả trong {summary['elapsed_s']}s, "
        f"/api/analyze giữa chừng: n={len(latencies)} max={max(latencies) * 1000:.0f}ms"
    )
