"""Dữ liệu demo dựng lúc build: chọn câu, định dạng tệp, nạp lúc chạy và nhánh `ok: false`."""

import csv
import io
import json
from collections import Counter
from pathlib import Path

import pytest

from edupulse import demo_data as demo
from edupulse.bundle_schema import DemoManifest
from edupulse.config import load_settings
from edupulse.labels import SENTIMENT, TOPIC

CELLS = [(s, t) for s in SENTIMENT for t in TOPIC]


def synthetic_rows(per_cell: int = 40) -> list[demo.SourceRow]:
    """Dân số giả có đủ 12 ô cảm xúc × chủ đề, kích thước mỗi ô khác nhau, câu không trùng."""
    rows = []
    for i, (sentiment, topic) in enumerate(CELLS):
        for j in range(per_cell * (i % 4 + 1)):
            rows.append(demo.SourceRow(f"câu số {i} {j} nói về điều gì đó hay .", sentiment, topic))
    return rows


# --- Làm đẹp câu và bỏ dấu ----------------------------------------------------------------------


def test_beautify_joins_punctuation_and_capitalizes():
    assert demo.beautify("slide giảng dạy rất hay ,  dễ hiểu .") == (
        "Slide giảng dạy rất hay, dễ hiểu."
    )
    assert demo.beautify("thầy ( cô ) dạy hay !") == "Thầy (cô) dạy hay!"


def test_beautify_shows_masked_names_as_a_placeholder():
    assert demo.beautify("thầy wzjwz208 dạy hay") == "Thầy [TÊN] dạy hay"


def test_beautify_keeps_emoticon_colons_attached_to_the_following_character():
    assert demo.beautify("hay lắm :)") == "Hay lắm :)"


@pytest.mark.parametrize("lead", ["=", "+", "-", "@", "- ", "\t=", "= + "])
def test_beautify_drops_characters_that_excel_reads_as_formulas(lead: str):
    assert demo.beautify(f"{lead}thầy dạy hay") == "Thầy dạy hay"


def test_strip_diacritics_handles_d_with_stroke():
    assert demo.strip_diacritics("Thầy giảng dạy đúng giờ") == "Thay giang day dung gio"
    assert demo.strip_diacritics("Đại học") == "Dai hoc"


# --- File mẫu ----------------------------------------------------------------------------------


def test_sample_has_the_requested_size_and_covers_every_cell():
    rows = synthetic_rows()
    sample = demo.pick_sample(rows, 60)
    assert len(sample) == 60
    assert len({r.sentence for r in sample}) == 60
    covered = Counter((r.sentiment, r.topic) for r in sample)
    assert set(covered) == set(CELLS)
    assert min(covered.values()) >= demo.MIN_PER_CELL


def test_sample_is_proportional_to_the_population():
    rows = synthetic_rows(per_cell=100)
    population = Counter((r.sentiment, r.topic) for r in rows)
    sample = Counter((r.sentiment, r.topic) for r in demo.pick_sample(rows, 120))
    for cell, size in population.items():
        expected = 120 * size / len(rows)
        assert abs(sample[cell] - expected) <= 2, cell


def test_sample_is_deterministic_per_seed():
    rows = synthetic_rows()
    first = demo.pick_sample(rows, 60, seed=7)
    assert first == demo.pick_sample(rows, 60, seed=7)
    assert first != demo.pick_sample(rows, 60, seed=8)


def test_sample_skips_masked_names_duplicates_and_unusual_lengths():
    rows = [
        demo.SourceRow("thầy wzjwz12 dạy hay lắm .", "positive", "lecturer"),
        demo.SourceRow("hay", "positive", "lecturer"),  # quá ngắn
        demo.SourceRow(" ".join(["từ"] * 60), "positive", "lecturer"),  # quá dài
        demo.SourceRow("Thầy dạy rất hay .", "positive", "lecturer"),
        demo.SourceRow("thầy dạy rất hay .", "positive", "lecturer"),  # trùng chữ
        demo.SourceRow("Phòng máy rất nóng .", "negative", "facility"),
    ]
    sample = demo.pick_sample(rows, 2)
    assert sorted(r.sentence.lower() for r in sample) == [
        "phòng máy rất nóng .",
        "thầy dạy rất hay .",
    ]


def test_sample_refuses_a_size_larger_than_the_population():
    with pytest.raises(demo.DemoBuildError, match="chỉ có"):
        demo.pick_sample(synthetic_rows(per_cell=1), 10_000)


def test_groups_are_fixed_by_the_seed_and_cover_a_to_e():
    groups = demo.assign_groups(200)
    assert groups == demo.assign_groups(200)
    assert set(groups) == set(demo.GROUP_NAMES)
    assert groups != demo.assign_groups(200, seed=1)


def test_sample_csv_has_a_bom_two_columns_and_marks_the_groups_as_illustrative():
    sample = demo.pick_sample(synthetic_rows(), 30)
    data = demo.sample_csv_bytes(sample)
    assert data.startswith(b"\xef\xbb\xbf")
    parsed = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    assert parsed[0] == ["Góp ý", "Nhóm (minh hoạ)"]
    assert "minh hoạ" in demo.GROUP_COLUMN
    assert len(parsed) == 31
    assert all(row[1] in demo.GROUP_NAMES for row in parsed[1:])
    assert all(row[0][0].isupper() for row in parsed[1:])


# --- Câu mẫu của trang 1 -----------------------------------------------------------------------


def test_ui_samples_cover_the_target_cells_and_end_with_a_no_diacritics_variant():
    # Mỗi ô một câu riêng biệt, đủ 6 từ.
    rows = [
        demo.SourceRow(f"câu mẫu thứ {chr(97 + i)} nói về điều hay nhé .", s, t)
        for i, (s, t) in enumerate(CELLS)
    ]
    samples = demo.pick_ui_samples(rows)
    assert len(samples) == len(demo.UI_SAMPLE_TARGETS) + 1
    assert [s.no_diacritics for s in samples] == [False] * 6 + [True]
    assert samples[-1].text == demo.strip_diacritics(samples[0].text)
    assert len({s.text for s in samples[:-1]}) == 6


def test_ui_samples_fail_loudly_when_a_target_cell_has_no_sentence():
    rows = [demo.SourceRow("câu mẫu thứ nhất nói về điều hay nhé .", "positive", "lecturer")]
    with pytest.raises(demo.DemoBuildError, match="không có câu ngắn"):
        demo.pick_ui_samples(rows)


def test_ui_samples_prefer_clean_sentences():
    rows = [
        demo.SourceRow(
            "thầy dạy ( rất hay ) mà lại vui vẻ nữa", "positive", "lecturer"
        ),  # có ngoặc
        demo.SourceRow("thầy dạy rất hay mà lại vui vẻ nữa .", "positive", "lecturer"),
        demo.SourceRow("thầy dạy tạm được thôi mà cũng ổn .", "neutral", "lecturer"),
        demo.SourceRow("chương trình học rất hay và bổ ích nữa .", "positive", "training_program"),
        demo.SourceRow("phòng máy cũ nên cần được nâng cấp .", "negative", "facility"),
        demo.SourceRow("em không có ý kiến gì thêm về việc này .", "neutral", "others"),
        demo.SourceRow("thầy hay trễ giờ nên cả lớp rất chán .", "negative", "lecturer"),
    ]
    samples = demo.pick_ui_samples(rows)
    assert samples[0].text == "Thầy dạy rất hay mà lại vui vẻ nữa."


# --- Nạp lúc chạy ------------------------------------------------------------------------------


def write_demo(directory: Path, bundle: str = "b1", n_rows: int = 2, ok: bool = True) -> None:
    if not ok:
        demo.write_manifest(directory, DemoManifest(ok=False, reason="vì sao đó"))
        return
    rows = [
        demo.DemoRowData(
            display_text=f"Câu {i}.",
            model_text=f"câu {i} .",
            sentiment="negative",
            topic="facility",
            sentiment_confidence=0.9,
            topic_confidence=0.8,
            n_words=2,
        )
        for i in range(n_rows)
    ]
    demo.write_demo_files(
        directory,
        sample=[],
        ui_samples=[demo.UiSampleItem(text="Một câu.", no_diacritics=False)],
        rows=rows,
        bundle_version=bundle,
        revision="rev",
    )


def test_load_demo_round_trips_what_the_builder_writes(tmp_path: Path):
    write_demo(tmp_path)
    loaded = demo.load_demo(tmp_path, "b1")
    assert loaded.available and loaded.reason is None
    assert len(loaded.rows) == 2 and loaded.rows[0].display_text == "Câu 0."
    assert loaded.ui_samples[0].text == "Một câu."
    assert loaded.sample_csv.startswith(b"\xef\xbb\xbf")
    assert loaded.manifest is not None and loaded.manifest.n_rows == 2


def test_load_demo_without_a_manifest_explains_how_to_build_it(tmp_path: Path):
    loaded = demo.load_demo(tmp_path, "b1")
    assert not loaded.available and loaded.rows == () and loaded.sample_csv == b""
    assert "make demo-data" in (loaded.reason or "")


def test_load_demo_passes_the_failure_reason_of_the_build(tmp_path: Path):
    write_demo(tmp_path, ok=False)
    loaded = demo.load_demo(tmp_path, "b1")
    assert not loaded.available and loaded.reason == "vì sao đó"


def test_load_demo_rejects_results_computed_with_another_bundle(tmp_path: Path):
    write_demo(tmp_path, bundle="b1")
    loaded = demo.load_demo(tmp_path, "b2")
    assert not loaded.available
    assert "b1" in (loaded.reason or "") and "b2" in (loaded.reason or "")


@pytest.mark.parametrize(
    "name", [demo.SAMPLE_CSV, demo.UI_SAMPLES, demo.DEMO_RESULTS, demo.DEMO_MANIFEST]
)
def test_load_demo_survives_missing_or_corrupt_files(tmp_path: Path, name: str):
    write_demo(tmp_path)
    (tmp_path / name).unlink()
    assert not demo.load_demo(tmp_path, "b1").available
    (tmp_path / name).write_text("{ hỏng", encoding="utf-8")
    loaded = demo.load_demo(tmp_path, "b1")
    assert not loaded.available and loaded.reason


def test_load_demo_rejects_a_row_count_that_disagrees_with_the_manifest(tmp_path: Path):
    write_demo(tmp_path, n_rows=3)
    manifest = json.loads((tmp_path / demo.DEMO_MANIFEST).read_text("utf-8"))
    manifest["n_rows"] = 99
    (tmp_path / demo.DEMO_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    assert not demo.load_demo(tmp_path, "b1").available


def test_demo_rows_never_carry_the_gold_labels():
    assert {"gold", "label", "sentiment_gold"}.isdisjoint(demo.DemoRowData.model_fields)


# --- Dòng lệnh: bỏ qua và nguồn không có -------------------------------------------------------


def test_skip_writes_a_manifest_that_disables_the_demo(tmp_path: Path):
    assert demo.main(["--out", str(tmp_path), "--skip"]) == 0
    manifest = DemoManifest.model_validate_json((tmp_path / demo.DEMO_MANIFEST).read_bytes())
    assert manifest.ok is False and "EDUPULSE_BUILD_DEMO=0" in (manifest.reason or "")
    assert not demo.load_demo(tmp_path, "b1").available
    assert not (tmp_path / demo.SAMPLE_CSV).exists()


def test_skip_accepts_a_custom_reason(tmp_path: Path):
    assert demo.main(["--out", str(tmp_path), "--skip", "tắt thủ công"]) == 0
    assert demo.load_demo(tmp_path, "b1").reason == "tắt thủ công"


def test_unreachable_hub_disables_the_demo_without_failing(tmp_path: Path, monkeypatch):
    def offline(*args, **kwargs):
        raise ConnectionError("không có mạng")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    # Một thư mục dữ liệu trống để không dùng bản `make data` có sẵn ở máy chạy test.
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path / "home"))
    assert demo.main(["--out", str(tmp_path / "out")]) == 0
    loaded = demo.load_demo(tmp_path / "out", "b1")
    assert not loaded.available
    assert "Hugging Face" in (loaded.reason or "") and "ConnectionError" in (loaded.reason or "")
    assert "không có mạng" not in (loaded.reason or "")  # chỉ nêu loại lỗi, không chép thông điệp


def test_a_wrong_source_dir_is_an_error_not_a_silent_skip(tmp_path: Path):
    with pytest.raises(demo.DemoBuildError, match="Thiếu"):
        demo.main(["--out", str(tmp_path / "out"), "--source-dir", str(tmp_path)])
    assert not (tmp_path / "out" / demo.DEMO_MANIFEST).exists()


def test_a_corrupt_local_source_is_rejected_by_its_pinned_hash(tmp_path: Path):
    for split in demo.REMOTE_PATHS:
        (tmp_path / f"{split}.parquet").write_bytes(b"khong phai parquet")
    with pytest.raises(demo.SourceUnavailable, match="sha256"):
        demo.local_sources(tmp_path)


def test_reasons_are_truncated():
    assert len(demo._short("x" * 1000)) == 300


# --- Dữ liệu thật (cần `make data`) -------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_dir() -> Path:
    pytest.importorskip("pyarrow.parquet")
    path = load_settings().data_dir / "raw"
    if not all((path / f"{split}.parquet").is_file() for split in demo.REMOTE_PATHS):
        pytest.skip("chưa có data/raw — chạy `make data`")
    return path


@pytest.fixture(scope="module")
def real_rows(raw_dir: Path) -> dict[str, list[demo.SourceRow]]:
    sources = demo.local_sources(raw_dir)
    return {split: demo.read_source(path, split) for split, path in sources.items()}


def test_pinned_hashes_match_the_manifest_of_the_processed_data():
    path = load_settings().data_dir / "processed" / "MANIFEST.json"
    if not path.is_file():
        pytest.skip("chưa có data/processed — chạy `make data`")
    raw = json.loads(path.read_text("utf-8"))["raw_sha256"]
    assert {s: raw[s] for s in demo.EXPECTED_SHA256} == demo.EXPECTED_SHA256


def test_find_sources_prefers_the_downloaded_raw_data(raw_dir: Path):
    assert demo.find_sources(None, load_settings()) == demo.local_sources(raw_dir)


def test_real_sample_has_200_rows_with_groups_a_to_e(real_rows):
    sample = demo.pick_sample(real_rows["validation"])
    assert len(sample) == demo.SAMPLE_SIZE
    validation = {r.sentence for r in real_rows["validation"]}
    assert all(r.sentence in validation and "wzjwz" not in r.sentence for r in sample)
    parsed = list(csv.DictReader(io.StringIO(demo.sample_csv_bytes(sample).decode("utf-8-sig"))))
    assert len(parsed) == 200
    assert set(r["Nhóm (minh hoạ)"] for r in parsed) == set(demo.GROUP_NAMES)
    eligible = Counter(
        (r.sentiment, r.topic)
        for r in real_rows["validation"]
        if demo._eligible(r, demo.SAMPLE_WORDS)
    )
    cells = Counter((r.sentiment, r.topic) for r in sample)
    assert set(cells) == set(eligible)  # mọi ô còn câu hợp lệ đều có mặt
    assert all(cells[c] >= min(demo.MIN_PER_CELL, n) for c, n in eligible.items())


def test_real_ui_samples_are_short_and_seven(real_rows):
    samples = demo.pick_ui_samples(real_rows["validation"])
    assert len(samples) == 7
    low, high = demo.UI_SAMPLE_WORDS
    # Làm đẹp bỏ dấu cách trước dấu câu nên số "từ" có thể ít hơn số token gốc một chút.
    assert all(low - 1 <= len(s.text.split()) <= high for s in samples[:6])
    assert samples[6].no_diacritics and samples[6].text.isascii()


# --- Chạy mô hình thật -------------------------------------------------------------------------


@pytest.mark.model
@pytest.mark.java
def test_build_writes_files_that_load_back_and_match_the_analyzer(
    tmp_path: Path, real_rows, analyzer_for_demo
):
    validation, test = real_rows["validation"][:400], real_rows["test"][:30]
    manifest = demo.build_demo(
        tmp_path, validation, test, analyzer_for_demo, revision="rev", sample_size=40
    )
    assert manifest.ok and manifest.n_rows == 30
    loaded = demo.load_demo(tmp_path, analyzer_for_demo.bundle.manifest.bundle_version)
    assert loaded.available and len(loaded.rows) == 30 and len(loaded.ui_samples) == 7
    assert len(list(csv.reader(io.StringIO(loaded.sample_csv.decode("utf-8-sig"))))) == 41
    for source, row in zip(test, loaded.rows, strict=True):
        direct = analyzer_for_demo.analyze(source.sentence)
        assert (row.sentiment, row.topic) == (direct.sentiment.label, direct.topic.label)
        assert row.model_text == direct.prepared.model_text
        assert 0 < row.sentiment_confidence <= 1 and row.n_words == direct.prepared.n_words
        assert "wzjwz" not in row.display_text


@pytest.fixture(scope="module")
def analyzer_for_demo():
    from edupulse.engine.service import Analyzer
    from edupulse.text.segment import SegmenterError

    try:
        return Analyzer.load(load_settings())
    except SegmenterError as exc:
        pytest.skip(f"không khởi động được bộ tách từ: {exc}")
