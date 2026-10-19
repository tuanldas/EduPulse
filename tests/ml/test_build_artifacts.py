"""Đóng gói bundle: whitelist, schema, đồng nhất normalizer, và guard không lọt câu gốc."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from bundle_factory import BASE_ROWS, TOTAL_ROWS, UNK, background, tiny_onnx, write_bundle

from edupulse.bundle_schema import GoldenPredictions
from edupulse.text import NORMALIZER_VERSION
from edupulse.text.pii import mask
from ml import build_artifacts as builder
from ml.common import load_processed, sha256_file, write_json


@pytest.fixture()
def bundle(tmp_path) -> Path:
    """Bundle tối thiểu nhưng đúng cấu trúc: đủ tệp, đúng schema, ONNX có đúng tên vào/ra."""
    return write_bundle(tmp_path / "bundle")


def test_bundle_file_list_is_flat_or_one_level_and_unique():
    assert len(set(builder.BUNDLE_FILES)) == len(builder.BUNDLE_FILES)
    for name in builder.BUNDLE_FILES:
        assert ".." not in name and not name.startswith("/")
        assert name.count("/") <= 1


def test_valid_bundle_passes_verification(bundle):
    manifest = builder.verify_bundle(bundle)
    assert manifest.bundle_version == "v-test"
    assert set(manifest.files) == set(builder.BUNDLE_FILES)


def test_tampered_file_is_caught_by_sha256(bundle):
    (bundle / "tokenizer.json").write_text('{"x": 1}', encoding="utf-8")
    with pytest.raises(builder.BundleError, match="tokenizer.json.*sha256"):
        builder.verify_bundle(bundle)


def test_file_outside_the_whitelist_is_refused(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
    manifest["files"]["extra.bin"] = "a" * 64
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(builder.BundleError, match="danh sách cho phép"):
        builder.verify_bundle(bundle)


def test_invalid_json_content_fails_the_schema(bundle):
    broken = json.loads((bundle / "calibration.json").read_text("utf-8"))
    broken["tau"].pop("10")
    write_json(bundle / "calibration.json", broken)
    manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
    manifest["files"]["calibration.json"] = sha256_file(bundle / "calibration.json")
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="ngân sách"):
        builder.verify_bundle(bundle)


def test_wrong_onnx_output_names_are_caught(bundle):
    tiny_onnx(bundle / "model.int8.onnx", output_names=("a", "b"))
    manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
    manifest["files"]["model.int8.onnx"] = sha256_file(bundle / "model.int8.onnx")
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(builder.BundleError, match="đầu ra ONNX"):
        builder.verify_bundle(bundle)


def test_expanded_embedding_rows_that_do_not_behave_as_unk_are_caught(bundle):
    tiny_onnx(bundle / "model.int8.onnx", extra_rows_match_unk=False)
    manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
    manifest["files"]["model.int8.onnx"] = sha256_file(bundle / "model.int8.onnx")
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(builder.BundleError, match="không hành xử như <unk>"):
        builder.verify_bundle(bundle)


def test_manifest_records_the_embedding_expansion(bundle):
    spec = builder.verify_bundle(bundle).onnx
    assert (spec.base_vocab_size, spec.unk_id, spec.embedding_rows) == (BASE_ROWS, UNK, TOTAL_ROWS)


def passing_gate(run_name="mtl-s1", passed=True):
    return {
        "run": run_name,
        "normalizer_version": NORMALIZER_VERSION,
        "gate": {"passed": passed},
        "embedding": {"base_vocab_size": 64001, "unk_id": 3, "embedding_rows": 66119},
    }


def test_gate_must_have_passed_for_the_same_run(tmp_path):
    run = tmp_path / "mtl-s1"
    assert (
        builder.require_passed_gate(passing_gate(), run, allow_failed=False)["embedding_rows"]
        == 66119
    )
    with pytest.raises(builder.BundleError, match="cổng int8 chưa qua"):
        builder.require_passed_gate(passing_gate(passed=False), run, allow_failed=False)
    assert builder.require_passed_gate(passing_gate(passed=False), run, allow_failed=True)
    with pytest.raises(builder.BundleError, match="không phải"):
        builder.require_passed_gate(passing_gate("mtl-s2"), run, allow_failed=False)
    stale = {**passing_gate(), "normalizer_version": "0"}
    with pytest.raises(builder.BundleError, match="bản chuẩn hoá khác"):
        builder.require_passed_gate(stale, run, allow_failed=False)


def test_a_gate_from_an_older_normalizer_needs_the_invariance_evidence(monkeypatch, tmp_path):
    run, old_gate = tmp_path / "mtl-s1", {**passing_gate(), "normalizer_version": "0"}
    monkeypatch.setattr(builder, "normalizer_compatible", lambda found: False)
    with pytest.raises(builder.BundleError, match="bản chuẩn hoá khác"):
        builder.require_passed_gate(old_gate, run, allow_failed=False)
    monkeypatch.setattr(builder, "normalizer_compatible", lambda found: found == "0")
    assert builder.require_passed_gate(old_gate, run, allow_failed=False)["embedding_rows"] == 66119


@pytest.mark.data
def test_a_run_from_an_older_normalizer_needs_the_invariance_evidence(monkeypatch):
    history = {"normalizer_version": "0"}
    golden = GoldenPredictions(normalizer_version=NORMALIZER_VERSION, items=[])
    monkeypatch.setattr(builder, "normalizer_compatible", lambda found: False)
    with pytest.raises(builder.BundleError, match="không đồng nhất.*run huấn luyện '0'"):
        builder.assert_same_normalizer(history, background(), golden)
    monkeypatch.setattr(builder, "normalizer_compatible", lambda found: found == "0")
    builder.assert_same_normalizer(history, background(), golden)  # có bằng chứng thì qua


def evidence(**checks):
    return {
        "from": "1",
        "to": NORMALIZER_VERSION,
        "checks": {"sentences": 16175, "parquet_files_byte_identical": True, **checks},
    }


def test_the_card_says_where_the_test_numbers_come_from_when_the_normalizer_changed(monkeypatch):
    monkeypatch.setattr(builder, "find_invariance_evidence", lambda found: evidence())
    (note,) = builder.evaluation_notes("1")
    assert builder.PREVIOUS_BUNDLE in note and "chuẩn hoá bản 1" in note
    assert "16.175 câu" in note and "giống hệt từng byte" in note
    assert f"normalizer_v{NORMALIZER_VERSION}_invariance.json" in note
    assert "không được chấm lại" in note
    monkeypatch.setattr(
        builder,
        "find_invariance_evidence",
        lambda found: evidence(parquet_files_byte_identical=False),
    )
    assert "giống hệt từng byte" not in builder.evaluation_notes("1")[0]


def test_no_note_is_needed_when_the_normalizer_did_not_change_or_has_no_proof(monkeypatch):
    monkeypatch.setattr(builder, "find_invariance_evidence", lambda found: None)
    assert builder.evaluation_notes(NORMALIZER_VERSION) == []
    assert builder.evaluation_notes("1") == []
    assert builder.evaluation_notes(None) == []


# Mỗi ví dụ trong hai câu giới hạn về thông tin cá nhân phải đúng với luật `mask`.
MASKED_EXAMPLES = [
    "(028) 3812 3456",
    "(024)38686868",
    "0912 345 678",
    "abc.def@gmail.com",
    "cccd 001095012345",
    "mssv 18520001",
    "B20DCCN123",
    "github.com/x/y",
    "instagram.com/x",
    "@tenban",
    "Thầy Nam",
    "Cô giáo Lan",
    "Thầy giáo Minh",
    "giảng viên Trần Văn Bình",
    "hiệu trưởng Hùng",
    "trưởng khoa Lan",
    "trợ giảng Minh",
    "cố vấn Hà",
]
MISSED_EXAMPLES = ["thầy nam", "cô giáo lan", "Nguyễn Văn An", "mã 1712345", "b20dccn123"]


@pytest.mark.parametrize("text", MASKED_EXAMPLES)
def test_what_the_card_says_is_masked_really_is_masked(text):
    assert mask(text).types, text


@pytest.mark.parametrize("text", MISSED_EXAMPLES)
def test_what_the_card_says_is_missed_really_is_missed(text):
    assert mask(text).types == (), text


def test_the_pii_limitations_are_in_the_release_list_and_name_the_known_gaps():
    assert builder.PII_MASKED in builder.STATIC_LIMITATIONS
    assert builder.PII_MISSED in builder.STATIC_LIMITATIONS
    for fragment in ("B20DCCN123", "hiệu trưởng", "cô giáo", "(di động, bàn", "@tên tài khoản"):
        assert fragment in builder.PII_MASKED
    for fragment in ("tên viết thường", "EDUPULSE_NER=0", "7 chữ số", "mã môn"):
        assert fragment in builder.PII_MISSED
    assert not any("tên viết thường thì sót" in item for item in builder.STATIC_LIMITATIONS)


def test_different_normalizer_versions_are_refused():
    history = {"normalizer_version": NORMALIZER_VERSION}
    golden = GoldenPredictions(normalizer_version=NORMALIZER_VERSION, items=[])
    stale = background(version="0")
    with pytest.raises(builder.BundleError, match="không đồng nhất"):
        builder.assert_same_normalizer(history, stale, golden)


@pytest.mark.data
def test_guard_rejects_a_bundle_that_contains_a_source_sentence(tmp_path):
    train = load_processed("train")
    long_sentences = [s for s in train["sentence"] if len(s.split()) >= builder.GUARD_MIN_TOKENS]
    leaked = tmp_path / "leaky"
    leaked.mkdir()
    write_json(leaked / "model_card.json", {"quote": long_sentences})
    with pytest.raises(builder.BundleError, match="câu gốc"):
        builder.assert_no_source_sentences(leaked)


@pytest.mark.data
def test_guard_accepts_a_bundle_without_source_sentences(bundle, tmp_path):
    clean = tmp_path / "clean"
    shutil.copytree(bundle, clean)
    assert builder.assert_no_source_sentences(clean) == builder.GUARD_SAMPLES


# --- Guard quét hết (trước khi công khai) ------------------------------------------------------

SENTENCE = "thầy giảng bài rất dễ hiểu và luôn nhiệt tình giúp đỡ sinh viên ."
SEGMENTED = "thầy giảng_bài rất dễ hiểu và luôn nhiệt_tình giúp_đỡ sinh_viên ."
SHORT = "em rất hài lòng ."


@pytest.fixture()
def tiny_corpus(monkeypatch):
    """Nguồn câu thu nhỏ (cùng cột với dữ liệu thật); logic guard vẫn chạy thật trên nó."""
    frames = {
        "train": pd.DataFrame(
            {"sentence": [SENTENCE, SHORT], "text_rdr": [SEGMENTED, "em rất hài_lòng ."]}
        ),
        "validation": pd.DataFrame({"sentence": [], "text_rdr": []}),
        "test": pd.DataFrame({"sentence": [], "text_rdr": []}),
    }
    monkeypatch.setattr(builder, "load_processed", lambda split, allow_test=False: frames[split])


def test_the_sampled_guard_reads_markdown_as_well_as_json(tmp_path, tiny_corpus):
    (tmp_path / "README.md").write_text(f"Ví dụ: {SENTENCE}\n", encoding="utf-8")
    with pytest.raises(builder.BundleError, match="README.md"):
        builder.assert_no_source_sentences(tmp_path, samples=2)


def test_the_full_scan_catches_a_long_sentence_in_any_text_file(tmp_path, tiny_corpus):
    (tmp_path / "notes.md").write_text(SENTENCE, encoding="utf-8")
    with pytest.raises(builder.BundleError, match="notes.md"):
        builder.assert_no_source_sentences_anywhere(tmp_path)


def test_the_full_scan_catches_the_word_segmented_form(tmp_path, tiny_corpus):
    write_json(tmp_path / "golden_predictions.json", {"model_text": SEGMENTED})
    with pytest.raises(builder.BundleError, match="golden_predictions.json"):
        builder.assert_no_source_sentences_anywhere(tmp_path)


def test_the_full_scan_leaves_short_generic_phrases_alone(tmp_path, tiny_corpus):
    write_json(tmp_path / "golden_predictions.json", {"model_text": "em rất hài_lòng ."})
    (tmp_path / "README.md").write_text(f"Câu như '{SHORT}' rất hay gặp.", encoding="utf-8")
    assert builder.assert_no_source_sentences_anywhere(tmp_path) == 2  # hai dạng của câu dài


def test_the_full_scan_skips_binary_files_and_subfolders(tmp_path, tiny_corpus):
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline" / "x.json").write_text(SENTENCE, encoding="utf-8")
    (tmp_path / "model.int8.onnx").write_bytes(SENTENCE.encode("utf-8"))
    assert builder.assert_no_source_sentences_anywhere(tmp_path) == 2


@pytest.mark.data
def test_the_full_scan_catches_one_real_sentence_the_sample_would_miss(tmp_path):
    train = load_processed("train")
    sentence = next(s for s in train["sentence"] if len(s.split()) >= builder.FULL_SCAN_MIN_TOKENS)
    (tmp_path / "README.md").write_text(sentence, encoding="utf-8")
    with pytest.raises(builder.BundleError, match="README.md"):
        builder.assert_no_source_sentences_anywhere(tmp_path)


@pytest.mark.data
def test_the_full_scan_covers_the_whole_corpus_and_accepts_a_clean_bundle(bundle):
    assert builder.assert_no_source_sentences_anywhere(bundle) > 10_000
