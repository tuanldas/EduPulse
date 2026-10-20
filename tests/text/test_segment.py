"""Bộ tách từ RDRSegmenter: thư mục lỗi, cwd, đa luồng. Cần JRE (marker `java`)."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from edupulse.text.segment import Segmenter, SegmenterError, get_segmenter

# --- Không cần khởi động JVM ----------------------------------------------------------------


def test_missing_model_files_fail_with_clear_message(tmp_path):
    with pytest.raises(SegmenterError, match="thiếu .*VnCoreNLP-1.2.jar"):
        Segmenter(tmp_path)


def test_invalid_heap_option_is_rejected_before_anything_starts(tmp_path):
    with pytest.raises(SegmenterError, match="-Xmx"):
        Segmenter(tmp_path, "512m")


# --- Cần JVM ----------------------------------------------------------------------------------


@pytest.mark.java
def test_cwd_is_restored_after_jvm_start(tmp_path):
    # py_vncorenlp chdir vĩnh viễn; kiểm trong process riêng vì JVM chỉ khởi động được một lần.
    code = (
        "import os\n"
        "from edupulse.text.segment import get_segmenter\n"
        "before = os.getcwd()\n"
        "get_segmenter()\n"
        "print('CWD_UNCHANGED' if os.getcwd() == before else 'CWD_CHANGED')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "CWD_UNCHANGED" in result.stdout


@pytest.mark.java
def test_segments_compound_words_with_underscore():
    assert get_segmenter().segment("thầy dạy rất nhiệt tình") == "thầy dạy rất nhiệt_tình"


@pytest.mark.java
def test_blank_input_gives_empty_string():
    segmenter = get_segmenter()
    assert segmenter.segment("") == ""
    assert segmenter.segment("   ") == ""


@pytest.mark.java
def test_multiple_sentences_are_joined_by_space():
    out = get_segmenter().segment("Học tốt. Thầy dạy hay.")
    assert out == "Học tốt . Thầy dạy hay ."


@pytest.mark.java
def test_dataset_special_tokens_survive_segmentation():
    out = get_segmenter().segment(
        "thầy wzjwz208 dạy hay colonsmile . ví dụ doubledot 11doubledot55 vdotv ."
    )
    tokens = out.split()
    for token in ("wzjwz208", "colonsmile", "doubledot", "11doubledot55", "vdotv"):
        assert token in tokens, out


@pytest.mark.java
def test_singleton_is_shared():
    assert get_segmenter() is get_segmenter()


@pytest.mark.java
def test_four_threads_match_sequential_results():
    segmenter = get_segmenter()
    base = [
        "bài giảng của thầy rõ ràng , ví dụ sát thực tế .",
        "phòng máy hay hỏng , máy chiếu mờ , wifi yếu .",
        "chương trình học nặng lý thuyết , cần thêm thực hành .",
        "em xin thầy thêm vài bài luyện tập ở nhà colonsmile .",
        "học phí 2 triệu 5 mỗi tín chỉ , giờ giấc không hợp lý .",
    ]
    sentences = [f"{base[i % len(base)]} lớp {i}" for i in range(200)]
    sequential = [segmenter.segment(s) for s in sentences]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(segmenter.segment, sentences))
    assert parallel == sequential
