"""README của repo Hugging Face: tiếng Việt, giấy phép, trích dẫn, số test và giới hạn."""

import re

import pytest
from bundle_factory import manifest_for, release_card_json

from edupulse.bundle_schema import ModelCard
from edupulse.config import BUDGETS
from edupulse.labels import LABELS_VI
from ml import build_artifacts as builder
from ml import hub_readme

REPO = "owner/edupulse-phobert-mtl"
SOURCE = "https://github.com/owner/edupulse"


@pytest.fixture()
def parts():
    files = {name: f"{i:064x}" for i, name in enumerate(builder.BUNDLE_FILES, start=1)}
    manifest = manifest_for(files, stage="release", git_commit="6c19cdb3cf")
    return manifest, ModelCard.model_validate(release_card_json())


def render(parts, **kwargs):
    manifest, card = parts
    return hub_readme.render_readme(manifest, card, repo_id=REPO, **kwargs)


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def test_front_matter_is_valid_yaml_with_the_hub_metadata(parts):
    yaml = pytest.importorskip("yaml")
    match = re.match(r"---\n(.*?)\n---\n", render(parts), re.DOTALL)
    assert match, "README phải mở đầu bằng khối siêu dữ liệu YAML"
    meta = yaml.safe_load(match.group(1))
    assert meta["license"] == "agpl-3.0"
    assert meta["language"] == ["vi"]
    assert meta["base_model"] == "vinai/phobert-base-v2"
    assert meta["datasets"] == ["uitnlp/vietnamese_students_feedback"]
    assert meta["pipeline_tag"] == "text-classification"
    assert meta["library_name"] == "onnx"
    assert {"vietnamese", "phobert", "onnx"} <= set(meta["tags"])


def test_the_readme_is_vietnamese_and_has_every_required_section(parts):
    text = render(parts)
    for heading in (
        "## Mục đích sử dụng",
        "## Kết quả trên tập test (3.166 câu)",
        "### Từng lớp",
        "### So sánh với các mô hình cơ sở",
        "### Đa nhiệm so với đơn nhiệm",
        "### Ngân sách duyệt (trên test)",
        "### Kiểm thử hành vi",
        "## Giới hạn đã biết",
        "## Dữ liệu, quyền riêng tư và đạo đức",
        "## Cách dùng",
        "## Tệp trong repo",
        "## Giấy phép",
        "## Trích dẫn",
    ):
        assert heading in text


def test_test_numbers_are_printed_with_a_decimal_comma(parts):
    results = section(render(parts), "## Kết quả trên tập test", "## Giới hạn đã biết")
    assert "| Cảm xúc | 94,0 | 82,1 | 90,0 | 0,014 |" in results
    assert "| Chủ đề | 90,0 | 81,3 | 90,0 | 0,023 |" in results
    for ascii_decimal in ("94.0", "82.1", "81.3", "0.014", "0.023"):
        assert ascii_decimal not in results


def test_per_class_rows_use_the_vietnamese_label_names(parts):
    text = render(parts)
    for names in LABELS_VI.values():
        for name in names.values():
            assert f"| {name} |" in text


def test_every_review_budget_is_listed_and_the_default_is_marked(parts):
    text = render(parts)
    assert "| 10% (mặc định) |" in text
    for budget in BUDGETS:
        assert f"| {budget}%" in text


def test_limitations_and_notes_are_copied_verbatim(parts):
    _, card = parts
    text = render(parts)
    for item in (*card.limitations, card.notes.x86, card.notes.demo):
        assert f"- {item}" in text


def test_license_citations_and_the_academic_use_notice_are_present(parts):
    _, card = parts
    text = render(parts)
    for needle in (
        "**AGPL-3.0**",
        "nguyen2018uitvsfc",
        "nguyen2020phobert",
        "10.1109/KSE.2018.8573337",
        "10.18653/v1/2020.findings-emnlp.92",
        "phi thương mại",
        "UIT-VSFC",
        "PhoBERT",
        "Repo này không chứa câu gốc của UIT-VSFC",
    ):
        assert needle in text
    assert f"**Không dùng cho:** {card.out_of_scope}" in text


def test_a_source_link_appears_only_when_one_is_given(parts):
    with_link = render(parts, source_url=SOURCE)
    assert f"| Mã nguồn | {SOURCE} |" in with_link
    assert f"AGPL-3.0, tại {SOURCE}" in with_link
    without = render(parts)
    assert "| Mã nguồn |" not in without
    assert "github.com" not in without
    assert "None" not in without


def test_the_checksum_block_lists_every_file_with_its_digest(parts):
    manifest, _ = parts
    text = render(parts)
    for name, digest in manifest.files.items():
        assert f"{digest}  {name}" in text


def test_every_file_of_the_bundle_has_a_description(parts):
    manifest, _ = parts
    assert set(builder.BUNDLE_FILES) <= set(hub_readme.FILE_NOTES)
    text = render(parts)
    for name in (*manifest.files, "manifest.json", "README.md"):
        assert f"| `{name}` |" in text


def test_a_file_without_a_description_is_refused(parts):
    manifest, card = parts
    manifest = manifest.model_copy(update={"files": {**manifest.files, "extra.bin": "a" * 64}})
    with pytest.raises(ValueError, match="extra.bin"):
        hub_readme.render_readme(manifest, card, repo_id=REPO)


def test_the_usage_snippet_compiles_and_follows_the_manifest(parts):
    manifest, _ = parts
    match = re.search(r"```python\n(.*?)\n```", render(parts), re.DOTALL)
    assert match
    snippet = match.group(1)
    compile(snippet, "<readme>", "exec")
    assert f"max_length={manifest.max_len}" in snippet
    for name in (*manifest.onnx.input_names, *manifest.onnx.output_names):
        assert name in snippet
    assert f"EDUPULSE_HF_REPO={REPO}" in render(parts)


def test_tables_are_rectangular(parts):
    blocks = re.findall(r"(?:^\|.*\|\n)+", render(parts), re.MULTILINE)
    assert len(blocks) >= 8
    for block in blocks:
        widths = {len(re.split(r"(?<!\\)\|", line.strip())) for line in block.splitlines()}
        assert len(widths) == 1, block


def test_a_pipe_inside_a_cell_is_escaped():
    assert hub_readme.table(["a"], [["x|y"]]) == "| a |\n| --- |\n| x\\|y |"


def test_rendering_is_deterministic(parts):
    assert render(parts, source_url=SOURCE) == render(parts, source_url=SOURCE)


@pytest.mark.data
def test_the_rendered_readme_contains_no_source_sentence(parts, tmp_path):
    (tmp_path / "README.md").write_text(render(parts, source_url=SOURCE), encoding="utf-8")
    assert builder.assert_no_source_sentences_anywhere(tmp_path) > 10_000
