"""Bằng chứng bất biến giữa hai bản chuẩn hoá và cầu nối `normalizer_compatible`."""

import copy
from pathlib import Path

import pandas as pd
import pytest

from edupulse.text import NORMALIZER_VERSION
from ml import common
from ml import normalizer_invariance as invariance
from ml.common import SPLITS, normalizer_compatible, sha256_file, write_json

OLD = "0"  # phiên bản cũ giả định; luôn khác bản của mã
REVISION = "r" * 40


def sentences(split: str) -> list[str]:
    return [f"câu {split} số {i} nói về bài giảng" for i in range(3)]


def write_processed(
    directory: Path, *, version: str, revision: str = REVISION, model_text_suffix: str = ""
) -> None:
    """Dữ liệu đã xử lý thu nhỏ đúng cấu trúc của `ml.preprocess` (parquet + MANIFEST)."""
    directory.mkdir(parents=True)
    files = {}
    for split in SPLITS:
        texts = sentences(split)
        frame = pd.DataFrame(
            {
                "sentence": texts,
                "sentiment": [0, 1, 2],
                "topic": [0, 1, 3],
                "text_rdr": [text.replace(" ", "_", 1) + model_text_suffix for text in texts],
                "n_words": [6, 6, 6],
                "no_diacritics": [False, False, False],
            }
        )
        path = directory / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        files[split] = {"file": path.name, "sha256": sha256_file(path), "rows": len(frame)}
    manifest = {
        "normalizer_version": version,
        "dataset_revision": revision,
        "raw_sha256": {split: "a" * 64 for split in SPLITS},
        "files": files,
        "stats": {},
    }
    write_json(directory / "MANIFEST.json", manifest)


@pytest.fixture()
def pair(tmp_path):
    write_processed(tmp_path / "old", version=OLD)
    write_processed(tmp_path / "new", version=NORMALIZER_VERSION)
    return invariance.read_processed(tmp_path / "old"), invariance.read_processed(tmp_path / "new")


def make_evidence(pair):
    old, new = pair
    return invariance.compare(old, new, old_version=OLD, new_version=NORMALIZER_VERSION)


# --- So sánh hai bản dữ liệu ---------------------------------------------------------------------


def test_identical_processed_data_gives_evidence_tied_to_revision_and_hashes(pair):
    old, new = pair
    evidence = make_evidence(pair)
    assert evidence["identical"] is True
    assert (evidence["from"], evidence["to"]) == (OLD, NORMALIZER_VERSION)
    assert evidence["dataset"]["revision"] == REVISION
    assert evidence["checks"]["sentences"] == 9
    assert evidence["checks"]["rows_with_different_model_text"] == 0
    assert evidence["checks"]["rows_with_different_columns"] == 0
    assert evidence["checks"]["sentences_with_capital_letters"] == 0
    assert evidence["processed"]["from"]["files"]["test"] == old.manifest["files"]["test"]["sha256"]
    assert evidence["processed"]["to"]["files"]["test"] == new.manifest["files"]["test"]["sha256"]
    assert evidence["processed"]["to"]["manifest_sha256"] == new.manifest_sha256


def test_one_different_model_text_means_no_evidence(tmp_path):
    write_processed(tmp_path / "old", version=OLD)
    write_processed(tmp_path / "new", version=NORMALIZER_VERSION, model_text_suffix=" thêm")
    with pytest.raises(invariance.InvarianceError, match="9 câu có model_text khác"):
        invariance.compare(
            invariance.read_processed(tmp_path / "old"),
            invariance.read_processed(tmp_path / "new"),
            old_version=OLD,
            new_version=NORMALIZER_VERSION,
        )


def test_capital_letters_and_pii_in_the_data_are_counted(tmp_path):
    for name, version in (("old", OLD), ("new", NORMALIZER_VERSION)):
        write_processed(tmp_path / name, version=version)
        for split in SPLITS:  # đưa chữ hoa và một số điện thoại vào cột câu của cả hai bản
            path = tmp_path / name / f"{split}.parquet"
            frame = pd.read_parquet(path)
            frame.loc[0, "sentence"] = "Thầy Nam gọi 0912345678"
            frame.to_parquet(path, index=False)
            manifest = common.read_json(tmp_path / name / "MANIFEST.json")
            manifest["files"][split]["sha256"] = sha256_file(path)
            write_json(tmp_path / name / "MANIFEST.json", manifest)
    evidence = invariance.compare(
        invariance.read_processed(tmp_path / "old"),
        invariance.read_processed(tmp_path / "new"),
        old_version=OLD,
        new_version=NORMALIZER_VERSION,
    )
    assert evidence["checks"]["sentences_with_capital_letters"] == 3
    assert evidence["checks"]["sentences_with_pii_under_new_rules"] == 3


@pytest.mark.parametrize(
    ("old_version", "new_version", "revision", "fragment"),
    [
        ("9", NORMALIZER_VERSION, REVISION, "dữ liệu cũ dựng bằng bản chuẩn hoá"),
        (OLD, "9", REVISION, "dữ liệu mới dựng bằng bản chuẩn hoá"),
        (OLD, NORMALIZER_VERSION, "x" * 40, "dataset_revision"),
    ],
)
def test_data_of_the_wrong_version_or_revision_cannot_be_compared(
    tmp_path, old_version, new_version, revision, fragment
):
    write_processed(tmp_path / "old", version=OLD)
    write_processed(tmp_path / "new", version=NORMALIZER_VERSION, revision=revision)
    old = invariance.read_processed(tmp_path / "old")
    new = invariance.read_processed(tmp_path / "new")
    with pytest.raises(invariance.InvarianceError, match=fragment):
        invariance.compare(old, new, old_version=old_version, new_version=new_version)


def test_a_parquet_file_that_does_not_match_its_manifest_is_refused(tmp_path):
    write_processed(tmp_path / "old", version=OLD)
    path = tmp_path / "old" / "test.parquet"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(invariance.InvarianceError, match="sha256"):
        invariance.read_processed(tmp_path / "old")


def test_missing_manifest_is_reported(tmp_path):
    with pytest.raises(invariance.InvarianceError, match="MANIFEST"):
        invariance.read_processed(tmp_path)


# --- Cầu nối normalizer_compatible ---------------------------------------------------------------


@pytest.fixture()
def evidence_dir(tmp_path, pair):
    directory = tmp_path / "metrics"
    write_json(directory / f"normalizer_v{NORMALIZER_VERSION}_invariance.json", make_evidence(pair))
    return directory


def compatible(found, evidence_dir, pair):
    return normalizer_compatible(found, evidence_dir=evidence_dir, manifest=pair[1].manifest)


def test_the_current_version_is_always_compatible(tmp_path, pair):
    assert compatible(NORMALIZER_VERSION, tmp_path / "none", pair)


def test_an_older_version_is_compatible_only_with_valid_evidence(evidence_dir, tmp_path, pair):
    assert compatible(OLD, evidence_dir, pair)
    assert not compatible(OLD, tmp_path / "none", pair)  # chưa có bằng chứng
    assert not compatible("7", evidence_dir, pair)  # bằng chứng không nói về bản 7
    assert not compatible(None, evidence_dir, pair)
    assert not compatible("", evidence_dir, pair)


def evidence_file(directory: Path) -> Path:
    return directory / f"normalizer_v{NORMALIZER_VERSION}_invariance.json"


@pytest.mark.parametrize(
    "tamper",
    [
        lambda e: e.update(identical=False),
        lambda e: e["checks"].update(rows_with_different_model_text=1),
        lambda e: e["checks"].update(rows_with_different_columns=2),
        lambda e: e["checks"].update(sentences=8),
        lambda e: e["dataset"].update(revision="khac"),
        lambda e: e["processed"]["to"]["files"].update(test="0" * 64),
        lambda e: e.update(to="99"),
        lambda e: e.update(schema_version=2),
        lambda e: e.pop("checks"),
    ],
    ids=[
        "not-identical",
        "different-text",
        "different-columns",
        "row-count",
        "dataset-revision",
        "processed-hash",
        "other-target",
        "schema",
        "missing-key",
    ],
)
def test_evidence_that_does_not_fit_the_current_data_is_ignored(evidence_dir, pair, tamper):
    path = evidence_file(evidence_dir)
    evidence = copy.deepcopy(common.read_json(path))
    tamper(evidence)
    write_json(path, evidence)
    assert not compatible(OLD, evidence_dir, pair)


def test_corrupt_evidence_and_missing_data_are_treated_as_no_evidence(
    evidence_dir, pair, monkeypatch
):
    good = common.read_json(evidence_file(evidence_dir))
    evidence_file(evidence_dir).write_text("{không phải json", encoding="utf-8")
    assert not compatible(OLD, evidence_dir, pair)

    write_json(evidence_file(evidence_dir), good)
    assert compatible(OLD, evidence_dir, pair)

    def no_manifest():
        raise FileNotFoundError("chưa có MANIFEST")

    monkeypatch.setattr(common, "processed_manifest", no_manifest)
    assert not normalizer_compatible(OLD, evidence_dir=evidence_dir)


# --- Dòng lệnh -----------------------------------------------------------------------------------


def test_main_writes_the_evidence_only_when_nothing_differs(tmp_path, monkeypatch, capsys):
    write_processed(tmp_path / "old", version=OLD)
    write_processed(tmp_path / "current", version=NORMALIZER_VERSION)
    monkeypatch.setattr(invariance, "processed_dir", lambda: tmp_path / "current")
    out = tmp_path / "out" / "evidence.json"
    argv = ["--old-processed", str(tmp_path / "old"), "--from", OLD, "--out", str(out)]
    assert invariance.main(argv) == 0
    assert common.read_json(out)["identical"] is True
    assert "giống hệt trên 9 câu" in capsys.readouterr().out

    write_processed(tmp_path / "changed", version=OLD, model_text_suffix=" khác")
    out.unlink()
    argv = ["--old-processed", str(tmp_path / "changed"), "--from", OLD, "--out", str(out)]
    assert invariance.main(argv) == 1
    assert not out.exists()
    assert "KHÔNG bất biến" in capsys.readouterr().err


@pytest.mark.data
def test_the_committed_evidence_matches_the_processed_data_on_disk():
    # Bằng chứng nằm trong git, dữ liệu nằm ở $EDUPULSE_HOME: lệch nhau thì cầu nối tự đóng.
    assert normalizer_compatible("1")
