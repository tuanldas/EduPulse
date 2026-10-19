"""Đọc file CSV/XLSX: mã hoá, dấu phân cách, giới hạn dòng, zip bom, đoán cột."""

import csv
import datetime
import io
import struct
import unicodedata
import zipfile

import openpyxl
import pytest

from edupulse.api import parsing
from edupulse.api.errors import ApiError
from edupulse.api.limits import MAX_CELL_CHARS, XLSX_MAX_UNCOMPRESSED_MB
from edupulse.config import MAX_ROWS, load_settings

HAND_WRITTEN = [
    "Thầy dạy rất nhiệt tình, dễ hiểu.",
    "Phòng máy nóng; quạt hỏng, máy chiếu mờ.",
    'Cô nói "rất nhanh" nên em không theo kịp.',
    "Slide ít ví dụ :) nhưng bài tập nhiều.",
    "Môn này lý thuyết nặng,\nthực hành ít.",
    "Cần thêm giáo trình tiếng Việt cho môn Cấu trúc dữ liệu & giải thuật.",
    "Wifi trong ký túc xá yếu, mong nhà trường nâng cấp.",
    "Em không có ý kiến gì thêm.",
    "Lịch thi dày đặc, mong dàn đều hơn!",
    "Giảng viên đến lớp đúng giờ và hỗ trợ tận tình ❤",
] * 6


@pytest.fixture(scope="module")
def sentences() -> list[str]:
    """Câu thật của tập val nếu máy có `make data`, không thì câu viết tay (đủ dấu phẩy, ngoặc)."""
    path = load_settings().data_dir / "raw" / "validation.parquet"
    if not path.is_file():
        return HAND_WRITTEN
    try:
        import pyarrow.parquet as pq
    except ImportError:  # môi trường chỉ có group `app`
        return HAND_WRITTEN

    column = pq.read_table(path, columns=["sentence"]).column(0).to_pylist()
    return [s for s in column[:300] if s.strip()] + HAND_WRITTEN


def csv_bytes(
    header: list[str],
    rows: list[list[str]],
    delimiter: str = ",",
    encoding: str = "utf-8",
    bom: bool = False,
    newline: str = "\r\n",
) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, delimiter=delimiter, lineterminator=newline)
    writer.writerow(header)
    writer.writerows(rows)
    data = out.getvalue().encode(encoding)
    return b"\xef\xbb\xbf" + data if bom else data


def to_cp1258(text: str) -> bytes:
    """Mã hoá cp1258 như Windows: chữ có dấu thanh tách thành chữ nền + dấu tổ hợp."""
    out = bytearray()
    for char in text:
        try:
            out += char.encode("cp1258")
            continue
        except UnicodeEncodeError:
            pass
        decomposed = unicodedata.normalize("NFD", char)
        base, tone = unicodedata.normalize("NFC", decomposed[:-1]), decomposed[-1]
        out += base.encode("cp1258") + tone.encode("cp1258")
    return bytes(out)


def xlsx_bytes(sheets: dict[str, list[list]], hidden: tuple[str, ...] = ()) -> bytes:
    book = openpyxl.Workbook()
    book.remove(book.active)
    for name, rows in sheets.items():
        sheet = book.create_sheet(name)
        for row in rows:
            sheet.append(row)
        if name in hidden:
            sheet.sheet_state = "hidden"
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def error_of(call) -> ApiError:
    with pytest.raises(ApiError) as caught:
        call()
    return caught.value


def code_of(call) -> str:
    return error_of(call).detail["code"]


# --- CSV: mã hoá --------------------------------------------------------------------------------


def test_utf8_with_bom(sentences):
    rows = [[s, "A"] for s in sentences]
    data = csv_bytes(["Góp ý", "Nhóm"], rows, bom=True)
    table = parsing.parse_upload("gopy.csv", data)
    assert table.encoding == "utf-8-sig"
    assert table.columns == ["Góp ý", "Nhóm"]
    assert [r[0] for r in table.rows] == [s.strip() for s in sentences]
    assert table.warnings == []


def test_utf8_without_bom(sentences):
    data = csv_bytes(["Góp ý"], [[s] for s in sentences])
    table = parsing.parse_upload("gopy.csv", data)
    assert table.encoding == "utf-8"
    assert len(table.rows) == len(sentences)


def test_cp1258_falls_back_with_a_warning_and_keeps_the_text(sentences):
    # Chỉ lấy câu mã hoá được: ký tự ngoài bảng Windows tiếng Việt (vd. emoji) không thể có.
    usable = []
    for sentence in sentences:
        try:
            to_cp1258(sentence)
        except UnicodeEncodeError:
            continue
        usable.append(sentence)
    assert len(usable) >= 20
    header = to_cp1258("Góp ý")
    body = b"\r\n".join(b'"' + to_cp1258(s).replace(b'"', b'""') + b'"' for s in usable)
    table = parsing.parse_upload("gopy.csv", header + b"\r\n" + body)
    assert table.encoding == "cp1258"
    assert [w.code for w in table.warnings] == ["encoding_fallback"]
    assert table.columns == ["Góp ý"]
    expected = [unicodedata.normalize("NFC", s.strip()) for s in usable]
    assert [r[0] for r in table.rows] == expected


@pytest.mark.parametrize("bom", [b"\xff\xfe", b"\xfe\xff"])
def test_utf16_is_rejected_with_a_helpful_message(bom: bytes):
    data = bom + "Góp ý\nthầy dạy hay\n".encode("utf-16-le")
    error = error_of(lambda: parsing.parse_upload("gopy.csv", data))
    assert error.status_code == 422 and error.detail["code"] == "decode_failed"
    assert "UTF-8" in error.detail["message"]


def test_binary_data_is_rejected():
    assert (
        code_of(lambda: parsing.parse_upload("a.csv", b"Gop y\n\x00\x01\x02\n")) == "decode_failed"
    )


# --- CSV: dấu phân cách -------------------------------------------------------------------------


@pytest.mark.parametrize("delimiter", [",", ";", "\t"])
def test_delimiter_is_detected_even_when_sentences_contain_commas(delimiter: str, sentences):
    rows = [[s, f"Nhóm {i % 3}", str(i)] for i, s in enumerate(sentences)]
    data = csv_bytes(["Góp ý", "Nhóm", "STT"], rows, delimiter=delimiter)
    table = parsing.parse_upload("gopy.csv", data)
    assert table.columns == ["Góp ý", "Nhóm", "STT"]
    assert len(table.rows) == len(sentences)
    assert table.rows[0][1:] == ["Nhóm 0", "0"]


def test_quoted_fields_may_contain_newlines_and_quotes():
    data = csv_bytes(["Góp ý", "Nhóm"], [['Cô nói "nhanh",\nkhó theo', "A"], ["Hay", "B"]])
    table = parsing.parse_upload("a.csv", data)
    assert table.rows == [['Cô nói "nhanh",\nkhó theo', "A"], ["Hay", "B"]]


def test_a_single_text_column_with_commas_is_not_split():
    lines = [
        "Góp ý",
        "Thầy dạy hay, dễ hiểu",
        "Phòng nóng, máy chiếu mờ",
        "Tốt, nhưng bài tập nhiều",
    ]
    table = parsing.parse_upload("a.csv", "\n".join(lines).encode("utf-8"))
    assert table.columns == ["Góp ý"]
    assert [r[0] for r in table.rows] == lines[1:]


def test_ragged_rows_are_padded_and_trimmed_to_the_header():
    data = b"Gop y,Nhom\nthay day hay\nmon nang,A,thua,thua\n"
    table = parsing.parse_upload("a.csv", data)
    assert table.rows == [["thay day hay", ""], ["mon nang", "A"]]


def test_cells_are_stripped_and_very_long_cells_are_cut():
    data = csv_bytes(["Góp ý"], [["  thầy dạy hay \t"], ["x" * (MAX_CELL_CHARS + 500)]])
    table = parsing.parse_upload("a.csv", data)
    assert table.rows[0] == ["thầy dạy hay"]
    assert len(table.rows[1][0]) == MAX_CELL_CHARS


# --- Dòng trống, tiêu đề, giới hạn -------------------------------------------------------------


def test_blank_rows_are_skipped_and_the_original_row_numbers_are_kept():
    data = b"Gop y,Nhom\nmot,A\n,\n \nhai,B\n\nba,C\n"
    table = parsing.parse_upload("a.csv", data)
    assert [r[0] for r in table.rows] == ["mot", "hai", "ba"]
    assert table.source_rows == [1, 4, 6]  # số dòng dữ liệu trong file gốc
    assert [(w.code, w.count) for w in table.warnings] == [("blank_rows_skipped", 3)]


def test_blank_and_duplicate_headers_are_renamed_with_a_warning():
    data = "Góp ý,,Góp ý,Nhóm\nhay,1,x,A\n".encode()
    table = parsing.parse_upload("a.csv", data)
    assert table.columns == ["Góp ý", "Cột 2", "Góp ý (2)", "Nhóm"]
    assert [(w.code, w.count) for w in table.warnings] == [("columns_renamed", 2)]


def test_exactly_the_row_limit_is_accepted_and_one_more_is_not():
    rows = [[f"câu số {i}"] for i in range(MAX_ROWS)]
    assert len(parsing.parse_upload("a.csv", csv_bytes(["Góp ý"], rows)).rows) == MAX_ROWS
    error = error_of(lambda: parsing.parse_upload("a.csv", csv_bytes(["Góp ý"], [*rows, ["x"]])))
    assert (error.status_code, error.detail["code"]) == (422, "too_many_rows")
    assert error.detail["params"] == {"n_rows": MAX_ROWS + 1, "max_rows": MAX_ROWS}


def test_the_real_row_count_is_reported_for_much_larger_files():
    rows = [[f"câu số {i}"] for i in range(5000)]
    error = error_of(lambda: parsing.parse_upload("a.csv", csv_bytes(["Góp ý"], rows)))
    assert error.detail["params"]["n_rows"] == 5000


@pytest.mark.parametrize("data", [b"", b"   \n\n  ", b"Gop y,Nhom\n", b"Gop y\n\n\n,\n"])
def test_files_without_data_rows_have_their_own_code(data: bytes):
    assert code_of(lambda: parsing.parse_upload("a.csv", data)) == "no_rows"


# --- Loại file -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a.txt", "a.xls", "a", "a.csv.exe", "a.pdf", ""])
def test_other_extensions_are_unsupported(name: str):
    error = error_of(lambda: parsing.parse_upload(name, b"Gop y\nhay\n"))
    assert (error.status_code, error.detail["code"]) == (415, "unsupported_file_type")


def test_extension_is_matched_case_insensitively():
    assert parsing.parse_upload("GOPY.CSV", b"Gop y\nhay\n").columns == ["Gop y"]


def test_content_must_match_the_extension():
    book = xlsx_bytes({"S": [["Góp ý"], ["hay"]]})
    assert code_of(lambda: parsing.parse_upload("a.csv", book)) == "unsupported_file_type"
    assert (
        code_of(lambda: parsing.parse_upload("a.xlsx", b"Gop y\nhay\n")) == "unsupported_file_type"
    )
    ole = b"\xd0\xcf\x11\xe0" + b"\x00" * 64
    assert code_of(lambda: parsing.parse_upload("a.csv", ole)) == "unsupported_file_type"


# --- XLSX ---------------------------------------------------------------------------------------


def test_xlsx_values_become_strings(sentences):
    data = xlsx_bytes(
        {
            "Phản hồi": [
                ["Góp ý", "Điểm", "Ngày", "Nhóm"],
                [sentences[0], 5, datetime.datetime(2026, 10, 1, 8, 30), "A"],
                [sentences[1], 4.5, datetime.date(2026, 10, 2), None],
            ]
        }
    )
    table = parsing.parse_upload("a.xlsx", data)
    assert table.encoding is None and table.warnings == []
    assert table.columns == ["Góp ý", "Điểm", "Ngày", "Nhóm"]
    assert table.rows[0] == [sentences[0].strip(), "5", "2026-10-01T08:30:00", "A"]
    assert table.rows[1] == [sentences[1].strip(), "4.5", "2026-10-02", ""]


def test_xlsx_reads_the_first_visible_sheet_and_warns_about_the_rest():
    data = xlsx_bytes(
        {"Ẩn": [["x"], ["y"]], "Dữ liệu": [["Góp ý"], ["hay"]], "Khác": [["z"], ["w"]]},
        hidden=("Ẩn",),
    )
    table = parsing.parse_upload("a.xlsx", data)
    assert table.columns == ["Góp ý"] and table.rows == [["hay"]]
    assert [(w.code, w.count) for w in table.warnings] == [("other_sheets_ignored", 1)]


def test_xlsx_skips_blank_rows_like_csv():
    data = xlsx_bytes({"S": [["Góp ý"], ["một"], [None], ["hai"]]})
    table = parsing.parse_upload("a.xlsx", data)
    assert [r[0] for r in table.rows] == ["một", "hai"]
    assert table.source_rows == [1, 3]


def test_xlsx_row_limit():
    rows = [["Góp ý"], *([f"câu {i}"] for i in range(MAX_ROWS + 1))]
    error = error_of(lambda: parsing.parse_upload("a.xlsx", xlsx_bytes({"S": rows})))
    assert (
        error.detail["code"] == "too_many_rows" and error.detail["params"]["n_rows"] == MAX_ROWS + 1
    )


def test_corrupt_xlsx_is_decode_failed():
    data = xlsx_bytes({"S": [["Góp ý"], ["hay"]]})
    assert (
        code_of(lambda: parsing.parse_upload("a.xlsx", data[: len(data) // 2])) == "decode_failed"
    )
    assert (
        code_of(lambda: parsing.parse_upload("a.xlsx", b"PK\x03\x04" + b"junk" * 50))
        == "decode_failed"
    )


# --- Zip bom ------------------------------------------------------------------------------------


def zip_with(member_bytes: int, name: str = "xl/worksheets/sheet1.xml") -> bytes:
    """Zip có một tệp con toàn số 0 (nén rất tốt): vài trăm KB nở ra `member_bytes` byte."""
    out = io.BytesIO()
    block = b"\x00" * (1 << 20)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        with archive.open(name, "w") as stream:
            for _ in range(member_bytes >> 20):
                stream.write(block)
    return out.getvalue()


def test_honest_zip_bomb_is_blocked_before_the_xlsx_parser_sees_it():
    bomb = zip_with((XLSX_MAX_UNCOMPRESSED_MB + 20) << 20)
    assert len(bomb) < 2 << 20  # nhỏ hơn nhiều so với giới hạn upload
    error = error_of(lambda: parsing.parse_upload("a.xlsx", bomb))
    assert (error.status_code, error.detail["code"]) == (422, "decode_failed")
    assert str(XLSX_MAX_UNCOMPRESSED_MB) in error.detail["message"]


def test_a_zip_that_lies_about_its_uncompressed_size_is_blocked_too():
    bomb = zip_with((XLSX_MAX_UNCOMPRESSED_MB + 20) << 20)
    # Khai giả cỡ trong cả header cục bộ lẫn danh mục: chỉ byte thực khi giải nén mới nói sự thật.
    forged = bytearray(bomb)
    for signature, field in ((b"PK\x03\x04", 22), (b"PK\x01\x02", 24)):
        start = bomb.index(signature) if signature == b"PK\x03\x04" else bomb.rindex(signature)
        forged[start + field : start + field + 4] = struct.pack("<L", 100)
    with zipfile.ZipFile(io.BytesIO(bytes(forged))) as archive:
        assert archive.infolist()[0].file_size != (XLSX_MAX_UNCOMPRESSED_MB + 20) << 20
    assert code_of(lambda: parsing.parse_upload("a.xlsx", bytes(forged))) == "decode_failed"


def test_many_members_are_rejected():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:
        for i in range(2500):
            archive.writestr(f"f{i}.xml", b"<x/>")
    assert code_of(lambda: parsing.parse_upload("a.xlsx", out.getvalue())) == "decode_failed"


def test_size_check_counts_actual_bytes_for_deflate_streams():
    payload = b"abc" * 100_000
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("a.xml", payload)
    members = zipfile.ZipFile(io.BytesIO(out.getvalue())).infolist()
    assert parsing._inflated_size(out.getvalue(), members[0], 10 << 20) == len(payload)


# --- Đoán cột -----------------------------------------------------------------------------------


def table_of(header: list[str], rows: list[list[str]]) -> parsing.ParsedTable:
    return parsing.parse_upload("a.csv", csv_bytes(header, rows))


def test_text_column_is_found_by_name_first(sentences):
    rows = [[str(i), sentences[i], "Môn " + str(i % 3)] for i in range(40)]
    assert parsing.suggest_text_column(table_of(["STT", "Góp ý", "Môn học"], rows)) == "Góp ý"
    assert parsing.suggest_text_column(table_of(["STT", "gop y", "Môn học"], rows)) == "gop y"
    assert parsing.suggest_text_column(table_of(["ID", "Feedback", "X"], rows)) == "Feedback"
    assert parsing.suggest_text_column(table_of(["STT", "Ý kiến sinh viên", "X"], rows)) == (
        "Ý kiến sinh viên"
    )


def test_text_column_falls_back_to_the_longest_texty_column(sentences):
    rows = [[str(i), sentences[i], f"M{i % 3}", str(i * 3.5)] for i in range(40)]
    assert parsing.suggest_text_column(table_of(["a", "b", "c", "d"], rows)) == "b"


def test_a_named_column_that_only_holds_numbers_is_not_chosen():
    rows = [[str(i), f"{i}.5", "thầy dạy rất hay và nhiệt tình ở lớp"] for i in range(20)]
    assert parsing.suggest_text_column(table_of(["Góp ý", "Điểm", "Khác"], rows)) == "Khác"


def test_no_text_column_when_everything_is_numeric_or_short():
    rows = [[str(i), f"{i}.5", "x"] for i in range(20)]
    assert parsing.suggest_text_column(table_of(["a", "b", "c"], rows)) is None


def test_group_column_needs_a_group_like_name_and_a_sensible_number_of_values():
    rows = [
        [f"câu dài số {i} để làm văn bản", f"Nhóm {i % 4}", str(i % 2), f"ID{i}"] for i in range(60)
    ]
    table = table_of(["Góp ý", "Nhóm", "Lớp", "Mã"], rows)
    assert parsing.suggest_group_column(table, "Góp ý") == "Nhóm"
    many = [[f"câu {i}", f"Nhóm {i}"] for i in range(60)]  # 60 giá trị: quá nhiều để làm nhóm
    assert parsing.suggest_group_column(table_of(["Góp ý", "Nhóm"], many), "Góp ý") is None
    one = [[f"câu {i}", "Nhóm A"] for i in range(10)]  # chỉ một giá trị
    assert parsing.suggest_group_column(table_of(["Góp ý", "Nhóm"], one), "Góp ý") is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Nhóm (minh hoạ)", True),
        ("NHÓM (MINH HỌA)", True),
        ("nhom minh hoa", True),
        ("Nhóm", False),
        ("Môn học", False),
    ],
)
def test_illustrative_group_columns(name: str, expected: bool):
    assert parsing.is_illustrative_name(name) is expected
