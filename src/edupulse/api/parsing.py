"""Đọc file CSV/XLSX do người dùng tải lên thành bảng chuỗi, và đoán cột văn bản / cột nhóm.

Quy ước:
- Dòng đầu (không trống) là tiêu đề. Dòng trống bị bỏ; số thứ tự dòng gốc được giữ lại để người
  dùng đối chiếu với file của họ.
- CSV: UTF-8 (có hoặc không BOM), rơi về cp1258 kèm cảnh báo; dấu phân cách `,` `;` hoặc TAB chọn
  theo độ nhất quán số cột trên những dòng đầu (Excel locale Việt hay xuất `;`, và câu tiếng Việt
  có dấu phẩy nên `csv.Sniffer` hay đoán sai).
- XLSX: sheet hiển thị đầu tiên, đọc bằng `python-calamine`. File zip được giải nén THỬ theo luồng
  với trần dung lượng thật (không tin `file_size` trong header, có thể khai giả) trước khi mở, rồi
  các sheet được quét lấy dòng/cột xa nhất từ thuộc tính `r` của `<row>` và `<c>` (không tin
  `<dimension>`), vì calamine cấp phát ma trận đặc theo ô xa nhất.
- Quá `MAX_ROWS` dòng dữ liệu thì từ chối, kèm số dòng thật để giao diện báo.

Lỗi là `ApiError` với mã ổn định; thông điệp không chứa nội dung file hay tên file.
"""

from __future__ import annotations

import csv
import datetime
import io
import posixpath
import re
import struct
import unicodedata
import zipfile
import zlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from xml.parsers import expat

from edupulse.api.errors import ApiError
from edupulse.api.limits import (
    MAX_CELL_CHARS,
    MAX_COLUMN_CHARS,
    MAX_COLUMNS,
    MAX_GROUPS,
    XLSX_MAX_MEMBERS,
    XLSX_MAX_SHEET_CELLS,
    XLSX_MAX_SHEET_ROWS,
    XLSX_MAX_UNCOMPRESSED_MB,
)
from edupulse.config import MAX_ROWS
from edupulse.demo_data import strip_diacritics

DELIMITERS = (",", ";", "\t")
SNIFF_ROWS = 100  # số dòng đầu (không trống) dùng để chọn dấu phân cách
SNIFF_RAW_ROWS = (
    2000  # trần số dòng thô được xem, kể cả dòng trống: file toàn dòng trống không quét hết
)
# File 10MB toàn dấu phân cách không phải góp ý (cả 2.000 dòng x 200 cột chỉ ~400.000): dừng sớm vì
# csv.reader dựng nguyên một dòng 10 triệu ô (~155MB) trước khi ta được xem nó.
MAX_DELIMITERS = 2_000_000
# Dòng trống tối đa trước khi từ chối: bằng số dòng tối đa của một sheet XLSX (cùng một trần, vì
# sheet đã tô định dạng tới hàng nghìn dòng vẫn là file hợp lệ), đủ nhỏ để quét CSV không tốn CPU.
MAX_BLANK_ROWS = XLSX_MAX_SHEET_ROWS
ZIP_MAGIC = b"PK\x03\x04"
UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
_XLSX_CHUNK = 1 << 16

_WARNING_MESSAGES = {
    "encoding_fallback": (
        "File không phải UTF-8 nên được đọc theo bảng mã Windows tiếng Việt (cp1258); "
        "hãy kiểm tra dấu trong phần xem trước."
    ),
    "other_sheets_ignored": "File có nhiều sheet; chỉ đọc sheet đầu tiên.",
}


@dataclass(frozen=True, slots=True)
class ParseWarning:
    code: str  # trùng `schemas.BatchWarningCode`
    message: str
    count: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedTable:
    columns: list[str]
    rows: list[list[str]]  # mỗi dòng đủ `len(columns)` ô, đã bỏ khoảng trắng hai đầu
    source_rows: list[int]  # số thứ tự dòng dữ liệu trong file gốc (từ 1) của từng dòng giữ lại
    encoding: str | None  # "utf-8" | "utf-8-sig" | "cp1258"; None với XLSX
    warnings: list[ParseWarning]


def _error(code: str, message: str, status: int = 422, **params: int) -> ApiError:
    return ApiError(status, code, message, params=params or None)


def _decode_failed(why: str) -> ApiError:
    return _error("decode_failed", f"Không đọc được file: {why}")


# --- Nhận dạng loại file và giải mã -------------------------------------------------------------


def detect_kind(filename: str, data: bytes) -> str:
    """ "csv" hoặc "xlsx" theo phần mở rộng, đối chiếu với nội dung; sai thì 415."""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    is_zip = data.startswith(ZIP_MAGIC)
    if extension == "xlsx":
        if not is_zip:
            raise _error("unsupported_file_type", "File .xlsx không phải XLSX hợp lệ.", 415)
        return "xlsx"
    if extension == "csv":
        if is_zip or data.startswith(b"\xd0\xcf\x11\xe0"):
            raise _error("unsupported_file_type", "File .csv này không phải văn bản CSV.", 415)
        return "csv"
    raise _error("unsupported_file_type", "Chỉ nhận file .csv hoặc .xlsx.", 415)


def decode_csv(data: bytes) -> tuple[str, str, list[ParseWarning]]:
    """Văn bản, tên bảng mã và cảnh báo. UTF-16 và file nhị phân bị từ chối."""
    if data.startswith(UTF16_BOMS):
        raise _decode_failed(
            "lưu lại dạng CSV UTF-8 (Excel: «CSV UTF-8 (phân cách bằng dấu phẩy)»)"
        )
    if b"\x00" in data:
        raise _decode_failed("file chứa dữ liệu nhị phân")
    if data.startswith(b"\xef\xbb\xbf"):
        try:
            return data.decode("utf-8-sig"), "utf-8-sig", []
        except UnicodeDecodeError:
            pass  # có BOM nhưng thân không phải UTF-8: thử bảng mã cũ
    else:
        try:
            return data.decode("utf-8"), "utf-8", []
        except UnicodeDecodeError:
            pass
    # cp1258 có vài byte không định nghĩa: thay bằng U+FFFD thay vì từ chối cả file.
    text = data.decode("cp1258", errors="replace")
    return (
        text,
        "cp1258",
        [ParseWarning("encoding_fallback", _WARNING_MESSAGES["encoding_fallback"])],
    )


# --- CSV ----------------------------------------------------------------------------------------


def _iter_csv(text: str, delimiter: str) -> Iterator[list[str]]:
    try:
        yield from csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    except csv.Error:
        raise _decode_failed("CSV sai cấu trúc hoặc có ô quá lớn") from None


def choose_delimiter(text: str) -> str:
    """Dấu phân cách cho số cột nhất quán nhất ở những dòng đầu.

    Dấu phân cách thật phải có mặt trong dòng tiêu đề (tiêu đề từ hai cột trở lên) và các dòng sau
    đa số cùng số cột. File nhiều cột vì thế chọn đúng dấu dù câu tiếng Việt có dấu phẩy, còn file
    một cột chứa dấu phẩy trong câu rơi vào `;` hoặc TAB, tức không bị tách câu.
    """
    best, best_score = DELIMITERS[0], None
    skipped = False
    for delimiter in DELIMITERS:
        if text.count(delimiter) > MAX_DELIMITERS:
            skipped = True
            continue
        widths: list[int] = []
        seen = 0
        try:
            for row in _iter_csv(text, delimiter):
                seen += 1
                if any(cell.strip() for cell in row):
                    widths.append(len(row))
                if len(widths) >= SNIFF_ROWS or seen >= SNIFF_RAW_ROWS:
                    break
        except ApiError:
            continue
        if not widths:
            continue
        consistent = sum(w == widths[0] for w in widths) / len(widths)
        has_header_delimiter = widths[0] >= 2
        score = (
            has_header_delimiter and consistent >= 0.8,
            has_header_delimiter,
            consistent,
            widths[0],
        )
        if best_score is None or score > best_score:
            best, best_score = delimiter, score
    if best_score is None and skipped:
        raise _decode_failed("CSV có quá nhiều dấu phân cách")
    return best


# --- XLSX ---------------------------------------------------------------------------------------


# Header cục bộ của tệp con: chữ ký, phiên bản, cờ, kiểu nén, giờ, ngày, crc, hai cỡ, tên, extra.
_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_LOCAL_SIGNATURE = b"PK\x03\x04"
_ENCRYPTED_FLAG = 0x1


def _inflate_chunks(raw: bytes, member: zipfile.ZipInfo, budget: int) -> Iterator[bytes]:
    """Các đoạn byte THỰC của một tệp con sau khi giải nén, dừng khi tổng đã quá `budget`.

    Giải nén thẳng luồng nén ở đúng vị trí của nó, không đi qua `file_size`/`crc` trong header
    (kẻ tấn công khai nhỏ để qua mặt trình đọc tôn trọng header). Tệp chỉ lưu (stored) không nở
    ra nên các đoạn chính là byte nằm trong file.
    """
    offset = member.header_offset
    header = raw[offset : offset + _LOCAL_HEADER.size]
    if len(header) < _LOCAL_HEADER.size:
        raise _decode_failed("XLSX hỏng")
    signature, *_, name_length, extra_length = _LOCAL_HEADER.unpack(header)
    if signature != _LOCAL_SIGNATURE or member.flag_bits & _ENCRYPTED_FLAG:
        raise _decode_failed("XLSX hỏng hoặc có mật khẩu")
    position = offset + _LOCAL_HEADER.size + name_length + extra_length
    if member.compress_type == zipfile.ZIP_STORED:
        end = position + max(0, min(member.compress_size, len(raw) - position))
        for start in range(position, end, _XLSX_CHUNK):
            yield raw[start : min(start + _XLSX_CHUNK, end)]
        return
    if member.compress_type != zipfile.ZIP_DEFLATED:
        raise _decode_failed("XLSX dùng kiểu nén không hỗ trợ")
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    total = 0
    try:
        while not inflater.eof and total <= budget:
            if inflater.unconsumed_tail:
                chunk = inflater.unconsumed_tail
            else:
                chunk = raw[position : position + _XLSX_CHUNK]
                position += len(chunk)
                if not chunk:
                    break  # luồng nén cụt: phần đã giải nén vẫn được tính
            data = inflater.decompress(chunk, _XLSX_CHUNK)
            total += len(data)
            if data:
                yield data
    except zlib.error:
        raise _decode_failed("XLSX hỏng") from None


def _inflated_size(raw: bytes, member: zipfile.ZipInfo, budget: int) -> int:
    """Số byte THỰC của một tệp con sau khi giải nén, đếm tối đa `budget + 1`."""
    return sum(len(chunk) for chunk in _inflate_chunks(raw, member, budget))


def check_xlsx_size(data: bytes) -> None:
    """Từ chối XLSX nở ra quá `XLSX_MAX_UNCOMPRESSED_MB` khi giải nén (zip bom).

    Đếm byte thực khi giải nén từng tệp con rồi dừng ngay khi vượt trần; không tin kích thước khai
    trong header. Mỗi mục trong danh mục đều được tính nên zip bom kiểu chồng lấn cũng bị chặn.
    """
    limit = XLSX_MAX_UNCOMPRESSED_MB * 1024 * 1024
    try:
        members = zipfile.ZipFile(io.BytesIO(data)).infolist()
    except (zipfile.BadZipFile, ValueError, OSError):
        raise _decode_failed("XLSX hỏng") from None
    if len(members) > XLSX_MAX_MEMBERS:
        raise _decode_failed("XLSX có quá nhiều thành phần")
    total = 0
    for member in members:
        if member.is_dir():
            continue
        total += _inflated_size(data, member, limit - total)
        if total > limit:
            raise _decode_failed(f"XLSX giải nén vượt {XLSX_MAX_UNCOMPRESSED_MB}MB")


_CELL_REF = re.compile(r"([A-Za-z]{1,4})([0-9]{1,12})")
_HUGE = 10**12  # số đo "vô cùng lớn" cho tham chiếu ô sai dạng hoặc quá dài, để chắc chắn bị chặn


def _column_number(letters: str) -> int:
    """Số thứ tự cột từ chữ cái (A = 1, XFD = 16384); 4 chữ cái trở lên vượt mọi cột có thật."""
    if len(letters) > 3:
        return _HUGE
    number = 0
    for char in letters.upper():
        number = number * 26 + ord(char) - 64
    return number


class _SheetExtent:
    """Dòng/cột xa nhất của một sheet theo các thuộc tính `r` của `<row>` và `<c>`.

    Ô không có `r` nằm ngay sau ô trước nó, dòng không có `r` nằm ngay sau dòng trước, nên ngoài
    tham chiếu còn đếm số `<row>` và số `<c>` nhiều nhất trong một dòng.
    """

    def __init__(self) -> None:
        self.rows = 0
        self.max_row = 0
        self.max_col = 0
        self.cells = 0
        self.max_cells = 0

    def start(self, name: str, attrs: dict[str, str]) -> None:
        tag = name.rpartition(":")[2]  # chấp nhận tiền tố không gian tên như <x:row>
        if tag == "row":
            self.rows += 1
            self.cells = 0
            ref = attrs.get("r", "")
            if ref.isdigit():
                self.max_row = max(self.max_row, int(ref) if len(ref) <= 12 else _HUGE)
        elif tag == "c":
            self.cells += 1
            self.max_cells = max(self.max_cells, self.cells)
            match = _CELL_REF.fullmatch(attrs.get("r", ""))
            if match:
                self.max_col = max(self.max_col, _column_number(match[1]))
                self.max_row = max(self.max_row, int(match[2]))
        else:
            return
        self.check()

    def check(self) -> None:
        """Từ chối ngay khi vùng ô vượt giới hạn: các số đo chỉ tăng nên dừng sớm là đúng."""
        rows, cols = max(self.rows, self.max_row), max(self.max_col, self.max_cells)
        if rows > XLSX_MAX_SHEET_ROWS or cols > MAX_COLUMNS or rows * cols > XLSX_MAX_SHEET_CELLS:
            raise _decode_failed(
                f"XLSX có vùng ô quá lớn (tối đa {XLSX_MAX_SHEET_ROWS} dòng, {MAX_COLUMNS} cột)"
            )


def _reject_doctype(*_: object) -> None:
    raise _decode_failed("XLSX chứa khai báo DTD")


def _scan_part(raw: bytes, member: zipfile.ZipInfo, budget: int) -> None:
    """Quét một tệp con XML bằng expat, không dựng cây, ném lỗi nếu vùng ô quá lớn.

    Parser thật nên thuộc tính được hiểu đúng như trình đọc XML của calamine hiểu (khoảng trắng
    quanh dấu `=`, nháy đơn hoặc kép, `>` trong giá trị); XML sai cấu trúc bị từ chối chứ không
    bỏ qua. DTD bị từ chối hẳn nên không có thực thể nào được mở rộng. Tệp không phải XML (ảnh,
    cài đặt máy in...) không thể là sheet nên được bỏ qua.
    """
    extent = _SheetExtent()
    parser = expat.ParserCreate()
    parser.StartElementHandler = extent.start
    parser.StartDoctypeDeclHandler = _reject_doctype
    started = False
    try:
        for chunk in _inflate_chunks(raw, member, budget):
            if not started:
                started = True
                if not chunk.lstrip(b"\xef\xbb\xbf\xff\xfe\xfe\xff \t\r\n\x00").startswith(b"<"):
                    return
            parser.Parse(chunk, False)
        if started:
            parser.Parse(b"", True)
    except expat.ExpatError:
        raise _decode_failed("XLSX hỏng") from None


def _relationship_targets(raw: bytes, member: zipfile.ZipInfo, budget: int) -> list[str]:
    """Tên tệp con (đường dẫn trong zip) mà các quan hệ của workbook trỏ tới."""
    targets: list[str] = []

    def start(name: str, attrs: dict[str, str]) -> None:
        target = attrs.get("Target")
        if name.rpartition(":")[2] == "Relationship" and target:
            if attrs.get("TargetMode", "").lower() != "external":
                # calamine ghép "xl/" vào đường dẫn tương đối; chuẩn hoá thêm để bắt cả `..`.
                name_in_zip = target[1:] if target.startswith("/") else f"xl/{target}"
                targets.extend((name_in_zip, posixpath.normpath(name_in_zip)))

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    parser.StartDoctypeDeclHandler = _reject_doctype
    try:
        for chunk in _inflate_chunks(raw, member, budget):
            parser.Parse(chunk, False)
        parser.Parse(b"", True)
    except expat.ExpatError:
        raise _decode_failed("XLSX hỏng") from None
    return targets


def check_xlsx_extent(data: bytes) -> None:
    """Từ chối XLSX có vùng ô quá lớn TRƯỚC khi calamine cấp phát ma trận đặc cho sheet.

    Calamine cấp phát cả hình chữ nhật từ ô đầu tới ô xa nhất (~31 byte mỗi ô), nên một file vài KB
    có ô ở dòng 1.048.576 cột 50 cũng ăn ~1,6GB và làm container chết. Kiểm này đọc đúng các tệp
    mà workbook trỏ tới (calamine mở sheet theo quan hệ, không theo tên thư mục) cùng các tệp
    dưới `xl/worksheets/`, nên đổi tên tệp sheet không qua mặt được.
    """
    try:
        members = zipfile.ZipFile(io.BytesIO(data)).infolist()
    except (zipfile.BadZipFile, ValueError, OSError):
        raise _decode_failed("XLSX hỏng") from None
    by_name: dict[str, list[zipfile.ZipInfo]] = {}
    for member in members:
        if not member.is_dir():
            by_name.setdefault(member.filename, []).append(member)
    if "xl/workbook.xml" not in by_name:
        raise _decode_failed("không phải XLSX (thiếu xl/workbook.xml)")
    limit = XLSX_MAX_UNCOMPRESSED_MB * 1024 * 1024
    names = {name for name in by_name if name.startswith("xl/worksheets/")}
    for rels in by_name.get("xl/_rels/workbook.xml.rels", []):
        names.update(_relationship_targets(data, rels, limit))
    for name in sorted(names & by_name.keys()):
        for member in by_name[
            name
        ]:  # tên trùng trong danh mục: quét hết, không đoán cái nào được đọc
            _scan_part(data, member, limit)  # ném lỗi ngay khi vùng ô vượt giới hạn


def _cell_text(cell: object) -> str:
    """Ô XLSX -> chuỗi: số nguyên không có ".0", ngày giờ theo ISO."""
    if cell is None or cell == "":
        return ""
    if isinstance(cell, bool):
        return "TRUE" if cell else "FALSE"
    if isinstance(cell, float):
        return str(int(cell)) if cell.is_integer() and abs(cell) < 1e15 else repr(cell)
    if isinstance(cell, datetime.datetime | datetime.date | datetime.time):
        return cell.isoformat()
    return str(cell)


def _iter_xlsx(data: bytes) -> tuple[Iterator[list[str]], list[ParseWarning]]:
    from python_calamine import CalamineWorkbook, SheetTypeEnum, SheetVisibleEnum

    check_xlsx_size(data)
    check_xlsx_extent(data)
    try:
        book = CalamineWorkbook.from_object(io.BytesIO(data))
        visible = [
            meta.name
            for meta in book.sheets_metadata
            if meta.typ == SheetTypeEnum.WorkSheet and meta.visible == SheetVisibleEnum.Visible
        ]
        if not visible:
            raise _decode_failed("XLSX không có sheet nào để đọc")
        sheet = book.get_sheet_by_name(visible[0])
    except ApiError:
        raise
    except Exception:  # calamine báo lỗi bằng nhiều kiểu riêng (Calamine/Xlsx/Zip...)
        raise _decode_failed("không mở được XLSX") from None
    warnings = []
    if len(visible) > 1:
        warnings.append(
            ParseWarning(
                "other_sheets_ignored", _WARNING_MESSAGES["other_sheets_ignored"], len(visible) - 1
            )
        )

    def rows() -> Iterator[list[str]]:
        for row in sheet.iter_rows():
            yield [_cell_text(cell) for cell in row]

    return rows(), warnings


# --- Dựng bảng ----------------------------------------------------------------------------------


def _clean_cell(cell: str) -> str:
    """Bỏ NUL và khoảng trắng hai đầu, chuẩn hoá NFC (cp1258 cho dấu tổ hợp), cắt ô quá dài."""
    return unicodedata.normalize("NFC", cell.replace("\x00", "").strip()[:MAX_CELL_CHARS])


def _clean_header(header: Sequence[str]) -> tuple[list[str], int]:
    """Tên cột sạch, không trống, không trùng. Trả thêm số cột phải đặt lại tên."""
    names: list[str] = []
    seen: set[str] = set()
    renamed = 0
    for index, raw in enumerate(header):
        name = _clean_cell(raw)[:MAX_COLUMN_CHARS]
        candidate = name or f"Cột {index + 1}"
        suffix = 2
        while candidate in seen:
            candidate = f"{name or f'Cột {index + 1}'} ({suffix})"
            suffix += 1
        if candidate != name:
            renamed += 1
        seen.add(candidate)
        names.append(candidate)
    return names, renamed


def build_table(
    raw_rows: Iterable[Sequence[str]], encoding: str | None, warnings: list[ParseWarning]
) -> ParsedTable:
    """Từ các dòng thô (dòng đầu là tiêu đề) thành bảng đã làm sạch và kiểm giới hạn."""
    iterator = iter(raw_rows)
    columns: list[str] | None = None
    renamed = 0
    rows: list[list[str]] = []
    source_rows: list[int] = []
    blank = 0  # dòng trống sau tiêu đề, để báo cho người dùng
    empty = 0  # mọi dòng trống kể cả trước tiêu đề, để dừng khi file toàn dòng trống
    total = 0  # số dòng dữ liệu không trống, kể cả phần vượt giới hạn không được lưu

    for raw in iterator:
        cells = [_clean_cell(cell) for cell in raw]
        if not any(cells):
            empty += 1
            if empty > MAX_BLANK_ROWS:
                raise _decode_failed(f"quá {MAX_BLANK_ROWS} dòng trống")
            if columns is not None:
                blank += 1
            continue
        if columns is None:
            if len(cells) > MAX_COLUMNS:
                raise _decode_failed(f"quá {MAX_COLUMNS} cột")
            columns, renamed = _clean_header(cells)
            continue
        total += 1
        if total <= MAX_ROWS:
            width = len(columns)
            rows.append((cells + [""] * width)[:width])
            source_rows.append(total + blank)  # số dòng dữ liệu gốc: kể cả dòng trống đã bỏ
    if columns is None or total == 0:
        raise _error("no_rows", "File không có dòng dữ liệu nào (chỉ có tiêu đề hoặc trống).")
    if total > MAX_ROWS:
        raise _error(
            "too_many_rows",
            f"File có {total} dòng, vượt giới hạn {MAX_ROWS} dòng mỗi lần.",
            n_rows=total,
            max_rows=MAX_ROWS,
        )
    extra = list(warnings)
    if blank:
        extra.append(ParseWarning("blank_rows_skipped", f"Đã bỏ {blank} dòng trống.", blank))
    if renamed:
        extra.append(
            ParseWarning(
                "columns_renamed",
                f"Đã đặt lại tên {renamed} cột có tiêu đề trống hoặc trùng.",
                renamed,
            )
        )
    return ParsedTable(columns, rows, source_rows, encoding, extra)


def parse_upload(filename: str, data: bytes) -> ParsedTable:
    """Phân tích nội dung file tải lên. `filename` chỉ dùng để nhận dạng phần mở rộng."""
    if not data.strip():
        raise _error("no_rows", "File trống.")
    if detect_kind(filename, data) == "xlsx":
        rows, warnings = _iter_xlsx(data)
        return build_table(rows, None, warnings)
    text, encoding, warnings = decode_csv(data)
    return build_table(_iter_csv(text, choose_delimiter(text)), encoding, warnings)


# --- Đoán cột -----------------------------------------------------------------------------------

# Tên cột văn bản, theo thứ tự ưu tiên; khớp trên tên đã bỏ dấu và viết thường.
_TEXT_HINTS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bgop y\b",
        r"\by kien\b",
        r"\bnhan xet\b",
        r"\bphan hoi\b",
        r"\bnoi dung\b",
        r"\bfeedback\b",
        r"\bcomment\b",
        r"\btext\b",
    )
)
_GROUP_HINT = re.compile(
    r"\b(nhom|mon|lop|khoa|bo mon|hoc phan|group|class|course|subject|department)\b"
)
_NUMBERISH = re.compile(r"[\d\s.,:/%+\-]+")
SAMPLE_FOR_GUESS = 300
MIN_AVG_CHARS = 8


def fold(name: str) -> str:
    """Tên cột để so khớp: bỏ dấu, chữ thường, ký tự lạ thành dấu cách."""
    return re.sub(r"[^a-z0-9]+", " ", strip_diacritics(name).lower()).strip()


def is_illustrative_name(name: str) -> bool:
    """Tên cột nhóm của file mẫu ("Nhóm (minh hoạ)"): dữ liệu nhóm là minh hoạ, không có thật."""
    return "minh hoa" in fold(name)


def is_illustrative(kind: str, group_col: str | None, columns: list[str]) -> bool:
    """Nhóm là dữ liệu minh hoạ khi batch tạo từ file mẫu hoặc tên cột nhóm ghi "minh hoạ".

    Chưa chọn cột nhóm (lúc vừa tải file) thì xét mọi tên cột.
    """
    if kind == "sample":
        return True
    if group_col is not None:
        return is_illustrative_name(group_col)
    return any(is_illustrative_name(name) for name in columns)


def _column_cells(table: ParsedTable, index: int) -> list[str]:
    return [row[index] for row in table.rows[:SAMPLE_FOR_GUESS] if row[index]]


def _is_texty(cells: Sequence[str]) -> bool:
    return bool(cells) and sum(not _NUMBERISH.fullmatch(c) for c in cells) / len(cells) >= 0.5


def suggest_text_column(table: ParsedTable) -> str | None:
    """Cột văn bản: theo tên (góp ý, ý kiến, nội dung...), rồi theo độ dài trung bình."""
    folded = [fold(name) for name in table.columns]
    for hint in _TEXT_HINTS:
        for index, name in enumerate(folded):
            if hint.search(name) and _is_texty(_column_cells(table, index)):
                return table.columns[index]
    best, best_length = None, 0.0
    for index, name in enumerate(table.columns):
        cells = _column_cells(table, index)
        if not _is_texty(cells):
            continue
        average = sum(len(c) for c in cells) / len(cells)
        if average >= MIN_AVG_CHARS and average > best_length:
            best, best_length = name, average
    return best


def suggest_group_column(table: ParsedTable, text_column: str | None) -> str | None:
    """Cột nhóm: tên gợi ý nhóm/môn/lớp và có từ 2 đến `MAX_GROUPS` giá trị khác nhau."""
    for index, name in enumerate(table.columns):
        if name == text_column or not _GROUP_HINT.search(fold(name)):
            continue
        distinct = {row[index] for row in table.rows if row[index]}
        if 2 <= len(distinct) <= MAX_GROUPS:
            return name
    return None
