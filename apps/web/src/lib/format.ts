// Định dạng số kiểu Việt: `54,1%` · `1.000` · `0,018`. Chênh lệch tỉ lệ ghi là "điểm"
// (docs/design-guidelines.md mục 3). Mọi hàm nhận số thật từ API; không làm tròn ngầm ở nơi khác.

const LOCALE = "vi-VN";
const MINUS = "−"; // dấu trừ đúng (không dùng gạch nối) như wireframe: "−47", "+2"

const formatters = new Map<number, Intl.NumberFormat>();

function numberFormat(digits: number): Intl.NumberFormat {
  let formatter = formatters.get(digits);
  if (!formatter) {
    formatter = new Intl.NumberFormat(LOCALE, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
    formatters.set(digits, formatter);
  }
  return formatter;
}

function finite(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

/** `1409` -> `1.409`; `0.018` với digits = 3 -> `0,018`. */
export function formatNumber(value: number, digits = 0): string {
  return numberFormat(digits).format(finite(value));
}

/** Tỉ lệ 0..1 -> phần trăm: `0.931` -> `93%`; digits = 1 -> `93,1%`. */
export function formatPercent(ratio: number, digits = 0): string {
  return `${formatNumber(finite(ratio) * 100, digits)}%`;
}

/** Điểm số 0..1 trên thang 100 không kèm dấu %: Macro-F1 `0.856` -> `85,6`. */
export function formatScore(ratio: number, digits = 1): string {
  return formatNumber(finite(ratio) * 100, digits);
}

/** Số thập phân giữ nguyên thang 0..1: ECE `0.018` -> `0,018`; F1 từng lớp `0.9607` -> `0,96`. */
export function formatDecimal(value: number, digits = 2): string {
  return formatNumber(value, digits);
}

/** Chênh lệch theo điểm (đã nhân 100), có dấu: `-47` -> `−47`, `2` -> `+2`, `0` -> `0`. */
export function formatPoints(points: number, digits = 0): string {
  const value = finite(points);
  const text = formatNumber(Math.abs(value), digits);
  const roundsToZero = Math.round(Math.abs(value) * 10 ** digits) === 0;
  if (roundsToZero) return text;
  return `${value < 0 ? MINUS : "+"}${text}`;
}

/** Thời gian còn lại: `3` -> `vài giây`, `25` -> `khoảng 25 giây`, `130` -> `khoảng 2 phút`. */
export function formatEta(seconds: number): string {
  const s = finite(seconds);
  if (s < 5) return "vài giây";
  if (s < 60) return `khoảng ${Math.round(s / 5) * 5} giây`;
  return `khoảng ${Math.round(s / 60)} phút`;
}

// Tối đa 1 chữ số thập phân và bỏ ",0" thừa: 5.242.880 byte -> "5 MB", 1.536 -> "1,5 KB".
const compact = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 1 });

/** Dung lượng: `10485760` -> `10 MB`, `1536` -> `1,5 KB`. */
export function formatBytes(bytes: number): string {
  const b = Math.max(0, finite(bytes));
  if (b < 1024) return `${formatNumber(b)} B`;
  if (b < 1024 * 1024) return `${compact.format(b / 1024)} KB`;
  return `${compact.format(b / (1024 * 1024))} MB`;
}
