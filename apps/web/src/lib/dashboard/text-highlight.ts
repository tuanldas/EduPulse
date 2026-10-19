/** Một đoạn văn bản; `strong` (đậm) hoặc `hit` (tô) tuỳ nơi dùng. */
export interface TextSegment {
  text: string;
  mark: boolean;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Ranh giới từ theo Unicode: `\b` của JavaScript không hiểu chữ có dấu tiếng Việt.
const WORD_CHARS = "\\p{L}\\p{N}_";

/**
 * Tách văn bản thành các đoạn, đánh dấu những chỗ khớp từ khoá (không phân biệt hoa/thường, khớp
 * nguyên từ). Từ khoá dài được thử trước để "phòng máy" thắng "phòng". Không dùng HTML: kết quả được
 * dựng bằng JSX nên chữ của sinh viên luôn được escape.
 */
export function splitByTerms(text: string, terms: readonly string[]): TextSegment[] {
  const source = text.normalize("NFC");
  const unique = [...new Set(terms.map((term) => term.normalize("NFC").trim()).filter(Boolean))];
  if (unique.length === 0) return [{ text: source, mark: false }];

  unique.sort((a, b) => b.length - a.length);
  const pattern = new RegExp(
    `(?<![${WORD_CHARS}])(?:${unique.map(escapeRegExp).join("|")})(?![${WORD_CHARS}])`,
    "giu",
  );

  const segments: TextSegment[] = [];
  let cursor = 0;
  for (const match of source.matchAll(pattern)) {
    const start = match.index;
    if (start > cursor) segments.push({ text: source.slice(cursor, start), mark: false });
    segments.push({ text: match[0], mark: true });
    cursor = start + match[0].length;
  }
  if (cursor < source.length) segments.push({ text: source.slice(cursor), mark: false });
  return segments;
}

// Số cần nhấn mạnh trong câu cảnh báo: "93%", "khoảng 15 điểm", "gấp đôi", "gấp 3 lần".
const EMPHASIS = /(\d+(?:,\d+)?%)|(khoảng \d+(?:,\d+)? điểm)|(gấp (?:đôi|\d+(?:,\d+)? lần))/gu;

/**
 * Câu cảnh báo do máy chủ viết ở dạng chữ thường; in đậm phần trăm đầu tiên (thường là tỉ lệ của chính
 * nhóm) và các cụm "khoảng N điểm", "gấp đôi" để người đọc quét được. Các phần trăm sau (mức nền,
 * khoảng tin cậy) giữ chữ thường.
 */
export function emphasizeSummary(summary: string): TextSegment[] {
  const segments: TextSegment[] = [];
  let cursor = 0;
  let seenPercent = false;
  for (const match of summary.matchAll(EMPHASIS)) {
    const start = match.index;
    const isPercent = match[1] !== undefined;
    const bold = !isPercent || !seenPercent;
    if (isPercent) seenPercent = true;
    if (!bold) continue;
    if (start > cursor) segments.push({ text: summary.slice(cursor, start), mark: false });
    segments.push({ text: match[0], mark: true });
    cursor = start + match[0].length;
  }
  if (cursor < summary.length) segments.push({ text: summary.slice(cursor), mark: false });
  return segments;
}
