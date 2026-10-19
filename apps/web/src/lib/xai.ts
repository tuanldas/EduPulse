// Chuẩn bị dữ liệu giải thích (occlusion) cho giao diện: ghép các từ thành câu có tô nổi và dựng
// các dòng của bảng "Bỏ từ này thì …". Logic thuần để test được, không phụ thuộc React.
import type { HeadExplanation, TokenEffect } from "@/lib/api/types";

export type TokenRole =
  | { kind: "none" }
  /** `rank` 1 = ủng hộ mạnh nhất. */
  | { kind: "support"; rank: number }
  | { kind: "oppose" };

export interface SentenceToken {
  /** Vị trí trong `tokens` của API. */
  index: number;
  text: string;
  role: TokenRole;
  /** Có chèn dấu cách trước từ này khi ghép câu không. */
  spaceBefore: boolean;
}

// Dấu câu dính vào từ đứng trước ("giờ." / "ví dụ:" / "..."); emoticon như ":)" thì không khớp.
const TRAILING_PUNCTUATION = /^(?:[.,;:!?…]+|[)\]}”’»%])$/;
// Dấu mở ngoặc/ngoặc kép dính vào từ đứng sau. "[TÊN]" nhiều ký tự nên không khớp.
const LEADING_PUNCTUATION = /^[([{“‘«]$/;

const WORD_CHAR = /[\p{L}\p{N}]/u;

/** Từ khớp phải đứng trọn vẹn: không nằm giữa một từ khác ("an" trong "bạn"). */
function isWholeWord(text: string, at: number, length: number): boolean {
  const startsInside = WORD_CHAR.test(text[at]) && at > 0 && WORD_CHAR.test(text[at - 1]);
  const last = at + length - 1;
  const endsInside = WORD_CHAR.test(text[last]) && last + 1 < text.length && WORD_CHAR.test(text[last + 1]);
  return !startsInside && !endsInside;
}

/**
 * Mô hình nhìn câu đã chuẩn hoá thành chữ thường, nên `tokens[].text` đều thường. Trả lại chữ hoa/thường
 * của câu gốc cho từng từ bằng cách dò theo thứ tự trong câu; từ nào không dò được thì giữ nguyên dạng
 * của mô hình. Chỉ để đọc cho dễ: không ảnh hưởng tới nhãn hay phần giải thích.
 */
export function restoreCasing(tokens: readonly TokenEffect[], original: string): string[] {
  const haystack = original.toLowerCase();
  if (haystack.length !== original.length) return tokens.map((token) => token.text);
  let cursor = 0;
  return tokens.map((token) => {
    const needle = token.text.toLowerCase();
    if (!needle) return token.text;
    for (let from = cursor; ; ) {
      const at = haystack.indexOf(needle, from);
      if (at === -1) return token.text;
      if (isWholeWord(haystack, at, needle.length)) {
        cursor = at + needle.length;
        return original.slice(at, cursor);
      }
      from = at + 1;
    }
  });
}

/**
 * Gắn vai trò (ủng hộ theo hạng / phản đối) và khoảng trắng cho từng từ của câu. Có `originalText`
 * (câu đã ẩn thông tin cá nhân, giữ hoa/thường) thì hiển thị đúng chữ hoa/thường của câu gốc.
 */
export function sentenceTokens(
  tokens: readonly TokenEffect[],
  support: readonly number[],
  oppose: readonly number[],
  originalText?: string,
): SentenceToken[] {
  const texts = originalText ? restoreCasing(tokens, originalText) : tokens.map((token) => token.text);
  const rankByIndex = new Map(support.map((tokenIndex, position) => [tokenIndex, position + 1]));
  const opposed = new Set(oppose);
  return tokens.map((token, index) => {
    const rank = rankByIndex.get(index);
    const role: TokenRole =
      rank !== undefined ? { kind: "support", rank } : opposed.has(index) ? { kind: "oppose" } : { kind: "none" };
    const previous = index > 0 ? tokens[index - 1].text : null;
    const spaceBefore =
      previous !== null && !TRAILING_PUNCTUATION.test(token.text) && !LEADING_PUNCTUATION.test(previous);
    return { index, text: texts[index], role, spaceBefore };
  });
}

export interface ExplainRow {
  index: number;
  role: "support" | "oppose";
  /** Hạng của từ ủng hộ (1 = mạnh nhất); null với từ phản đối. */
  rank: number | null;
  text: string;
  /** Xác suất của nhãn khi còn đủ câu, và khi bỏ từ này (0..1). */
  pFull: number;
  pWithout: number;
  /** Thay đổi khi bỏ từ này, tính bằng điểm phần trăm: âm = xác suất giảm. */
  changePoints: number;
}

/** Các dòng bảng giải thích: từ ủng hộ (mạnh trước) rồi từ phản đối, đúng thứ tự API. */
export function explainRows(head: HeadExplanation): ExplainRow[] {
  const row = (index: number, role: ExplainRow["role"], rank: number | null): ExplainRow | null => {
    const token = head.tokens[index];
    if (!token) return null;
    return {
      index,
      role,
      rank,
      text: token.text,
      pFull: head.probability,
      pWithout: token.p_without,
      changePoints: (token.p_without - head.probability) * 100,
    };
  };
  const rows = [
    ...head.support.map((index, position) => row(index, "support", position + 1)),
    ...head.oppose.map((index) => row(index, "oppose", null)),
  ];
  return rows.filter((item): item is ExplainRow => item !== null);
}
