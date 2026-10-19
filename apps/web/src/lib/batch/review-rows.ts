import type { ResultRow, ResultsResponse } from "./types";

/** Số dòng mỗi trang của bảng duyệt. */
export const REVIEW_PAGE_SIZE = 50;

export type ReviewTab = "review" | "all";

/**
 * Các dòng của một tab. "Cần duyệt": đúng thứ tự cố định `review_order` của máy chủ (độ tin cậy tăng
 * dần), không đổi khi người duyệt sửa nhãn. "Tất cả": mọi dòng theo thứ tự trong file.
 */
export function rowsForTab(
  results: Pick<ResultsResponse, "rows" | "review_order">,
  tab: ReviewTab,
): ResultRow[] {
  if (tab === "all") return results.rows;
  const byIndex = new Map<number, ResultRow>();
  for (const row of results.rows) byIndex.set(row.index, row);
  const ordered: ResultRow[] = [];
  for (const index of results.review_order) {
    const row = byIndex.get(index);
    if (row !== undefined) ordered.push(row);
  }
  return ordered;
}

export interface Page<T> {
  items: T[];
  /** Trang hiện tại, đã kẹp vào [1, pageCount]. */
  page: number;
  pageCount: number;
  total: number;
  /** Thứ tự (từ 1) của dòng đầu và dòng cuối trang; 0 khi rỗng. */
  first: number;
  last: number;
}

export function paginate<T>(
  items: readonly T[],
  page: number,
  size: number = REVIEW_PAGE_SIZE,
): Page<T> {
  const total = items.length;
  const pageCount = Math.max(1, Math.ceil(total / size));
  const wanted = Number.isFinite(page) ? Math.trunc(page) : 1;
  const current = Math.min(Math.max(1, wanted), pageCount);
  const start = (current - 1) * size;
  const slice = items.slice(start, start + size);
  return {
    items: slice,
    page: current,
    pageCount,
    total,
    first: total === 0 ? 0 : start + 1,
    last: start + slice.length,
  };
}

/** Trang (từ 1) chứa dòng ở vị trí `position` (từ 0), để Câu trước/Câu sau kéo bảng theo câu đang xem. */
export function pageOfPosition(position: number, size: number = REVIEW_PAGE_SIZE): number {
  return Math.floor(Math.max(0, position) / size) + 1;
}

/**
 * Cột "Lý do" của bảng, viết ngắn: "Cả hai không chắc", "Cảm xúc không chắc", "Chủ đề không chắc",
 * "Câu không dấu", "Ngẫu nhiên (kiểm tra)". Câu dài hơn giới hạn giải thích thêm số từ để người duyệt
 * hiểu vì sao không tô được từ. Phần chi tiết (hai lớp cao nhất và xác suất) nằm ở `reasonDetail`.
 */
export function reasonText(row: ResultRow, xaiMaxWords: number): string {
  const parts: string[] = [];
  if (row.source === "audit") {
    parts.push("Ngẫu nhiên (kiểm tra)");
  } else {
    const unsure = new Set<string>();
    let noDiacritics = false;
    for (const reason of row.reasons) {
      if (reason.code === "low_confidence" && reason.task !== null) unsure.add(reason.task);
      else if (reason.code === "no_diacritics") noDiacritics = true;
    }
    if (unsure.has("sentiment") && unsure.has("topic")) parts.push("Cả hai không chắc");
    else if (unsure.has("sentiment")) parts.push("Cảm xúc không chắc");
    else if (unsure.has("topic")) parts.push("Chủ đề không chắc");
    if (noDiacritics) parts.push("Câu không dấu");
    if (parts.length === 0 && row.source === "flag") parts.push("Cần xem xét");
  }
  if (row.n_words > xaiMaxWords) parts.push(`dài ${row.n_words} từ`);
  return parts.length === 0 ? "—" : parts.join(" · ");
}

/** Lý do đầy đủ do máy chủ viết (kèm các lớp cao nhất), dùng làm gợi ý khi rê chuột vào cột "Lý do". */
export function reasonDetail(row: ResultRow): string | undefined {
  const messages = row.reasons.map((reason) => reason.message).filter((message) => message !== "");
  return messages.length === 0 ? undefined : [...new Set(messages)].join("; ");
}
