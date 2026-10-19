/**
 * Phân loại lỗi theo hình dạng thay vì `instanceof`, để các hook không phụ thuộc vào lớp lỗi cụ thể
 * của client API (lỗi HTTP có `status`, lỗi huỷ có `name === "AbortError"`).
 */

export function httpStatusOf(error: unknown): number | null {
  if (typeof error !== "object" || error === null) return null;
  const { status } = error as { status?: unknown };
  return typeof status === "number" ? status : null;
}

/** 404: batch không còn (hết hạn, đã đóng hoặc id sai). */
export function isNotFound(error: unknown): boolean {
  return httpStatusOf(error) === 404;
}

/** Yêu cầu bị huỷ chủ động (đổi trang, đổi bộ lọc): không phải lỗi để báo người dùng. */
export function isAbort(error: unknown): boolean {
  if (typeof error !== "object" || error === null) return false;
  return (error as { name?: unknown }).name === "AbortError";
}
