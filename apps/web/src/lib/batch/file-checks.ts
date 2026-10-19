import { failureFromResponse, type ApiFailure } from "@/lib/api/errors";

const ALLOWED_EXTENSION = /\.(csv|xlsx)$/i;
const BYTES_PER_MB = 1024 * 1024;

/**
 * Kiểm tra file ngay trên trình duyệt trước khi tải lên, để lỗi hiển thị tức thì thay vì sau khi đã
 * gửi vài MB. Máy chủ vẫn kiểm lại tất cả; lỗi trả về cùng khuôn mã (`file_too_large`, ...) với lỗi của
 * máy chủ nên giao diện dùng chung một bộ câu thông báo.
 */
export function checkUploadFile(
  file: { name: string; size: number },
  maxUploadMb: number | undefined,
): ApiFailure | null {
  const fail = (status: number, code: string) => failureFromResponse(status, { detail: { code } });
  if (!ALLOWED_EXTENSION.test(file.name)) return fail(415, "unsupported_file_type");
  if (file.size === 0) return fail(422, "no_rows");
  if (maxUploadMb !== undefined && file.size > maxUploadMb * BYTES_PER_MB) {
    return fail(413, "file_too_large");
  }
  return null;
}
