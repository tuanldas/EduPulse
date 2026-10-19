import { failureMessage, type ApiFailure } from "@/lib/api/errors";
import { formatNumber } from "@/lib/format";

export interface BatchErrorContext {
  maxUploadMb?: number;
  maxRows?: number;
}

function numberParam(failure: ApiFailure, name: string): number | undefined {
  const value = failure.params?.[name];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/**
 * Câu tiếng Việt cho mã lỗi của phân tích hàng loạt (`detail.code`). Mã dùng chung (mạng, bận, lỗi
 * máy chủ) do `failureMessage` lo; thông điệp `detail.message` chỉ dành cho người phát triển nên
 * không bao giờ hiện ra.
 */
export function batchFailureMessage(failure: ApiFailure, context: BatchErrorContext = {}): string {
  switch (failure.code) {
    case "file_too_large": {
      const limit = numberParam(failure, "max_mb") ?? context.maxUploadMb;
      return limit === undefined
        ? "File quá lớn. Hãy giảm dung lượng hoặc tách file rồi tải lại."
        : `File lớn hơn ${formatNumber(limit)} MB. Hãy giảm dung lượng hoặc tách file rồi tải lại.`;
    }
    case "unsupported_file_type":
      return "Chỉ nhận file CSV hoặc XLSX.";
    case "too_many_rows": {
      const rows = numberParam(failure, "n_rows");
      const max = numberParam(failure, "max_rows") ?? context.maxRows;
      const head =
        rows !== undefined && max !== undefined
          ? `File có ${formatNumber(rows)} câu, vượt giới hạn ${formatNumber(max)} câu mỗi lần.`
          : "File có quá nhiều câu cho một lần phân tích.";
      return `${head} Hãy tách file (ví dụ theo học kỳ hoặc nhóm môn) rồi phân tích lần lượt.`;
    }
    case "no_rows":
      return "File không có dòng dữ liệu nào. Kiểm tra file rồi tải lại.";
    case "decode_failed":
      return "Không đọc được file. Hãy lưu lại dưới dạng CSV UTF-8 hoặc XLSX rồi tải lên lại.";
    case "no_text_column":
      return "Không tìm thấy cột văn bản. File cần ít nhất một cột chữ (ví dụ “Góp ý”). Kiểm tra dòng tiêu đề rồi tải lại.";
    case "invalid_column":
      return "Cột đã chọn không có trong file. Hãy chọn lại cột văn bản.";
    case "insights_unavailable":
      return "Dashboard chưa dùng được: bản mô hình đang chạy thiếu số liệu mức nền. Hãy báo người triển khai.";
    case "invalid_budget":
      return "Ngân sách duyệt không hợp lệ. Hãy chọn lại một trong các mức có sẵn.";
    case "too_many_groups": {
      const groups = numberParam(failure, "n_groups");
      const max = numberParam(failure, "max_groups");
      return groups !== undefined && max !== undefined
        ? `Cột nhóm có ${formatNumber(groups)} giá trị khác nhau, nhiều hơn giới hạn ${formatNumber(max)}. Hãy chọn cột khác hoặc bỏ cột nhóm.`
        : "Cột nhóm có quá nhiều giá trị khác nhau. Hãy chọn cột khác hoặc bỏ cột nhóm.";
    }
    case "batch_not_found":
      return "Phiên làm việc đã kết thúc. Hãy tải file lên lại.";
    case "already_running":
      return "File này đã được phân tích. Muốn đổi cột hoặc ngân sách, hãy tải file lên lại.";
    case "batch_readonly":
      return "Dữ liệu demo chỉ để xem, không chạy lại hay xuất file được.";
    case "not_done":
      return "Phân tích chưa xong. Hãy chờ thanh tiến trình hoàn tất.";
    case "row_not_found":
      return "Không tìm thấy câu này trong kết quả.";
    case "invalid_override":
      return "Nhãn đã sửa không khớp với kết quả hiện có. Hãy tải lại trang rồi thử lại.";
    default:
      return failureMessage(failure);
  }
}

/** Mã lỗi của job nền (`BatchStatus.error_code`, khi `state = failed`). */
export function jobFailureMessage(code: string | null): string {
  const base = "Phân tích không thành công.";
  if (code === "no_content") {
    return `${base} Cột đã chọn không còn nội dung nào để phân tích (chỉ có thông tin cá nhân hoặc ký hiệu). Hãy kiểm tra lại cột văn bản.`;
  }
  return `${base} Hãy thử lại; nếu vẫn lỗi, tách file thành phần nhỏ hơn.`;
}
