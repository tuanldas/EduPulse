import { describe, expect, it } from "vitest";

import { failureFromResponse } from "@/lib/api/errors";
import { batchFailureMessage, jobFailureMessage } from "@/lib/batch/batch-errors";

function failure(status: number, code: string, params?: Record<string, number>) {
  return failureFromResponse(status, { detail: { code, message: "chỉ cho người phát triển", params } });
}

describe("batchFailureMessage", () => {
  it("giới hạn dung lượng lấy từ máy chủ trước, rồi mới đến cấu hình", () => {
    expect(batchFailureMessage(failure(413, "file_too_large", { max_mb: 10 }))).toContain("10 MB");
    expect(batchFailureMessage(failure(413, "file_too_large"), { maxUploadMb: 8 })).toContain("8 MB");
    expect(batchFailureMessage(failure(413, "file_too_large"))).toContain("File quá lớn");
  });

  it("ghép số dòng của file vào câu báo quá giới hạn", () => {
    const message = batchFailureMessage(failure(422, "too_many_rows", { n_rows: 3450, max_rows: 2000 }));
    expect(message).toContain("3.450 câu");
    expect(message).toContain("2.000 câu mỗi lần");
    expect(message).toContain("tách file");
  });

  it("dùng giới hạn của cấu hình khi máy chủ không gửi kèm", () => {
    const message = batchFailureMessage(failure(422, "too_many_rows", { n_rows: 3000 }), { maxRows: 2000 });
    expect(message).toContain("3.000 câu");
    expect(message).toContain("2.000");
  });

  it("vẫn có câu hoàn chỉnh khi thiếu số", () => {
    expect(batchFailureMessage(failure(422, "too_many_rows"))).toContain("quá nhiều câu");
    expect(batchFailureMessage(failure(422, "too_many_groups"))).toContain("quá nhiều giá trị");
  });

  it("báo thiếu cột văn bản theo câu của wireframe", () => {
    expect(batchFailureMessage(failure(422, "no_text_column"))).toContain("Không tìm thấy cột văn bản");
  });

  it("nói rõ khi phiên đã kết thúc", () => {
    expect(batchFailureMessage(failure(404, "batch_not_found"))).toContain("Phiên làm việc đã kết thúc");
  });

  it("không bao giờ lộ message của máy chủ", () => {
    for (const code of [
      "file_too_large",
      "unsupported_file_type",
      "no_rows",
      "decode_failed",
      "invalid_column",
      "already_running",
      "batch_readonly",
      "not_done",
      "row_not_found",
      "invalid_override",
      "ma_la_chua_biet",
    ]) {
      expect(batchFailureMessage(failure(409, code))).not.toContain("người phát triển");
    }
  });

  it("mã dùng chung và mã lạ rơi về câu chung của nền tảng", () => {
    expect(batchFailureMessage(failure(429, "busy"))).toContain("bận");
    expect(batchFailureMessage(failure(404, "demo_unavailable"))).toContain("Dữ liệu demo chưa sẵn");
    expect(batchFailureMessage(failure(418, "ma_la_chua_biet"))).toBe(
      "Yêu cầu không thành công. Hãy thử lại.",
    );
  });
});

describe("jobFailureMessage", () => {
  it("mã lạ hoặc thiếu mã vẫn có hướng dẫn xử lý", () => {
    expect(jobFailureMessage(null)).toContain("Phân tích không thành công");
    expect(jobFailureMessage("lỗi_gì_đó")).toContain("thử lại");
  });

  it("báo rõ khi cột đã chọn không còn nội dung", () => {
    expect(jobFailureMessage("no_content")).toContain("không còn nội dung");
  });
});
