import { describe, expect, it } from "vitest";

import {
  ApiFailure,
  failureFromResponse,
  failureMessage,
  isAbortError,
  isRetryable,
  isTimeoutError,
  networkFailure,
} from "@/lib/api/errors";

describe("lỗi gọi API", () => {
  it("đọc mã và tham số từ {detail: {code, params}}", () => {
    const failure = failureFromResponse(422, {
      detail: { code: "text_too_long", message: "x", params: { max_chars: 500 } },
    });
    expect(failure).toBeInstanceOf(ApiFailure);
    expect(failure.kind).toBe("http");
    expect(failure.status).toBe(422);
    expect(failure.code).toBe("text_too_long");
    expect(failure.params).toEqual({ max_chars: 500 });
    expect(failureMessage(failure)).toBe("Câu dài hơn 500 ký tự. Hãy rút gọn rồi thử lại.");
  });

  it("ghép câu xai_too_long từ số từ và giới hạn", () => {
    const failure = failureFromResponse(422, {
      detail: { code: "xai_too_long", message: "x", params: { n_words: 45, max_words: 25 } },
    });
    expect(failureMessage(failure)).toBe("Câu có 45 từ, giải thích chỉ hỗ trợ tối đa 25 từ.");
  });

  it("bỏ qua tham số không phải số", () => {
    const failure = failureFromResponse(422, { detail: { code: "x", params: { a: "1", b: 2, c: null } } });
    expect(failure.params).toEqual({ b: 2 });
  });

  it("chịu được thân phản hồi không phải JSON hoặc thiếu mã", () => {
    expect(failureFromResponse(502, "<html>Bad gateway</html>").code).toBe("http_502");
    expect(failureFromResponse(500, null).code).toBe("http_500");
    expect(failureFromResponse(400, { detail: "chuỗi" }).code).toBe("http_400");
  });

  it("dịch mã đã biết và dự phòng theo mã HTTP cho mã lạ", () => {
    expect(failureMessage(failureFromResponse(429, { detail: { code: "busy" } }))).toBe(
      "Máy chủ đang bận, thử lại sau ít phút.",
    );
    expect(failureMessage(failureFromResponse(429, { detail: { code: "la" } }))).toBe(
      "Máy chủ đang bận, thử lại sau ít phút.",
    );
    expect(failureMessage(failureFromResponse(500, null))).toBe("Máy chủ gặp lỗi. Hãy thử lại sau.");
    expect(failureMessage(failureFromResponse(404, { detail: { code: "la" } }))).toBe(
      "Yêu cầu không thành công. Hãy thử lại.",
    );
    expect(failureMessage(failureFromResponse(422, { detail: { code: "no_content" } }))).toContain(
      "không có nội dung để phân tích",
    );
  });

  it("lỗi mạng và hết giờ có câu riêng và cho phép thử lại", () => {
    const offline = networkFailure();
    expect(offline.kind).toBe("network");
    expect(offline.status).toBe(0);
    expect(failureMessage(offline)).toBe("Không kết nối được máy chủ. Kiểm tra mạng rồi thử lại.");
    expect(isRetryable(offline)).toBe(true);
    expect(failureMessage(networkFailure("timeout"))).toBe("Máy chủ phản hồi quá lâu. Hãy thử lại.");
  });

  it("chỉ lỗi tạm thời mới cho thử lại, lỗi do dữ liệu nhập thì không", () => {
    expect(isRetryable(failureFromResponse(429, { detail: { code: "busy" } }))).toBe(true);
    expect(isRetryable(failureFromResponse(503, { detail: { code: "engine_unavailable" } }))).toBe(true);
    expect(isRetryable(failureFromResponse(500, null))).toBe(true);
    expect(isRetryable(failureFromResponse(422, { detail: { code: "text_empty" } }))).toBe(false);
  });

  it("phân biệt huỷ chủ động với hết thời gian chờ", () => {
    expect(isAbortError(new DOMException("x", "AbortError"))).toBe(true);
    expect(isAbortError(new DOMException("x", "TimeoutError"))).toBe(false);
    expect(isTimeoutError(new DOMException("x", "TimeoutError"))).toBe(true);
    expect(isAbortError(new Error("x"))).toBe(false);
  });
});
