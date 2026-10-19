// Lỗi gọi API ở dạng một kiểu duy nhất + câu tiếng Việt theo `detail.code`.
// Máy chủ luôn trả `{detail: {code, message, params?}}`; `message` chỉ để người phát triển đọc,
// giao diện tự ghép câu từ `code` và `params` để giọng văn thống nhất.
import { formatNumber } from "@/lib/format";

export type FailureKind = "network" | "http";

type Params = Readonly<Record<string, number>>;

export class ApiFailure extends Error {
  readonly kind: FailureKind;
  /** Mã HTTP; 0 khi chưa nhận được phản hồi (mất mạng, quá thời gian chờ). */
  readonly status: number;
  /** `detail.code` của máy chủ, hoặc `network` / `timeout` khi không có phản hồi. */
  readonly code: string;
  readonly params: Params;

  constructor(init: { kind: FailureKind; status: number; code: string; params?: Params }) {
    super(`${init.kind}:${init.status}:${init.code}`);
    this.name = "ApiFailure";
    this.kind = init.kind;
    this.status = init.status;
    this.code = init.code;
    this.params = init.params ?? {};
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readParams(value: unknown): Params {
  if (!isRecord(value)) return {};
  const out: Record<string, number> = {};
  for (const [key, item] of Object.entries(value)) {
    if (typeof item === "number" && Number.isFinite(item)) out[key] = item;
  }
  return out;
}

/** Dựng lỗi từ phản hồi không thành công; chịu được thân không phải JSON của máy chủ. */
export function failureFromResponse(status: number, body: unknown): ApiFailure {
  const detail = isRecord(body) && isRecord(body.detail) ? body.detail : null;
  const code = detail && typeof detail.code === "string" ? detail.code : `http_${status}`;
  return new ApiFailure({ kind: "http", status, code, params: readParams(detail?.params) });
}

export function networkFailure(code: "network" | "timeout" = "network"): ApiFailure {
  return new ApiFailure({ kind: "network", status: 0, code });
}

/** Người dùng/ứng dụng tự huỷ yêu cầu (AbortController); khác với hết thời gian chờ. */
export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export function isTimeoutError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "TimeoutError";
}

const MESSAGES: Record<string, (params: Params) => string> = {
  network: () => "Không kết nối được máy chủ. Kiểm tra mạng rồi thử lại.",
  timeout: () => "Máy chủ phản hồi quá lâu. Hãy thử lại.",
  busy: () => "Máy chủ đang bận, thử lại sau ít phút.",
  engine_unavailable: () => "Mô hình chưa sẵn sàng, thử lại sau ít phút.",
  not_implemented: () => "Tính năng này chưa sẵn sàng trên máy chủ.",
  no_content: () =>
    "Câu chỉ gồm thông tin cá nhân hoặc ký hiệu nên không có nội dung để phân tích.",
  text_empty: () => "Hãy nhập một câu góp ý.",
  text_too_long: (p) =>
    p.max_chars
      ? `Câu dài hơn ${formatNumber(p.max_chars)} ký tự. Hãy rút gọn rồi thử lại.`
      : "Câu quá dài. Hãy rút gọn rồi thử lại.",
  xai_too_long: (p) =>
    p.n_words && p.max_words
      ? `Câu có ${formatNumber(p.n_words)} từ, giải thích chỉ hỗ trợ tối đa ${formatNumber(p.max_words)} từ.`
      : "Câu quá dài để tô từ giải thích.",
  validation_error: () => "Dữ liệu gửi lên không hợp lệ.",
  payload_too_large: () => "Dữ liệu gửi lên quá lớn.",
  not_found: () => "Không tìm thấy dữ liệu yêu cầu.",
  demo_unavailable: () => "Dữ liệu demo chưa sẵn (bản build không tải được dataset).",
  model_card_unavailable: () => "Chưa có model card cho bản mô hình đang chạy.",
  internal_error: () => "Máy chủ gặp lỗi. Hãy thử lại sau.",
};

/** Câu tiếng Việt cho người dùng; mã lạ thì dựa vào mã HTTP. */
export function failureMessage(failure: ApiFailure): string {
  const known = MESSAGES[failure.code];
  if (known) return known(failure.params);
  if (failure.status === 429) return MESSAGES.busy({});
  if (failure.status >= 500) return MESSAGES.internal_error({});
  return "Yêu cầu không thành công. Hãy thử lại.";
}

/** Lỗi nên cho người dùng bấm "Thử lại" (khác với lỗi do dữ liệu họ nhập). */
export function isRetryable(failure: ApiFailure): boolean {
  return (
    failure.kind === "network" ||
    failure.status === 429 ||
    failure.status === 503 ||
    failure.status >= 500
  );
}
