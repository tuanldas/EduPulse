// Client API: `openapi-fetch` gõ kiểu theo `schema.d.ts` (sinh từ OpenAPI của FastAPI).
//
// Bản export tĩnh được FastAPI phục vụ cùng origin với `/api`, nên mặc định gọi đường dẫn tương
// đối. Lúc dev (`make web` cổng 3000, `make api` cổng 8000) đặt NEXT_PUBLIC_API_BASE để trình
// duyệt gọi thẳng API; phía API bật CORS bằng EDUPULSE_CORS_ORIGINS.
import createClient from "openapi-fetch";

import {
  ApiFailure,
  failureFromResponse,
  isAbortError,
  isTimeoutError,
  networkFailure,
} from "./errors";
import type { paths } from "./schema";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/+$/, "");

// `fetch` được tra lúc gọi (không giữ tham chiếu lúc nạp module) để test thay `fetch` toàn cục bất kỳ lúc nào.
export const api = createClient<paths>({ baseUrl: API_BASE, fetch: (request) => globalThis.fetch(request) });

/** Địa chỉ đầy đủ của một đường dẫn API, cho `fetch` thường (upload multipart, sendBeacon, tải CSV). */
export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

/** Thời gian chờ mặc định cho một lần gọi ngắn (phân tích, giải thích, cấu hình). */
export const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * Tín hiệu huỷ gộp: `signal` của người gọi (đổi câu khác, rời trang) và hết thời gian chờ.
 * Hết giờ bị báo là `timeout` chứ không phải huỷ, để giao diện hiện lỗi thay vì im lặng.
 */
export function requestSignal(signal?: AbortSignal, timeoutMs = DEFAULT_TIMEOUT_MS): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs);
  return signal ? anySignal([signal, timeout]) : timeout;
}

/** `AbortSignal.any` (Chrome 116, Safari 17.4, Firefox 124) hoặc bản thay thế cho trình duyệt cũ hơn. */
function anySignal(signals: AbortSignal[]): AbortSignal {
  if (typeof AbortSignal.any === "function") return AbortSignal.any(signals);
  const controller = new AbortController();
  for (const source of signals) {
    if (source.aborted) {
      controller.abort(source.reason);
      break;
    }
    source.addEventListener("abort", () => controller.abort(source.reason), { once: true });
  }
  return controller.signal;
}

interface Settled<T> {
  data?: T;
  error?: unknown;
  response: Response;
}

/**
 * Gỡ kết quả của openapi-fetch: trả `data`, còn lỗi thì ném `ApiFailure`.
 * Huỷ chủ động (AbortError) được ném nguyên để người gọi bỏ qua im lặng.
 *
 *     const analysis = await call(api.POST("/api/analyze", { body, signal: requestSignal(s) }));
 */
export async function call<T>(request: Promise<Settled<T>>): Promise<T> {
  let settled: Settled<T>;
  try {
    settled = await request;
  } catch (error) {
    if (isAbortError(error)) throw error;
    throw networkFailure(isTimeoutError(error) ? "timeout" : "network");
  }
  if (!settled.response.ok) {
    throw failureFromResponse(settled.response.status, settled.error);
  }
  // 204 không có thân: `data` là undefined và kiểu phản hồi của route cũng là không có gì.
  return settled.data as T;
}

/** Chuẩn hoá mọi thứ bắt được thành `ApiFailure` (lỗi lập trình cũng không làm vỡ giao diện). */
export function toFailure(error: unknown): ApiFailure {
  if (error instanceof ApiFailure) return error;
  return new ApiFailure({ kind: "http", status: 0, code: "internal_error" });
}
