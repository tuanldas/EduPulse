import { afterEach, describe, expect, it } from "vitest";

import { API_BASE, apiUrl, call, requestSignal, toFailure } from "@/lib/api/client";
import { ApiFailure } from "@/lib/api/errors";

const ok = (status = 200) => new Response(null, { status });

describe("gỡ kết quả của openapi-fetch (call)", () => {
  it("trả dữ liệu khi thành công và undefined khi 204 không có thân", async () => {
    await expect(call(Promise.resolve({ data: { a: 1 }, response: ok() }))).resolves.toEqual({ a: 1 });
    await expect(call(Promise.resolve({ response: ok(204) }))).resolves.toBeUndefined();
  });

  it("ném ApiFailure có mã và tham số từ {detail: {code, params}}", async () => {
    const failure = await call(
      Promise.resolve({
        error: { detail: { code: "too_many_rows", message: "x", params: { n_rows: 3000, max_rows: 2000 } } },
        response: ok(422),
      }),
    ).catch((error: unknown) => error);
    expect(failure).toBeInstanceOf(ApiFailure);
    expect(failure).toMatchObject({ kind: "http", status: 422, code: "too_many_rows", params: { n_rows: 3000, max_rows: 2000 } });
  });

  it("mất mạng hoặc hết giờ thành lỗi mạng, huỷ chủ động được ném nguyên", async () => {
    const offline = await call(Promise.reject(new TypeError("fetch failed"))).catch((error: unknown) => error);
    expect(offline).toMatchObject({ kind: "network", code: "network", status: 0 });

    const slow = await call(Promise.reject(new DOMException("hết giờ", "TimeoutError"))).catch((error: unknown) => error);
    expect(slow).toMatchObject({ kind: "network", code: "timeout" });

    const aborted = new DOMException("huỷ", "AbortError");
    await expect(call(Promise.reject(aborted))).rejects.toBe(aborted);
  });

  it("chuẩn hoá lỗi lạ thành ApiFailure để giao diện không vỡ", () => {
    const failure = new ApiFailure({ kind: "http", status: 400, code: "x" });
    expect(toFailure(failure)).toBe(failure);
    expect(toFailure(new Error("lỗi lập trình"))).toMatchObject({ code: "internal_error" });
  });
});

describe("địa chỉ API và tín hiệu huỷ", () => {
  const original = Object.getOwnPropertyDescriptor(AbortSignal, "any");

  afterEach(() => {
    if (original) Object.defineProperty(AbortSignal, "any", original);
  });

  it("ghép địa chỉ với NEXT_PUBLIC_API_BASE (đã bỏ dấu / cuối)", () => {
    expect(API_BASE).toBe("http://api.test");
    expect(apiUrl("/api/batches")).toBe("http://api.test/api/batches");
  });

  it("tín hiệu gộp huỷ khi người gọi huỷ", () => {
    const controller = new AbortController();
    const signal = requestSignal(controller.signal);
    expect(signal.aborted).toBe(false);
    controller.abort();
    expect(signal.aborted).toBe(true);
  });

  it("tín hiệu gộp hết giờ báo TimeoutError (khác với huỷ chủ động)", async () => {
    const signal = requestSignal(undefined, 5);
    await new Promise((resolve) => setTimeout(resolve, 40));
    expect(signal.aborted).toBe(true);
    expect((signal.reason as DOMException).name).toBe("TimeoutError");
  });

  it("trình duyệt chưa có AbortSignal.any vẫn gộp được tín hiệu", () => {
    Object.defineProperty(AbortSignal, "any", { value: undefined, configurable: true });
    const controller = new AbortController();
    const signal = requestSignal(controller.signal);
    expect(signal.aborted).toBe(false);
    controller.abort(new DOMException("đổi câu", "AbortError"));
    expect(signal.aborted).toBe(true);
    expect((signal.reason as DOMException).name).toBe("AbortError");

    const already = new AbortController();
    already.abort();
    expect(requestSignal(already.signal).aborted).toBe(true);
  });
});
