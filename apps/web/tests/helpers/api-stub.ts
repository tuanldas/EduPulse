// Thay `fetch` toàn cục bằng một bộ định tuyến phục vụ phản hồi đã ghi từ API thật (tests/fixtures/).
// Client API tra `fetch` lúc gọi nên chỉ cần gọi `stubApi({...})` ở đầu mỗi test.
import fs from "node:fs";
import path from "node:path";

export type Handler = (request: Request) => Response | Promise<Response>;

let routes: Record<string, Handler> = {};
export const calls: { key: string; body: unknown }[] = [];

function installFetchStub(): void {
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(input, init);
    const key = `${request.method} ${new URL(request.url).pathname}`;
    const text = await request.clone().text();
    calls.push({ key, body: text ? JSON.parse(text) : undefined });
    const handler = routes[key];
    // Không có tuyến thì giống mất mạng: fetch ném TypeError.
    if (!handler) throw new TypeError("fetch failed");
    return handler(request);
  }) as typeof fetch;
}

/** Đặt bảng tuyến cho test hiện tại, khoá dạng "POST /api/analyze". */
export function stubApi(table: Record<string, Handler>): void {
  routes = table;
  calls.length = 0;
  installFetchStub();
}

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

/** Đọc một fixture đã ghi bằng `pnpm record:fixtures`. */
export function fixture<T = unknown>(name: string): T {
  const file = path.join(import.meta.dirname, "..", "fixtures", name);
  return JSON.parse(fs.readFileSync(file, "utf-8")) as T;
}

/** Một lời hứa mở khoá được từ bên ngoài, để điều khiển thứ tự các phản hồi trong test. */
export function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

export function callsTo(key: string): { key: string; body: unknown }[] {
  return calls.filter((call) => call.key === key);
}
