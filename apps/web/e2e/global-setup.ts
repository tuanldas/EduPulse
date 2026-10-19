import type { FullConfig } from "@playwright/test";

// Làm nóng mô hình cơ sở (TF-IDF + LinearSVC) một lần trước mọi spec. Máy chủ nạp nó lười ở lần gọi
// `compare: true` đầu tiên (trên máy dev, skops kéo cả torch nên mất hàng chục giây), làm kịch bản "so sánh mô hình"
// đầu tiên sau khi khởi động API trượt vì hết thời gian chờ. Câu dùng để làm nóng là câu tự viết.
//
// Làm nóng chỉ là tối ưu thời gian: không gọi được (vd. chạy với `next dev` mà API ở origin khác) thì bỏ qua
// và để từng spec tự báo lỗi nếu API thật sự không dùng được.
const WARM_UP_TIMEOUT_MS = 180_000;

export default async function globalSetup(config: FullConfig): Promise<void> {
  const baseURL = config.projects[0]?.use.baseURL ?? process.env.E2E_BASE_URL ?? "http://localhost:3000";
  const url = new URL("/api/analyze", baseURL);
  const started = Date.now();
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text: "Thầy giảng bài rất dễ hiểu và nhiệt tình.", compare: true }),
      signal: AbortSignal.timeout(WARM_UP_TIMEOUT_MS),
    });
    const seconds = ((Date.now() - started) / 1000).toFixed(1);
    console.log(`Làm nóng mô hình cơ sở: HTTP ${response.status} sau ${seconds} s (${url.origin})`);
  } catch (error) {
    console.log(`Bỏ qua làm nóng mô hình cơ sở, không gọi được ${url.href}: ${String(error)}`);
  }
}
