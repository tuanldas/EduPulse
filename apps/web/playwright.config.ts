import { defineConfig, devices } from "@playwright/test";

// E2E_BASE_URL trỏ tới app đang chạy: `next dev` khi dựng giao diện, hoặc container/uvicorn phục vụ
// bản export (phase kiểm thử). Không tự bật server ở đây để cùng một cấu hình dùng được cho cả hai.
const baseURL = process.env.E2E_BASE_URL ?? "http://localhost:3000";

export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.ts",
  outputDir: "./test-results/artifacts",
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL,
    locale: "vi-VN",
    timezoneId: "Asia/Ho_Chi_Minh",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
