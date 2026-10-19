import { expect, test, type Page } from "@playwright/test";

import { leavePage } from "./batch-helpers";

// Vòng đời phiên của batch trên trình duyệt thật: heartbeat khi tab mở, `close` khi rời trang, và việc
// chuyển sang dashboard bằng liên kết không đóng batch. Cần chạy trên bản export do uvicorn phục vụ cùng
// origin với /api (E2E_BASE_URL), vì kiểm tra trực tiếp trạng thái batch trên máy chủ bằng `request`.

test.describe.configure({ timeout: 120_000 });

// Máy chủ giữ tối đa 8 batch: rời trang để pagehide xoá batch của test vừa chạy.
test.afterEach(async ({ page }) => {
  await leavePage(page);
});

async function createSampleBatch(page: Page): Promise<string> {
  await page.getByRole("button", { name: /Dùng file mẫu/ }).click();
  await expect(page.getByRole("button", { name: /^Phân tích \d/ })).toBeEnabled({ timeout: 30_000 });
  const stored = await page.evaluate(() => window.sessionStorage.getItem("edupulse.batch"));
  expect(stored).not.toBeNull();
  return (JSON.parse(stored ?? "{}") as { batchId: string }).batchId;
}

test("gửi heartbeat theo chu kỳ cấu hình khi tab còn mở", async ({ page }) => {
  await page.clock.install();
  await page.goto("/batch/");
  const batchId = await createSampleBatch(page);

  const config = await page.request.get("/api/config");
  const { heartbeat_s: heartbeatS } = (await config.json()) as { heartbeat_s: number };

  const beat = page.waitForRequest(
    (request) => request.method() === "POST" && request.url().endsWith(`/api/batches/${batchId}/heartbeat`),
  );
  await page.clock.runFor(heartbeatS * 1000 + 500);
  const request = await beat;
  const response = await request.response();
  expect(response?.status()).toBe(204);
});

test("rời trang thì báo máy chủ xoá batch (sendBeacon khi pagehide)", async ({ page }) => {
  await page.goto("/batch/");
  const batchId = await createSampleBatch(page);
  expect((await page.request.get(`/api/batches/${batchId}`)).status()).toBe(200);

  await page.goto("about:blank");
  await expect
    .poll(async () => (await page.request.get(`/api/batches/${batchId}`)).status(), { timeout: 10_000 })
    .toBe(404);
});

test("sang dashboard bằng liên kết trong ứng dụng thì batch vẫn còn và dashboard hiện đúng file", async ({ page }) => {
  await page.goto("/batch/");
  const batchId = await createSampleBatch(page);
  await page.getByRole("button", { name: /^Phân tích \d/ }).click();
  await expect(page.getByText("Đã phân tích xong.")).toBeVisible({ timeout: 90_000 });

  await page.getByRole("link", { name: /Xem dashboard cho file này/ }).click();
  await expect(page).toHaveURL(new RegExp(`/dashboard/\\?batch=${batchId}`));
  await expect(page.getByRole("heading", { name: "Cảnh báo so với mức nền" })).toBeVisible({ timeout: 30_000 });
  // Không có tải lại trang nào: batch chưa bị đóng.
  expect((await page.request.get(`/api/batches/${batchId}`)).status()).toBe(200);

  // Quay về trang 2 bằng thanh bên: kết quả được dựng lại từ batch đã lưu trong tab.
  await page.getByRole("link", { name: "Phân tích hàng loạt" }).first().click();
  await expect(page.getByRole("heading", { name: /Kết quả/ })).toBeVisible({ timeout: 30_000 });
  expect((await page.request.get(`/api/batches/${batchId}`)).status()).toBe(200);
});

test("đang có nhãn sửa chưa xuất thì rời trang bị hỏi lại", async ({ page }) => {
  await page.goto("/batch/");
  await createSampleBatch(page);
  await page.getByRole("button", { name: /^Phân tích \d/ }).click();
  await expect(page.getByText("Đã phân tích xong.")).toBeVisible({ timeout: 90_000 });

  await page.getByLabel(/^Sửa cảm xúc câu/).first().selectOption({ index: 1 });

  const dialogTypes: string[] = [];
  page.on("dialog", async (dialog) => {
    dialogTypes.push(dialog.type());
    // Bỏ qua = ở lại trang, nên điều hướng bị huỷ.
    await dialog.dismiss();
  });
  await page.goto("about:blank").catch(() => undefined);
  expect(dialogTypes).toEqual(["beforeunload"]);
  // Ở lại trang: kết quả vẫn còn trên màn hình.
  await expect(page.getByRole("heading", { name: /Kết quả/ })).toBeVisible();
});
