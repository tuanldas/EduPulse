import { expect, type Browser, type Page } from "@playwright/test";

import { openPage, type Theme, type Viewport } from "./helpers";

// Dùng chung cho các spec E2E của trang 2 (Phân tích hàng loạt) và trang 3 (Dashboard & insight).

/** Thời gian chờ một lần phân tích file mẫu (200 câu) qua mô hình thật. */
export const RUN_TIMEOUT = 90_000;

/** Dùng file mẫu rồi chạy với thiết lập mặc định; trả về khi bảng kết quả đã sẵn sàng. */
export async function analyzeSample(page: Page) {
  await page.getByRole("button", { name: /Dùng file mẫu/ }).click();
  const run = page.getByRole("button", { name: /^Phân tích \d/ });
  await expect(run).toBeEnabled({ timeout: 30_000 });
  await run.click();
  await expect(page.getByText("Đã phân tích xong.")).toBeVisible({ timeout: RUN_TIMEOUT });
  await expect(page.getByRole("table")).toBeVisible();
}

/**
 * Rời trang để trình duyệt gửi `pagehide`, nhờ đó giao diện báo máy chủ xoá batch. Máy chủ chỉ giữ tối đa 8
 * batch cùng lúc, nên test nào tạo batch cũng phải gọi hàm này; đóng ngữ cảnh trình duyệt đột ngột thì không
 * có sự kiện nào được gửi. Hộp thoại "còn nhãn sửa chưa xuất" được chấp nhận để việc rời trang không bị chặn.
 */
export async function leavePage(page: Page) {
  if (page.isClosed()) return;
  // Bỏ trình nghe của chính test (vd. test kiểm hộp thoại đã chọn "ở lại") để chỉ còn một nơi xử lý hộp thoại.
  page.removeAllListeners("dialog");
  page.on("dialog", (dialog) => void dialog.accept());
  await page.goto("about:blank").catch(() => undefined);
}

/** Mở trang trong một ngữ cảnh riêng, chạy `body`, rồi luôn dọn batch và đóng ngữ cảnh kể cả khi test lỗi. */
export async function usingPage(
  browser: Browser,
  viewport: Viewport,
  theme: Theme,
  body: (page: Page, problems: string[]) => Promise<void>,
) {
  const { context, page, problems } = await openPage(browser, viewport, theme);
  try {
    await body(page, problems);
  } finally {
    await leavePage(page);
    await context.close();
  }
}
