import { expect, type Browser, type Page } from "@playwright/test";

import { feedbackFile } from "./feedback-file";
import { SENTENCES, openPage, type Theme, type Viewport } from "./helpers";

// Dùng chung cho các spec E2E của trang 2 (Phân tích hàng loạt) và trang 3 (Dashboard & insight).

/** Thời gian chờ một lần phân tích file góp ý (khoảng 300 câu) qua mô hình thật. */
export const RUN_TIMEOUT = 90_000;

/**
 * File góp ý tự viết (không có câu UIT-VSFC) cho mọi spec của trang 2–3. Ngoài các câu ghép sẵn còn có một câu
 * không dấu, một câu có số điện thoại, một câu quá dài chưa giải thích được và một dòng chỉ có số điện thoại
 * (sau khi ẩn thông tin cá nhân không còn nội dung nên bị bỏ qua).
 */
export const FEEDBACK = feedbackFile([SENTENCES.noDiacritics, SENTENCES.withPhone, SENTENCES.long, SENTENCES.onlyPhone]);

/** Tải file góp ý lên trang 2 và đợi bước thiết lập sẵn sàng; trả về id batch đang giữ trong tab. */
export async function uploadFeedback(page: Page): Promise<string> {
  await page.locator('input[type="file"]').setInputFiles({
    name: FEEDBACK.name,
    mimeType: FEEDBACK.mimeType,
    buffer: FEEDBACK.buffer,
  });
  await expect(page.getByRole("button", { name: /^Phân tích \d/ })).toBeEnabled({ timeout: 30_000 });
  const stored = await page.evaluate(() => window.sessionStorage.getItem("edupulse.batch"));
  expect(stored).not.toBeNull();
  return (JSON.parse(stored ?? "{}") as { batchId: string }).batchId;
}

/** Bấm "Phân tích" với thiết lập mặc định; trả về khi bảng kết quả đã sẵn sàng. */
export async function runAnalysis(page: Page) {
  await page.getByRole("button", { name: /^Phân tích \d/ }).click();
  await expect(page.getByText("Đã phân tích xong.")).toBeVisible({ timeout: RUN_TIMEOUT });
  await expect(page.getByRole("table")).toBeVisible();
}

/** Tải file góp ý lên rồi chạy với thiết lập mặc định. */
export async function analyzeFeedback(page: Page): Promise<string> {
  const batchId = await uploadFeedback(page);
  await runAnalysis(page);
  return batchId;
}

const MAX_EXPLAIN_STEPS = 12;

/**
 * Trong ngăn giải thích đang mở, đi qua "Câu sau" tới câu có từ được tô rồi mở bảng xác suất của nó. Câu đầu danh
 * sách (độ tin cậy thấp nhất) thường không có từ nào đủ ảnh hưởng để tô, nên ảnh/quét trợ năng của ngăn cần cả hai.
 */
export async function showHighlightedExplanation(page: Page) {
  const dialog = page.getByRole("dialog");
  const probabilities = dialog.getByRole("button", { name: /Xem bảng xác suất/ });
  for (let step = 0; step < MAX_EXPLAIN_STEPS; step += 1) {
    await expect(dialog.getByLabel("Đang tính giải thích")).toHaveCount(0, { timeout: 30_000 });
    if (await probabilities.isVisible()) {
      await probabilities.click();
      await expect(dialog.getByRole("table")).toBeVisible();
      return;
    }
    await dialog.getByRole("button", { name: /^Câu sau/ }).click();
  }
  throw new Error(`Không câu nào trong ${MAX_EXPLAIN_STEPS} câu đầu có từ được tô để kiểm ngăn giải thích`);
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
