import { expect, test, type Page } from "@playwright/test";

import { runAnalysis, uploadFeedback, usingPage } from "./batch-helpers";
import { expectNoHorizontalScroll, settle } from "./helpers";

// Co giãn của trang 2 (Phân tích hàng loạt) và trang 3 (Dashboard & insight), WCAG 1.4.10: không cuộn ngang cả trang
// ở 320 (chuẩn reflow), 390 (di động), 900 và 1100 (hai bên các điểm ngắt của bố cục). Nhãn "Dữ liệu minh hoạ — nhóm gán
// ngẫu nhiên" dài khoảng 310px phải xuống dòng theo khung chứa thay vì bị cắt. File tải lên là file góp ý tự viết.
// Mỗi kịch bản chạy thật một batch qua mô hình nên cho thời gian rộng hơn mặc định.
test.describe.configure({ timeout: 180_000 });

const ILLUSTRATIVE = "Dữ liệu minh hoạ — nhóm gán ngẫu nhiên";

/** Mọi nhãn minh hoạ đang hiện phải hiện trọn chữ: nội dung không rộng hơn khung của chính nó (không bị cắt). */
async function expectIllustrativeBadgesNotClipped(page: Page, label: string) {
  const clipped = await page.evaluate((text) => {
    const badges = [...document.querySelectorAll<HTMLElement>('[data-slot="badge"]')].filter(
      (badge) => badge.textContent?.trim() === text && badge.getClientRects().length > 0,
    );
    return badges
      .filter((badge) => badge.scrollWidth > badge.clientWidth + 1 || badge.getBoundingClientRect().right > window.innerWidth + 1)
      .length;
  }, ILLUSTRATIVE);
  expect(clipped, `${label}: nhãn minh hoạ bị cắt hoặc tràn khung nhìn`).toBe(0);
}

for (const width of [320, 390, 900, 1100]) {
  test(`trang 2 và 3 không cuộn ngang ở ${width}px, nhãn minh hoạ xuống dòng`, async ({ browser }) => {
    await usingPage(browser, { label: String(width), width, height: 900 }, "light", async (page) => {
      await page.goto("/batch/");
      await expect(page.getByRole("heading", { name: "1 · Tải file" })).toBeVisible();
      await settle(page);
      await expectNoHorizontalScroll(page);

      await uploadFeedback(page);
      await expectNoHorizontalScroll(page);
      await expectIllustrativeBadgesNotClipped(page, "trang 2 bước thiết lập");

      await runAnalysis(page);
      await expectNoHorizontalScroll(page);
      await page.getByRole("tab", { name: /Tất cả kết quả/ }).click();
      await expectNoHorizontalScroll(page);

      await page.getByRole("link", { name: /Xem dashboard cho file này/ }).click();
      await expect(page.getByRole("heading", { name: "Cảnh báo so với mức nền" })).toBeVisible({ timeout: 30_000 });
      await expect(page.getByText(ILLUSTRATIVE).first()).toBeVisible();
      await expectNoHorizontalScroll(page);
      await expectIllustrativeBadgesNotClipped(page, "trang 3 dữ liệu file");

      // Dữ liệu demo (số liệu thật của tập test): không có nhóm minh hoạ.
      await page.goto("/dashboard/");
      await expect(page.getByRole("heading", { name: "Cảnh báo so với mức nền" })).toBeVisible({ timeout: 30_000 });
      await expectNoHorizontalScroll(page);
    });
  });
}
