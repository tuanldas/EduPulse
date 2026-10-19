import { expect, test } from "@playwright/test";

import {
  SENTENCES,
  THEMES,
  VIEWPORTS,
  analyze,
  explainSettled,
  expectNoA11yViolations,
  openPage,
  settle,
} from "./helpers";

// Quét trợ năng bằng axe-core (WCAG 2.1 AA + thực hành tốt) trên trang 1 ở các trạng thái chính, trang 4 và
// trang 404, sáng/tối, 1440/390. Chạy trên app thật (E2E_BASE_URL), tốt nhất là bản export do FastAPI phục vụ.
for (const viewport of VIEWPORTS) {
  for (const theme of THEMES) {
    const tag = `${viewport.label} ${theme}`;

    test.describe(`trợ năng · ${tag}`, () => {
      test("trang 1: trạng thái đầu và kết quả có giải thích", async ({ browser }) => {
        const { context, page } = await openPage(browser, viewport, theme);
        await page.goto("/");
        // Đợi /api/config nạp xong: hoặc có nhóm chip câu mẫu, hoặc có thông báo thiếu dữ liệu demo.
        await expect(
          page.getByRole("group", { name: /Hoặc thử câu mẫu/ }).or(page.getByText("Dữ liệu demo chưa sẵn")),
        ).toBeVisible();
        await settle(page);
        await expectNoA11yViolations(page, "trang 1 trạng thái đầu");

        await analyze(page, SENTENCES.highlighted);
        await explainSettled(page);
        await expectNoA11yViolations(page, "trang 1 có kết quả");
        await context.close();
      });

      test("trang 1: không dấu, thông tin cá nhân, câu dài, không còn nội dung", async ({ browser }) => {
        const { context, page } = await openPage(browser, viewport, theme);
        await page.goto("/");
        await settle(page);

        await analyze(page, SENTENCES.noDiacritics);
        await expect(page.getByText(/Câu có vẻ được gõ không dấu/)).toBeVisible();
        await explainSettled(page);
        await expectNoA11yViolations(page, "trang 1 không dấu");

        await analyze(page, SENTENCES.withPhone);
        await expect(page.getByText(/Đã ẩn \d+ thông tin cá nhân/)).toBeVisible();
        await explainSettled(page);
        await expectNoA11yViolations(page, "trang 1 thông tin cá nhân");

        await analyze(page, SENTENCES.long);
        await expect(page.getByText(/nên không tô từ để giữ tốc độ/)).toBeVisible();
        await expectNoA11yViolations(page, "trang 1 câu dài");

        await page.getByLabel("Câu góp ý").fill(SENTENCES.onlyPhone);
        await page.getByRole("button", { name: "Phân tích" }).click();
        await expect(page.getByText("Không còn nội dung để phân tích.")).toBeVisible();
        await expectNoA11yViolations(page, "trang 1 không còn nội dung");
        await context.close();
      });

      test("trang 1: bật so sánh mô hình", async ({ browser }) => {
        const { context, page } = await openPage(browser, viewport, theme);
        await page.goto("/");
        await settle(page);
        await page.getByRole("switch", { name: /So sánh với mô hình cơ sở/ }).click();
        await analyze(page, SENTENCES.plain);
        await explainSettled(page);
        await expect(page.getByRole("region", { name: "So sánh mô hình" })).toBeVisible();
        await expect(page.getByLabel("Đang tính mô hình cơ sở")).toHaveCount(0, { timeout: 30_000 });
        await expectNoA11yViolations(page, "trang 1 so sánh mô hình");
        await context.close();
      });

      test("trang 4: mô hình và đạo đức, cả hai tác vụ", async ({ browser }) => {
        const { context, page } = await openPage(browser, viewport, theme);
        await page.goto("/model/");
        await expect(page.getByRole("heading", { name: "Model card" })).toBeVisible({ timeout: 20_000 });
        await settle(page);
        await expectNoA11yViolations(page, "trang 4 cảm xúc");

        await page.getByRole("button", { name: "Chủ đề" }).click();
        await expectNoA11yViolations(page, "trang 4 chủ đề");

        await page.getByText("Xem bảng số").click();
        await expect(page.getByRole("table", { name: /Độ tin cậy báo so với tỉ lệ đúng thực tế/ })).toBeVisible();
        await expectNoA11yViolations(page, "trang 4 bảng số độ tin cậy");
        await context.close();
      });

      test("trang 404", async ({ browser }) => {
        const { context, page } = await openPage(browser, viewport, theme);
        const response = await page.goto("/khong-co-trang-nay/");
        expect(response?.status()).toBe(404);
        await expect(page.getByRole("heading", { name: "Không tìm thấy trang" })).toBeVisible();
        await settle(page);
        await expectNoA11yViolations(page, "trang 404");
        await context.close();
      });
    });
  }
}

test.describe("trợ năng · điều hướng di động", () => {
  for (const theme of THEMES) {
    test(`ngăn điều hướng mở ra ${theme}`, async ({ browser }) => {
      const { context, page } = await openPage(browser, { width: 390, height: 844 }, theme);
      await page.goto("/");
      await settle(page);
      await page.getByRole("button", { name: "Mở điều hướng" }).click();
      await expect(page.getByRole("dialog")).toBeVisible();
      await expectNoA11yViolations(page, "ngăn điều hướng di động");
      await context.close();
    });
  }
});
