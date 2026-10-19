import path from "node:path";

import { expect, test } from "@playwright/test";

import {
  SCREENSHOTS,
  SENTENCES,
  THEMES,
  VIEWPORTS,
  analyze,
  explainSettled,
  expectNoHorizontalScroll,
  openPage,
  settle,
} from "./helpers";

// Chụp ảnh trang 1 và trang 4 ở 1440 và 390, sáng và tối, để đối chiếu với `docs/wireframes/screenshots/`.
// Ảnh nằm ở apps/web/test-results/visual/ (đã gitignore). Cần app đang chạy (E2E_BASE_URL) và API
// thật phía sau. Trang 2 và 3 có spec riêng của người làm hai trang đó.
const shot = (name: string) => path.join(SCREENSHOTS, name);

for (const viewport of VIEWPORTS) {
  for (const theme of THEMES) {
    const tag = `${viewport.label}-${theme}`;

    test(`trang 1 · phân tích câu · ${tag}`, async ({ browser }) => {
      const { context, page, problems } = await openPage(browser, viewport, theme);
      await page.goto("/");
      await settle(page);
      await page.screenshot({ path: shot(`01-phan-tich-cau-trong-${tag}.png`), fullPage: true });

      await analyze(page, SENTENCES.highlighted);
      await explainSettled(page);
      await settle(page);
      await page.screenshot({ path: shot(`01-phan-tich-cau-${tag}.png`), fullPage: true });
      await expectNoHorizontalScroll(page);
      expect(problems).toEqual([]);
      await context.close();
    });

    test(`trang 1 · các trạng thái · ${tag}`, async ({ browser }) => {
      const { context, page, problems } = await openPage(browser, viewport, theme);
      await page.goto("/");
      await settle(page);

      await analyze(page, SENTENCES.noDiacritics);
      await expect(page.getByText("Câu có vẻ được gõ không dấu")).toBeVisible();
      await explainSettled(page);
      await page.screenshot({ path: shot(`01-khong-dau-${tag}.png`), fullPage: true });

      await analyze(page, SENTENCES.withPhone);
      await expect(page.getByText(/Đã ẩn \d+ thông tin cá nhân/)).toBeVisible();
      await explainSettled(page);
      await page.screenshot({ path: shot(`01-an-thong-tin-ca-nhan-${tag}.png`), fullPage: true });

      await analyze(page, SENTENCES.long);
      await expect(page.getByText(/nên không tô từ để giữ tốc độ/)).toBeVisible();
      await page.screenshot({ path: shot(`01-cau-dai-${tag}.png`), fullPage: true });

      await page.getByLabel("Câu góp ý").fill(SENTENCES.onlyPhone);
      await page.getByRole("button", { name: "Phân tích" }).click();
      await expect(page.getByText("Không còn nội dung để phân tích.")).toBeVisible();
      await page.screenshot({ path: shot(`01-khong-con-noi-dung-${tag}.png`), fullPage: true });
      // Câu chỉ có số điện thoại cố ý nhận 422 `no_content`.
      expect(problems.filter((problem) => problem !== "http 422: /api/analyze")).toEqual([]);
      await context.close();
    });

    test(`trang 1 · so sánh mô hình · ${tag}`, async ({ browser }) => {
      const { context, page, problems } = await openPage(browser, viewport, theme);
      await page.goto("/");
      await settle(page);
      await page.getByRole("switch", { name: /So sánh với mô hình cơ sở/ }).click();
      await analyze(page, SENTENCES.plain);
      await explainSettled(page);
      // Khung so sánh hiện sau khi mô hình cơ sở trả lời, hoặc thông báo lỗi nếu máy chủ chưa hỗ trợ.
      await expect(page.getByRole("region", { name: "So sánh mô hình" })).toBeVisible();
      await expect(page.getByLabel("Đang tính mô hình cơ sở")).toHaveCount(0, { timeout: 30_000 });
      await page.screenshot({ path: shot(`01-so-sanh-mo-hinh-${tag}.png`), fullPage: true });
      // Route so sánh có thể còn trả 501 (chưa cài đặt).
      expect(problems.filter((problem) => problem !== "http 501: /api/analyze")).toEqual([]);
      await context.close();
    });

    test(`trang 1 · lỗi mạng · ${tag}`, async ({ browser }) => {
      const { context, page } = await openPage(browser, viewport, theme);
      await page.route("**/api/analyze", (route) => route.abort("failed"));
      await page.goto("/");
      await settle(page);
      await page.getByLabel("Câu góp ý").fill(SENTENCES.plain);
      await page.getByRole("button", { name: "Phân tích" }).click();
      await expect(page.getByRole("alert").filter({ hasText: "Không kết nối được máy chủ" })).toBeVisible();
      await page.screenshot({ path: shot(`01-loi-mang-${tag}.png`), fullPage: true });
      await context.close();
    });

    test(`trang 4 · mô hình và đạo đức · ${tag}`, async ({ browser }) => {
      const { context, page, problems } = await openPage(browser, viewport, theme);
      await page.goto("/model/");
      await expect(page.getByRole("heading", { name: "Model card" })).toBeVisible({ timeout: 20_000 });
      await settle(page);
      await page.screenshot({ path: shot(`04-mo-hinh-dao-duc-${tag}.png`), fullPage: true });
      await expectNoHorizontalScroll(page);
      expect(problems).toEqual([]);
      await context.close();
    });
  }
}
