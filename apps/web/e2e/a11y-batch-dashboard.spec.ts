import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

import { analyzeFeedback, showHighlightedExplanation, uploadFeedback, usingPage } from "./batch-helpers";
import {
  THEMES,
  VIEWPORTS,
  expectNoA11yViolations,
  expectNoHorizontalScroll,
  settle,
  waitForAnimations,
} from "./helpers";

// Quét trợ năng bằng axe-core (WCAG 2.1 AA + thực hành tốt) trên trang 2 và trang 3 ở các trạng thái chính,
// sáng/tối, 1440/390; ở 390 còn kiểm trang không cuộn ngang. Mọi file tải lên là file góp ý tự viết (không có câu
// UIT-VSFC), xem feedback-file.ts. Chạy trên app thật (E2E_BASE_URL), tốt nhất là bản export do FastAPI phục vụ
// cùng origin với /api (xem helpers.ts và golden-path.spec.ts).

test.describe.configure({ timeout: 180_000 });

/**
 * Như `expectNoA11yViolations` nhưng bỏ luật `region` ("nội dung phải nằm trong landmark"): menu của Base UI được
 * dựng trong portal ở cuối `<body>`, ngoài `<main>`, và đó là cách mọi popup hoạt động. Các luật còn lại
 * (vai trò, tên, tương phản, bàn phím) vẫn áp dụng cho chính menu.
 */
async function expectNoA11yViolationsInPopup(page: Page, label: string) {
  await waitForAnimations(page);
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"])
    .disableRules(["region"])
    .analyze();
  const summary = results.violations.map(
    (violation) => `${violation.id} (${violation.impact}): ${violation.help} — ${violation.nodes[0].html.slice(0, 160)}`,
  );
  expect(summary, `${label}: vi phạm trợ năng`).toEqual([]);
}

for (const viewport of VIEWPORTS) {
  for (const theme of THEMES) {
    const tag = `${viewport.label} ${theme}`;

    test.describe(`trợ năng trang 2–3 · ${tag}`, () => {
      test("trang 2: trạng thái đầu, thiết lập, kết quả và ngăn giải thích", async ({ browser }) => {
        await usingPage(browser, viewport, theme, async (page) => {
          await page.goto("/batch/");
          await expect(page.getByRole("heading", { name: "1 · Tải file" })).toBeVisible();
          await settle(page);
          await expectNoA11yViolations(page, "trang 2 trạng thái đầu");

          await uploadFeedback(page);
          await expectNoA11yViolations(page, "trang 2 thiết lập");

          await page.getByRole("button", { name: /^Phân tích \d/ }).click();
          await expect(page.getByText("Đã phân tích xong.")).toBeVisible({ timeout: 90_000 });
          await expect(page.getByRole("table")).toBeVisible();
          await expectNoA11yViolations(page, "trang 2 kết quả");
          if (viewport.width <= 640) await expectNoHorizontalScroll(page);

          await page.getByRole("tab", { name: /Tất cả kết quả/ }).click();
          await expectNoA11yViolations(page, "trang 2 tab tất cả kết quả");

          await page.getByRole("button", { name: /^Giải thích câu/ }).first().click();
          await expect(page.getByRole("dialog")).toBeVisible();
          await expect(page.getByLabel("Đang tính giải thích")).toHaveCount(0, { timeout: 30_000 });
          await expectNoA11yViolations(page, "trang 2 ngăn giải thích");

          await showHighlightedExplanation(page);
          await expectNoA11yViolations(page, "trang 2 ngăn giải thích có từ được tô và bảng xác suất");
        });
      });

      test("trang 2: lỗi file, quá giới hạn và phiên đã kết thúc", async ({ browser }) => {
        await usingPage(browser, viewport, theme, async (page) => {
          await page.goto("/batch/");
          await expect(page.getByRole("heading", { name: "1 · Tải file" })).toBeVisible();

          await page.locator('input[type="file"]').setInputFiles({
            name: "ghi-chu.pdf",
            mimeType: "application/pdf",
            buffer: Buffer.from("%PDF-1.4"),
          });
          await expect(page.getByText("Chỉ nhận file CSV hoặc XLSX.")).toBeVisible();
          await expectNoA11yViolations(page, "trang 2 lỗi file");

          const rows = Array.from({ length: 2100 }, (_, i) => `Thầy dạy dễ hiểu số ${i + 1}.`).join("\n");
          await page.locator('input[type="file"]').setInputFiles({
            name: "qua-nhieu-cau.csv",
            mimeType: "text/csv",
            buffer: Buffer.from(`Góp ý\n${rows}\n`, "utf-8"),
          });
          await expect(page.getByText(/vượt giới hạn/)).toBeVisible({ timeout: 30_000 });
          await expectNoA11yViolations(page, "trang 2 quá giới hạn");

          await uploadFeedback(page);
          await page.reload();
          await expect(page.getByText("Phiên làm việc đã kết thúc.")).toBeVisible({ timeout: 30_000 });
          await expectNoA11yViolations(page, "trang 2 phiên đã kết thúc");
        });
      });

      test("trang 3: dữ liệu demo, file vừa tải, bộ lọc nhóm và phiên đã kết thúc", async ({ browser }) => {
        await usingPage(browser, viewport, theme, async (page) => {
          await page.goto("/dashboard/");
          await expect(page.getByRole("heading", { name: "Cảnh báo so với mức nền" })).toBeVisible({ timeout: 30_000 });
          await settle(page);
          await expectNoA11yViolations(page, "trang 3 dữ liệu demo");
          if (viewport.width <= 640) await expectNoHorizontalScroll(page);

          await page.goto("/batch/");
          await analyzeFeedback(page);
          await page.getByRole("link", { name: /Xem dashboard cho file này/ }).click();
          await expect(page.getByRole("heading", { name: "Cảnh báo so với mức nền" })).toBeVisible({ timeout: 30_000 });
          await expect(page.getByRole("button", { name: /Tất cả \d+ nhóm/ })).toBeVisible();
          // File tự viết được dựng để có thẻ cảnh báo theo nhóm (nhóm B bị chê giảng viên); nếu mô hình đổi mà thẻ
          // này biến mất thì phải chỉnh lại bảng số lượng ở feedback-file.ts chứ không bỏ qua phần quét thẻ cảnh báo.
          const groupAlert = page.locator('[data-slot="card"]', {
            has: page.getByRole("heading", { name: /ở nhóm/ }),
          });
          await expect(groupAlert).toBeVisible();
          await expectNoA11yViolations(page, "trang 3 file vừa tải");
          if (viewport.width <= 640) await expectNoHorizontalScroll(page);

          // Từ thẻ cảnh báo sang danh sách câu liên quan: bộ lọc nhóm hiện thành chip có nút bỏ lọc.
          await groupAlert.getByRole("button", { name: "Xem câu liên quan" }).click();
          // Panel của tab cũ còn nằm lại trong lúc chuyển tab; đợi chỉ còn panel của chủ đề mới rồi mới tìm chip.
          await expect(page.getByRole("tabpanel")).toHaveCount(1);
          await expect(page.getByRole("button", { name: /^Bỏ lọc nhóm/ })).toBeVisible();
          await expectNoA11yViolations(page, "trang 3 câu tiêu biểu đã lọc theo cảnh báo");

          await page.getByRole("button", { name: /Tất cả \d+ nhóm/ }).click();
          await expect(page.getByRole("menu")).toBeVisible();
          await expectNoA11yViolationsInPopup(page, "trang 3 bộ lọc nhóm đang mở");
          await page.keyboard.press("Escape");

          await page.goto("/dashboard/?batch=khong-ton-tai");
          await expect(page.getByText("Phiên làm việc đã kết thúc.")).toBeVisible({ timeout: 30_000 });
          await expectNoA11yViolations(page, "trang 3 phiên đã kết thúc");
        });
      });
    });
  }
}
