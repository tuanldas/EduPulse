import path from "node:path";

import { expect, test, type Browser, type Page } from "@playwright/test";

// Chụp ảnh trang 1 và trang 4 ở 1440 và 390, sáng và tối, để đối chiếu với `docs/wireframes/screenshots/`.
// Ảnh nằm ở apps/web/test-results/visual/ (đã gitignore). Cần app đang chạy (E2E_BASE_URL) và API
// thật phía sau. Trang 2 và 3 có spec riêng của người làm hai trang đó.
const OUTPUT = path.join("test-results", "visual");
const VIEWPORTS = [
  { label: "1440", width: 1440, height: 900 },
  { label: "390", width: 390, height: 844 },
] as const;
const THEMES = ["light", "dark"] as const;

const SENTENCE = "[đã gỡ câu UIT-VSFC].";
const NO_DIACRITICS = "thay giang day nhiet tinh de hieu";
const WITH_PII = "Cô Nguyễn Thị Lan dạy rất nhiệt tình, em xin số 0912345678 để hỏi bài.";
const LONG_SENTENCE = Array.from({ length: 6 }, () => "[đã gỡ câu UIT-VSFC] thử nghiệm ngay những gì vừa học.").join(
  " ",
);

async function openPage(browser: Browser, viewport: (typeof VIEWPORTS)[number], theme: (typeof THEMES)[number]) {
  const context = await browser.newContext({
    viewport: { width: viewport.width, height: viewport.height },
    colorScheme: theme,
    locale: "vi-VN",
    timezoneId: "Asia/Ho_Chi_Minh",
  });
  const page = await context.newPage();
  const problems: string[] = [];
  page.on("pageerror", (error) => problems.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") problems.push(`console: ${message.text()}`);
  });
  return { context, page, problems };
}

async function settle(page: Page) {
  await page.waitForLoadState("networkidle");
  await page.evaluate(() => document.fonts.ready);
  // Nút chỉ báo của `next dev` che góc trái dưới; bản export không có nó.
  await page.addStyleTag({ content: "nextjs-portal { display: none !important; }" });
}

/** Đợi phần giải thích tính xong (có bảng, hoặc báo không có từ nào để tô, hoặc câu quá dài). */
async function explainSettled(page: Page) {
  await expect(page.getByLabel("Đang tính giải thích")).toHaveCount(0, { timeout: 30_000 });
}

async function analyze(page: Page, text: string) {
  await page.getByLabel("Câu góp ý").fill(text);
  await page.getByRole("button", { name: "Phân tích" }).click();
  await expect(page.getByRole("heading", { name: "Kết quả" })).toBeVisible();
}

for (const viewport of VIEWPORTS) {
  for (const theme of THEMES) {
    const tag = `${viewport.label}-${theme}`;

    test(`trang 1 · phân tích câu · ${tag}`, async ({ browser }) => {
      const { context, page, problems } = await openPage(browser, viewport, theme);
      await page.goto("/");
      await settle(page);
      await page.screenshot({ path: path.join(OUTPUT, `01-phan-tich-cau-trong-${tag}.png`), fullPage: true });

      await analyze(page, SENTENCE);
      await explainSettled(page);
      await settle(page);
      await page.screenshot({ path: path.join(OUTPUT, `01-phan-tich-cau-${tag}.png`), fullPage: true });
      expect(problems).toEqual([]);
      await context.close();
    });

    test(`trang 1 · các trạng thái · ${tag}`, async ({ browser }) => {
      const { context, page, problems } = await openPage(browser, viewport, theme);
      await page.goto("/");
      await settle(page);

      await analyze(page, NO_DIACRITICS);
      await expect(page.getByText("Câu có vẻ được gõ không dấu")).toBeVisible();
      await explainSettled(page);
      await page.screenshot({ path: path.join(OUTPUT, `01-khong-dau-${tag}.png`), fullPage: true });

      await analyze(page, WITH_PII);
      await expect(page.getByText(/Đã ẩn \d+ thông tin cá nhân/)).toBeVisible();
      await explainSettled(page);
      await page.screenshot({ path: path.join(OUTPUT, `01-an-thong-tin-ca-nhan-${tag}.png`), fullPage: true });

      await analyze(page, LONG_SENTENCE);
      await expect(page.getByText(/nên không tô từ để giữ tốc độ/)).toBeVisible();
      await page.screenshot({ path: path.join(OUTPUT, `01-cau-dai-${tag}.png`), fullPage: true });

      await page.getByLabel("Câu góp ý").fill("0912345678");
      await page.getByRole("button", { name: "Phân tích" }).click();
      await expect(page.getByText("Không còn nội dung để phân tích.")).toBeVisible();
      await page.screenshot({ path: path.join(OUTPUT, `01-khong-con-noi-dung-${tag}.png`), fullPage: true });
      expect(problems).toEqual([]);
      await context.close();
    });

    test(`trang 4 · mô hình và đạo đức · ${tag}`, async ({ browser }) => {
      const { context, page, problems } = await openPage(browser, viewport, theme);
      await page.goto("/model/");
      await expect(page.getByRole("heading", { name: "Model card" })).toBeVisible({ timeout: 20_000 });
      await settle(page);
      await page.screenshot({ path: path.join(OUTPUT, `04-mo-hinh-dao-duc-${tag}.png`), fullPage: true });
      expect(problems).toEqual([]);
      await context.close();
    });
  }
}
