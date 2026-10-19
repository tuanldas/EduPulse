import path from "node:path";

import { expect, test, type Browser, type Page } from "@playwright/test";

// Chụp ảnh trang 2 (Phân tích hàng loạt) và trang 3 (Dashboard & insight) ở 1440 và 390, sáng và tối, để
// đối chiếu với `docs/wireframes/screenshots/`. Ảnh nằm ở apps/web/test-results/visual/ (đã gitignore).
// Cần app đang chạy (E2E_BASE_URL) và API thật phía sau, có dữ liệu demo (file mẫu 200 câu).
const OUTPUT = path.join("test-results", "visual");
const VIEWPORTS = [
  { label: "1440", width: 1440, height: 900 },
  { label: "390", width: 390, height: 844 },
] as const;
const THEMES = ["light", "dark"] as const;
const RUN_TIMEOUT = 120_000;

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

/** Đợi font và mạng yên, ẩn nút chỉ báo của `next dev`, rồi cuộn lên đầu (thanh bên dính theo viewport). */
async function settle(page: Page) {
  await page.waitForLoadState("networkidle");
  await page.evaluate(() => document.fonts.ready);
  await page.addStyleTag({ content: "nextjs-portal { display: none !important; }" });
  await page.evaluate(() => window.scrollTo(0, 0));
}

async function shot(page: Page, name: string, fullPage = true) {
  await settle(page);
  await page.screenshot({ path: path.join(OUTPUT, `${name}.png`), fullPage });
}

/** Dùng file mẫu rồi chạy với thiết lập mặc định; trả về khi bảng kết quả đã sẵn sàng. */
async function runSampleBatch(page: Page) {
  await page.getByRole("button", { name: /Dùng file mẫu/ }).click();
  const run = page.getByRole("button", { name: /^Phân tích \d/ });
  await expect(run).toBeEnabled({ timeout: 30_000 });
  await run.click();
  await expect(page.getByText("Đã phân tích xong.")).toBeVisible({ timeout: RUN_TIMEOUT });
  await expect(page.getByRole("heading", { name: /Kết quả/ })).toBeVisible();
}

// Mỗi kịch bản chạy thật một batch qua mô hình, nên cho thời gian rộng hơn mặc định 30 giây.
test.describe.configure({ timeout: 180_000 });

for (const viewport of VIEWPORTS) {
  for (const theme of THEMES) {
    const tag = `${viewport.label}-${theme}`;

    test(`trang 2 · phân tích hàng loạt · ${tag}`, async ({ browser }) => {
      const { context, page, problems } = await openPage(browser, viewport, theme);
      await page.goto("/batch/");
      await expect(page.getByRole("heading", { name: "1 · Tải file" })).toBeVisible();
      await shot(page, `02-phan-tich-hang-loat-trong-${tag}`);

      await page.getByRole("button", { name: /Dùng file mẫu/ }).click();
      await expect(page.getByRole("button", { name: /^Phân tích \d/ })).toBeEnabled({ timeout: 30_000 });
      await shot(page, `02-phan-tich-hang-loat-thiet-lap-${tag}`);

      await page.getByRole("button", { name: /^Phân tích \d/ }).click();
      await expect(page.getByText("Đã phân tích xong.")).toBeVisible({ timeout: RUN_TIMEOUT });
      await expect(page.getByRole("heading", { name: /Kết quả/ })).toBeVisible();
      await shot(page, `02-phan-tich-hang-loat-${tag}`);

      // Ngăn giải thích bên phải (chụp theo viewport vì là lớp phủ cố định).
      await page.getByRole("button", { name: /^Giải thích câu/ }).first().click();
      await expect(page.getByRole("dialog")).toBeVisible();
      await expect(page.getByLabel("Đang tính giải thích")).toHaveCount(0, { timeout: 30_000 });
      await shot(page, `02-giai-thich-${tag}`, false);
      await page.keyboard.press("Escape");
      await expect(page.getByRole("dialog")).toHaveCount(0);

      // Sửa một nhãn rồi xem tab "Tất cả kết quả".
      const edit = page.getByLabel(/^Sửa cảm xúc câu/).first();
      await edit.selectOption({ index: 1 });
      await expect(page.getByText(/^1 \/ \d+$/)).toBeVisible();
      await page.getByRole("tab", { name: /Tất cả kết quả/ }).click();
      await shot(page, `02-tat-ca-ket-qua-${tag}`);

      expect(problems).toEqual([]);
      await context.close();
    });

    test(`trang 2 · các trạng thái lỗi · ${tag}`, async ({ browser }) => {
      const { context, page, problems } = await openPage(browser, viewport, theme);
      await page.goto("/batch/");
      await expect(page.getByRole("heading", { name: "1 · Tải file" })).toBeVisible();

      // B · file lỗi: sai loại file bị chặn ngay trên trình duyệt.
      await page.locator('input[type="file"]').setInputFiles({
        name: "ghi-chu.pdf",
        mimeType: "application/pdf",
        buffer: Buffer.from("%PDF-1.4"),
      });
      await expect(page.getByText("Chỉ nhận file CSV hoặc XLSX.")).toBeVisible();
      await shot(page, `02-file-loi-${tag}`);

      // D · quá giới hạn số câu: máy chủ trả 422 `too_many_rows`, nút Phân tích bị khoá.
      const header = "Góp ý\n";
      const rows = Array.from({ length: 2100 }, (_, i) => `Thầy dạy dễ hiểu số ${i + 1}.`).join("\n");
      await page.locator('input[type="file"]').setInputFiles({
        name: "qua-nhieu-cau.csv",
        mimeType: "text/csv",
        buffer: Buffer.from(`${header}${rows}\n`, "utf-8"),
      });
      await expect(page.getByText(/vượt giới hạn/)).toBeVisible({ timeout: 30_000 });
      await expect(page.getByRole("button", { name: "Phân tích", exact: true })).toBeDisabled();
      await shot(page, `02-qua-gioi-han-${tag}`);

      // F5 sau khi có file: tab cũ đã đóng batch (pagehide) nên trang báo phiên đã kết thúc.
      await page.getByRole("button", { name: /Dùng file mẫu/ }).click();
      await expect(page.getByRole("button", { name: /^Phân tích \d/ })).toBeEnabled({ timeout: 30_000 });
      await page.reload();
      await expect(page.getByText("Phiên làm việc đã kết thúc.")).toBeVisible({ timeout: 30_000 });
      await shot(page, `02-phien-ket-thuc-${tag}`);

      // 422/404 là lỗi cố ý của các bước trên; trình duyệt tự ghi chúng vào console.
      expect(
        problems.filter((problem) => !problem.includes("status of 422") && !problem.includes("status of 404")),
      ).toEqual([]);
      await context.close();
    });

    test(`trang 3 · dashboard & insight · ${tag}`, async ({ browser }) => {
      const { context, page, problems } = await openPage(browser, viewport, theme);

      // Mặc định: dữ liệu demo (số thật của tập test UIT-VSFC).
      await page.goto("/dashboard/");
      await expect(page.getByRole("heading", { name: "Cảnh báo so với mức nền" })).toBeVisible({ timeout: 30_000 });
      await shot(page, `03-dashboard-demo-${tag}`);

      // File vừa tải: chạy file mẫu ở trang 2 rồi sang dashboard bằng liên kết (điều hướng trong ứng dụng).
      await page.goto("/batch/");
      await runSampleBatch(page);
      await page.getByRole("link", { name: /Xem dashboard cho file này/ }).click();
      await expect(page).toHaveURL(/\/dashboard\/\?batch=/);
      await expect(page.getByRole("heading", { name: "Cảnh báo so với mức nền" })).toBeVisible({ timeout: 30_000 });
      await expect(page.getByText("Dữ liệu minh hoạ — nhóm gán ngẫu nhiên").first()).toBeVisible();
      await shot(page, `03-dashboard-file-${tag}`);

      // Lọc theo một nhóm: số câu giảm và dashboard tính lại.
      await page.getByRole("button", { name: /Tất cả \d+ nhóm/ }).click();
      await page.getByRole("menuitemcheckbox").first().click();
      await page.keyboard.press("Escape");
      // Đợi menu đóng hẳn (kể cả hiệu ứng mờ dần) trước khi chụp.
      await expect(page.getByRole("menu")).toHaveCount(0);
      await expect(page.getByRole("button", { name: /Tất cả \d+ nhóm/ })).toHaveCount(0);
      await expect(page.getByRole("heading", { name: "Cảnh báo so với mức nền" })).toBeVisible();
      await shot(page, `03-dashboard-loc-nhom-${tag}`);

      // Batch không còn trên máy chủ: báo phiên đã kết thúc, vẫn cho xem dữ liệu demo.
      await page.goto("/dashboard/?batch=khong-ton-tai");
      await expect(page.getByText("Phiên làm việc đã kết thúc.")).toBeVisible({ timeout: 30_000 });
      await shot(page, `03-dashboard-phien-ket-thuc-${tag}`);

      expect(
        problems.filter((problem) => !problem.includes("status of 404") && !problem.includes("status of 422")),
      ).toEqual([]);
      await context.close();
    });
  }
}
