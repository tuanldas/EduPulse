import path from "node:path";

import { expect, test, type Locator, type Page } from "@playwright/test";

import { analyzeFeedback, runAnalysis, showHighlightedExplanation, uploadFeedback, usingPage } from "./batch-helpers";
import { SCREENSHOTS, THEMES, VIEWPORTS, settle } from "./helpers";

// Chụp ảnh trang 2 (Phân tích hàng loạt) và trang 3 (Dashboard & insight) ở 1440 và 390, sáng và tối, để
// đối chiếu với `docs/wireframes/screenshots/`. Ảnh nằm ở apps/web/test-results/visual/ (đã gitignore). Cần app
// đang chạy (E2E_BASE_URL) và API thật phía sau. Mọi file tải lên là file góp ý tự viết (feedback-file.ts) nên
// ảnh không có câu UIT-VSFC nào; riêng ảnh dashboard của dữ liệu demo (số liệu thật của tập test UIT-VSFC) có
// danh sách "Câu tiêu biểu" là câu của tập đó, nên vùng này được phủ khối màu xám trong ảnh.

// Mỗi kịch bản chạy thật một batch qua mô hình, nên cho thời gian rộng hơn mặc định 30 giây.
test.describe.configure({ timeout: 180_000 });

interface ShotOptions {
  /** Chụp cả trang (mặc định) hay chỉ khung nhìn, vd. ngăn giải thích là lớp phủ cố định. */
  fullPage?: boolean;
  /** Vùng được phủ khối màu xám để không lộ nội dung. */
  mask?: Locator[];
}

/** Cuộn lên đầu trước khi chụp: thanh bên dính theo khung nhìn nên chụp cả trang lúc đang cuộn sẽ lệch chỗ. */
async function shot(page: Page, name: string, { fullPage = true, mask }: ShotOptions = {}) {
  await settle(page);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: path.join(SCREENSHOTS, `${name}.png`), fullPage, mask, maskColor: "#a1a1aa" });
}

for (const viewport of VIEWPORTS) {
  for (const theme of THEMES) {
    const tag = `${viewport.label}-${theme}`;

    test(`trang 2 · phân tích hàng loạt · ${tag}`, async ({ browser }) => {
      await usingPage(browser, viewport, theme, async (page, problems) => {
        await page.goto("/batch/");
        await expect(page.getByRole("heading", { name: "1 · Tải file" })).toBeVisible();
        await shot(page, `02-phan-tich-hang-loat-trong-${tag}`);

        await uploadFeedback(page);
        await shot(page, `02-phan-tich-hang-loat-thiet-lap-${tag}`);

        await runAnalysis(page);
        await expect(page.getByRole("heading", { name: /Kết quả/ })).toBeVisible();
        await shot(page, `02-phan-tich-hang-loat-${tag}`);

        // Ngăn giải thích bên phải (chụp theo khung nhìn vì là lớp phủ cố định).
        await page.getByRole("button", { name: /^Giải thích câu/ }).first().click();
        await expect(page.getByRole("dialog")).toBeVisible();
        await expect(page.getByLabel("Đang tính giải thích")).toHaveCount(0, { timeout: 30_000 });
        await shot(page, `02-giai-thich-${tag}`, { fullPage: false });
        // Câu có từ được tô kèm bảng xác suất (trạng thái chính của ngăn, như trong wireframe).
        await showHighlightedExplanation(page);
        await shot(page, `02-giai-thich-co-tu-to-${tag}`, { fullPage: false });
        await page.keyboard.press("Escape");
        await expect(page.getByRole("dialog")).toHaveCount(0);

        // Sửa một nhãn rồi xem tab "Tất cả kết quả".
        await page.getByLabel(/^Sửa cảm xúc câu/).first().selectOption({ index: 1 });
        await expect(page.getByText(/^1 \/ \d+$/)).toBeVisible();
        await page.getByRole("tab", { name: /Tất cả kết quả/ }).click();
        await shot(page, `02-tat-ca-ket-qua-${tag}`);

        expect(problems).toEqual([]);
      });
    });

    test(`trang 2 · các trạng thái lỗi · ${tag}`, async ({ browser }) => {
      await usingPage(browser, viewport, theme, async (page, problems) => {
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
        const rows = Array.from({ length: 2100 }, (_, i) => `Thầy dạy dễ hiểu số ${i + 1}.`).join("\n");
        await page.locator('input[type="file"]').setInputFiles({
          name: "qua-nhieu-cau.csv",
          mimeType: "text/csv",
          buffer: Buffer.from(`Góp ý\n${rows}\n`, "utf-8"),
        });
        await expect(page.getByText(/vượt giới hạn/)).toBeVisible({ timeout: 30_000 });
        await expect(page.getByRole("button", { name: "Phân tích", exact: true })).toBeDisabled();
        await shot(page, `02-qua-gioi-han-${tag}`);

        // F5 sau khi có file: tab cũ đã đóng batch (pagehide) nên trang báo phiên đã kết thúc.
        await uploadFeedback(page);
        await page.reload();
        await expect(page.getByText("Phiên làm việc đã kết thúc.")).toBeVisible({ timeout: 30_000 });
        await shot(page, `02-phien-ket-thuc-${tag}`);

        // 422/404 là lỗi cố ý của các bước trên (phản hồi HTTP >= 400 được helpers.openPage ghi lại).
        expect(problems.filter((problem) => !/^http (404|422):/.test(problem))).toEqual([]);
      });
    });

    test(`trang 3 · dashboard & insight · ${tag}`, async ({ browser }) => {
      await usingPage(browser, viewport, theme, async (page, problems) => {
        // Mặc định: dữ liệu demo (số thật của tập test UIT-VSFC); che danh sách câu của tập đó.
        await page.goto("/dashboard/");
        await expect(page.getByRole("heading", { name: "Cảnh báo so với mức nền" })).toBeVisible({ timeout: 30_000 });
        await shot(page, `03-dashboard-demo-${tag}`, { mask: [page.getByRole("tabpanel").locator("ul")] });

        // File vừa tải: chạy file góp ý tự viết ở trang 2 rồi sang dashboard bằng liên kết (điều hướng trong ứng dụng).
        await page.goto("/batch/");
        await analyzeFeedback(page);
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

        expect(problems.filter((problem) => !/^http (404|422):/.test(problem))).toEqual([]);
      });
    });
  }
}
