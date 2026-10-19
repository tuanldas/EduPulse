import path from "node:path";

import AxeBuilder from "@axe-core/playwright";
import { expect, type Browser, type BrowserContextOptions, type Page } from "@playwright/test";

// Dùng chung cho các spec E2E của trang 1 và 4. Spec của trang 2–3 có thể dùng lại.

export const SCREENSHOTS = path.join("test-results", "visual");

export const VIEWPORTS = [
  { label: "1440", width: 1440, height: 900 },
  { label: "390", width: 390, height: 844 },
] as const;
export const THEMES = ["light", "dark"] as const;
export type Viewport = (typeof VIEWPORTS)[number];
export type Theme = (typeof THEMES)[number];

// Câu tự viết (tests/fixtures/golden_sentences.json ở thư mục gốc), không phải câu UIT-VSFC.
export const SENTENCES = {
  /** Có 3 từ ủng hộ và 1 từ phản đối ở đầu ra cảm xúc. */
  highlighted: "Thư viện mới rất rộng rãi, có máy lạnh và ổ cắm điện ở mọi chỗ ngồi.",
  plain: "Cô dạy quá nhanh, em không kịp ghi chép và rất ít khi được hỏi lại.",
  noDiacritics: "thay day rat de hieu va nhiet tinh voi sinh vien",
  withPhone: "Thầy giảng hay, em xin liên hệ 0912345678 để hỏi thêm về bài tập.",
  long: Array.from({ length: 6 }, () => "[đã gỡ câu UIT-VSFC] thử nghiệm ngay những gì vừa học.").join(" "),
  onlyPhone: "0912345678",
} as const;

/** Mở một ngữ cảnh trình duyệt với khung nhìn và theme cho trước; gom lỗi trang/console/HTTP để test kiểm sau. */
export async function openPage(
  browser: Browser,
  viewport: { width: number; height: number },
  theme: Theme,
  options: BrowserContextOptions = {},
) {
  const context = await browser.newContext({
    viewport: { width: viewport.width, height: viewport.height },
    colorScheme: theme,
    locale: "vi-VN",
    timezoneId: "Asia/Ho_Chi_Minh",
    ...options,
  });
  const page = await context.newPage();
  const problems: string[] = [];
  page.on("pageerror", (error) => problems.push(`pageerror: ${error.message}`));
  // Dòng "Failed to load resource" của trình duyệt không nói địa chỉ nào hỏng, nên ghi theo phản hồi HTTP.
  page.on("console", (message) => {
    if (message.type() === "error" && !message.text().startsWith("Failed to load resource")) {
      problems.push(`console: ${message.text()}`);
    }
  });
  page.on("response", (response) => {
    // Next tải trước (HEAD) các trang đích của liên kết điều hướng; mã nguồn của app không gửi HEAD nên bỏ qua loại này.
    if (response.request().method() === "HEAD") return;
    if (response.status() >= 400) problems.push(`http ${response.status()}: ${new URL(response.url()).pathname}`);
  });
  return { context, page, problems };
}

/** Đợi mạng yên và font nạp xong; ẩn nút chỉ báo của `next dev` (bản export không có). */
export async function settle(page: Page) {
  await page.waitForLoadState("networkidle");
  await page.evaluate(() => document.fonts.ready);
  await page.addStyleTag({ content: "nextjs-portal { display: none !important; }" });
}

/** Gõ câu vào ô nhập rồi bấm "Phân tích"; đợi tiêu đề "Kết quả". */
export async function analyze(page: Page, text: string) {
  await page.getByLabel("Câu góp ý").fill(text);
  await page.getByRole("button", { name: "Phân tích" }).click();
  await expect(page.getByRole("heading", { name: "Kết quả" })).toBeVisible();
}

/** Đợi phần giải thích tính xong (có bảng, hoặc báo không có từ nào để tô, hoặc câu quá dài). */
export async function explainSettled(page: Page) {
  await expect(page.getByLabel("Đang tính giải thích")).toHaveCount(0, { timeout: 30_000 });
}

/** Thiết kế cho di động: không cuộn ngang cả trang (bảng rộng chỉ cuộn trong khung của nó). */
export async function expectNoHorizontalScroll(page: Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow, "trang bị cuộn ngang").toBeLessThanOrEqual(1);
}

/** Đợi các hiệu ứng chuyển/animation hữu hạn chạy xong (vd. ngăn điều hướng đang mờ dần hiện ra làm axe đo sai tương phản). */
export async function waitForAnimations(page: Page) {
  await page.evaluate(() =>
    Promise.all(
      document
        .getAnimations()
        .filter((animation) => animation.effect?.getComputedTiming().iterations !== Infinity)
        .map((animation) => animation.finished.catch(() => undefined)),
    ),
  );
}

/** axe-core theo WCAG 2.1 AA (+ thực hành tốt): không được có vi phạm nào. */
export async function expectNoA11yViolations(page: Page, label: string) {
  await waitForAnimations(page);
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"])
    .analyze();
  const summary = results.violations.map((violation) => {
    const first = violation.nodes[0];
    const data = first.any[0]?.data as Record<string, unknown> | undefined;
    const detail = data && "contrastRatio" in data ? ` (tương phản ${data.contrastRatio}, chữ ${data.fgColor} trên ${data.bgColor})` : "";
    return `${violation.id} (${violation.impact}): ${violation.help} — ${violation.nodes.length} nút, vd ${first.target.join(" ")}${detail} ${first.html.slice(0, 160)}`;
  });
  expect(summary, `${label}: vi phạm trợ năng`).toEqual([]);
}
