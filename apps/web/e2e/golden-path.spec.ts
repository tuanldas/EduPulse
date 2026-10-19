import { expect, test } from "@playwright/test";

import { SENTENCES, analyze, expectNoHorizontalScroll, explainSettled, openPage, settle } from "./helpers";

// "Đường vàng" của trang 1 (Phân tích câu) và trang 4 (Mô hình & đạo đức), kèm điều hướng, theme, bàn phím
// và các lỗi thường gặp. Chạy trên bản export do FastAPI phục vụ CÙNG origin với /api (như khi triển khai):
//
//   cd apps/web && pnpm build                      # sinh out/
//   (API: make api, hoặc uvicorn với EDUPULSE_WEB_DIR trỏ tới apps/web/out)
//   E2E_BASE_URL=http://127.0.0.1:8000 pnpm e2e
//
// Mọi khẳng định không phụ thuộc vào số liệu của bundle (smoke hay release): chỉ kiểm cấu trúc, nhãn tiếng
// Việt, trạng thái và hành vi. Gây lỗi bằng `page.route` chỉ để mô phỏng sự cố mạng, không thay dữ liệu.

const DESKTOP = { width: 1440, height: 900 };
const MOBILE = { width: 390, height: 844 };

test.describe("đường vàng · trang 1 · phân tích câu", () => {
  test("nhãn hiện trước, giải thích tải sau, đổi tác vụ, xoá", async ({ browser }) => {
    const { context, page, problems } = await openPage(browser, DESKTOP, "light");

    // Giữ /api/explain lại để chứng minh nhãn không phải chờ giải thích.
    let releaseExplain: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      releaseExplain = resolve;
    });
    await page.route("**/api/explain", async (route) => {
      await gate;
      await route.continue();
    });

    await page.goto("/");
    await expect(page).toHaveTitle("Phân tích câu · EduPulse");
    await expect(page.locator("html")).toHaveAttribute("lang", "vi");
    const nav = page.getByRole("navigation", { name: "Điều hướng chính" });
    await expect(nav.getByRole("link")).toHaveCount(4);
    await expect(nav.getByRole("link", { name: "Phân tích câu" })).toHaveAttribute("aria-current", "page");
    await expect(page.getByText("Nhập một câu hoặc chọn câu mẫu ở trên để bắt đầu.")).toBeVisible();
    await expect(page.getByRole("switch", { name: /So sánh với mô hình cơ sở/ })).toHaveAttribute("aria-checked", "false");

    await analyze(page, SENTENCES.highlighted);
    // Nhãn và độ tin cậy đã hiện, còn giải thích đang chờ.
    await expect(page.getByRole("group", { name: "Cảm xúc" })).toBeVisible();
    await expect(page.getByRole("group", { name: "Chủ đề" })).toBeVisible();
    await expect(page.getByRole("progressbar").first()).toBeVisible();
    await expect(page.getByLabel("Đang tính giải thích")).toBeVisible();

    releaseExplain();
    await explainSettled(page);
    await expect(
      page.getByRole("table", { name: /Từ ảnh hưởng nhất/ }).or(page.getByText(/Không có từ nào ảnh hưởng đủ lớn/)),
    ).toBeVisible();

    // Đổi tác vụ giải thích.
    const sentimentTab = page.getByRole("button", { name: /^Cảm xúc ·/ });
    const topicTab = page.getByRole("button", { name: /^Chủ đề ·/ });
    await expect(sentimentTab).toHaveAttribute("aria-pressed", "true");
    await topicTab.click();
    await expect(topicTab).toHaveAttribute("aria-pressed", "true");
    await expect(sentimentTab).toHaveAttribute("aria-pressed", "false");

    // Xoá: ô nhập rỗng, về trạng thái đầu.
    await page.getByRole("button", { name: "Xoá" }).click();
    await expect(page.getByLabel("Câu góp ý")).toHaveValue("");
    await expect(page.getByRole("heading", { name: "Kết quả" })).toHaveCount(0);
    await expect(page.getByText("Nhập một câu hoặc chọn câu mẫu ở trên để bắt đầu.")).toBeVisible();
    expect(problems).toEqual([]);
    await context.close();
  });

  test("chọn câu mẫu thì điền vào ô nhập và chạy luôn", async ({ browser }) => {
    const { context, page } = await openPage(browser, DESKTOP, "light");
    await page.goto("/");
    await expect(
      page.getByRole("group", { name: /Hoặc thử câu mẫu/ }).or(page.getByText("Dữ liệu demo chưa sẵn")),
    ).toBeVisible();
    const chips = page.getByRole("group", { name: /Hoặc thử câu mẫu/ }).getByRole("button");
    test.skip((await chips.count()) === 0, "không có câu mẫu: dữ liệu demo chưa được dựng");

    const chip = chips.first();
    const sample = (await chip.getAttribute("title")) ?? "";
    await chip.click();
    await expect(page.getByLabel("Câu góp ý")).toHaveValue(sample);
    await expect(chip).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("heading", { name: "Kết quả" })).toBeVisible();
    await context.close();
  });

  test("phím tắt Ctrl/⌘ + Enter chạy phân tích", async ({ browser }) => {
    const { context, page } = await openPage(browser, DESKTOP, "light");
    await page.goto("/");
    const box = page.getByLabel("Câu góp ý");
    await box.fill(SENTENCES.plain);
    await box.press("ControlOrMeta+Enter");
    await expect(page.getByRole("heading", { name: "Kết quả" })).toBeVisible();
    await context.close();
  });

  test("các trạng thái: không dấu, thông tin cá nhân, câu dài, không còn nội dung", async ({ browser }) => {
    const { context, page, problems } = await openPage(browser, DESKTOP, "light");
    const explainRequests: string[] = [];
    page.on("request", (request) => {
      if (request.url().endsWith("/api/explain")) explainRequests.push(request.postData() ?? "");
    });
    await page.goto("/");

    await analyze(page, SENTENCES.noDiacritics);
    await expect(page.getByText(/Câu có vẻ được gõ không dấu/)).toBeVisible();
    await expect(page.getByText("Cần xem xét", { exact: true })).toBeVisible();
    await expect(page.getByText("Câu không dấu", { exact: true })).toBeVisible();
    await explainSettled(page);

    await analyze(page, SENTENCES.withPhone);
    await expect(page.getByText(/Đã ẩn 1 thông tin cá nhân/)).toBeVisible();
    await expect(page.getByText("[SĐT]").first()).toBeVisible();
    await explainSettled(page);

    // Câu dài: vẫn có kết quả, báo không tô từ và KHÔNG gọi /api/explain cho câu này.
    const before = explainRequests.length;
    await analyze(page, SENTENCES.long);
    await expect(page.getByText(/nên không tô từ để giữ tốc độ/)).toBeVisible();
    await page.waitForTimeout(500);
    expect(explainRequests.length).toBe(before);

    await page.getByLabel("Câu góp ý").fill(SENTENCES.onlyPhone);
    await page.getByRole("button", { name: "Phân tích" }).click();
    await expect(page.getByText("Không còn nội dung để phân tích.")).toBeVisible();
    // Câu chỉ có số điện thoại cố ý nhận 422 `no_content`.
    expect(problems.filter((problem) => problem !== "http 422: /api/analyze")).toEqual([]);
    await context.close();
  });

  test("so sánh mô hình mặc định tắt; bật thì hiện khung so sánh", async ({ browser }) => {
    const { context, page, problems } = await openPage(browser, DESKTOP, "light");
    await page.goto("/");
    const toggle = page.getByRole("switch", { name: /So sánh với mô hình cơ sở/ });
    await expect(toggle).toHaveAttribute("aria-checked", "false");
    await analyze(page, SENTENCES.plain);
    await explainSettled(page);
    await expect(page.getByRole("region", { name: "So sánh mô hình" })).toHaveCount(0);

    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-checked", "true");
    await expect(page.getByRole("region", { name: "So sánh mô hình" })).toBeVisible();
    await expect(page.getByLabel("Đang tính mô hình cơ sở")).toHaveCount(0, { timeout: 30_000 });
    // Hoặc đã có kết quả mô hình cơ sở, hoặc máy chủ báo chưa hỗ trợ (501); kết quả chính vẫn còn.
    await expect(
      page.getByText("Mô hình cơ sở", { exact: true }).or(page.getByText(/Tính năng này chưa sẵn sàng/)),
    ).toBeVisible();
    await expect(page.getByRole("heading", { name: "Kết quả" })).toBeVisible();

    await toggle.click();
    await expect(page.getByRole("region", { name: "So sánh mô hình" })).toHaveCount(0);
    expect(problems.filter((problem) => problem !== "http 501: /api/analyze")).toEqual([]);
    await context.close();
  });

  test("lỗi mạng khi phân tích: báo bằng tiếng Việt và thử lại được", async ({ browser }) => {
    const { context, page } = await openPage(browser, DESKTOP, "light");
    await page.route("**/api/analyze", (route) => route.abort("failed"));
    await page.goto("/");
    await page.getByLabel("Câu góp ý").fill(SENTENCES.plain);
    await page.getByRole("button", { name: "Phân tích" }).click();
    await expect(page.getByRole("alert").filter({ hasText: "Không kết nối được máy chủ" })).toBeVisible();

    await page.unroute("**/api/analyze");
    await page.getByRole("button", { name: /Thử lại/ }).click();
    await expect(page.getByRole("heading", { name: "Kết quả" })).toBeVisible();
    await context.close();
  });
});

test.describe("đường vàng · khung ứng dụng", () => {
  test("công tắc giao diện nhớ lựa chọn sau khi tải lại", async ({ browser }) => {
    const { context, page } = await openPage(browser, DESKTOP, "light");
    await page.goto("/");
    const html = page.locator("html");
    await expect(html).not.toHaveClass(/dark/);

    await page.getByRole("button", { name: "Tối" }).click();
    await expect(html).toHaveClass(/dark/);
    await page.reload();
    await expect(html).toHaveClass(/dark/);
    await expect(page.getByRole("button", { name: "Tối" })).toHaveAttribute("aria-pressed", "true");

    await page.getByRole("button", { name: "Sáng" }).click();
    await expect(html).not.toHaveClass(/dark/);
    await page.getByRole("button", { name: "Theo hệ thống" }).click();
    await expect(page.getByRole("button", { name: "Theo hệ thống" })).toHaveAttribute("aria-pressed", "true");
    await context.close();
  });

  test("bàn phím: 'Bỏ qua điều hướng' đưa tiêu điểm vào nội dung chính", async ({ browser }) => {
    const { context, page } = await openPage(browser, DESKTOP, "light");
    await page.goto("/");
    await page.keyboard.press("Tab");
    const skip = page.getByRole("link", { name: "Bỏ qua điều hướng" });
    await expect(skip).toBeFocused();
    await expect(skip).toBeVisible();
    await page.keyboard.press("Enter");
    await expect(page.locator("#noi-dung")).toBeFocused();
    await context.close();
  });

  test("bàn phím: điều khiển phân đoạn chọn được bằng phím", async ({ browser }) => {
    const { context, page } = await openPage(browser, DESKTOP, "light");
    await page.goto("/");
    await analyze(page, SENTENCES.highlighted);
    await explainSettled(page);
    const sentimentTab = page.getByRole("button", { name: /^Cảm xúc ·/ });
    const topicTab = page.getByRole("button", { name: /^Chủ đề ·/ });
    await sentimentTab.focus();
    await page.keyboard.press("ArrowRight");
    await expect(topicTab).toBeFocused();
    await page.keyboard.press("Space");
    await expect(topicTab).toHaveAttribute("aria-pressed", "true");
    await context.close();
  });

  test("trang không có: 404 tiếng Việt và đường về trang chính", async ({ browser }) => {
    const { context, page } = await openPage(browser, DESKTOP, "light");
    const response = await page.goto("/khong-co-trang-nay/");
    expect(response?.status()).toBe(404);
    await expect(page.getByRole("heading", { name: "Không tìm thấy trang" })).toBeVisible();
    await page.getByRole("link", { name: "Về trang Phân tích câu", exact: true }).click();
    await expect(page).toHaveURL(/\/$/);
    await expect(page.getByRole("heading", { name: "Phân tích câu", level: 1 })).toBeVisible();
    await context.close();
  });

  test("di động: ngăn điều hướng mở, đóng bằng Esc trả tiêu điểm, chọn trang thì tự đóng", async ({ browser }) => {
    const { context, page, problems } = await openPage(browser, MOBILE, "light");
    await page.goto("/");
    const menu = page.getByRole("button", { name: "Mở điều hướng" });
    await menu.click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    await expect(menu).toBeFocused();

    await menu.click();
    await dialog.getByRole("link", { name: "Mô hình & đạo đức" }).click();
    await expect(page).toHaveURL(/\/model\/$/);
    await expect(page.getByRole("dialog")).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "Mô hình & đạo đức", level: 1 })).toBeVisible();
    await expectNoHorizontalScroll(page);
    expect(problems).toEqual([]);
    await context.close();
  });
});

test.describe("đường vàng · trang 4 · mô hình & đạo đức", () => {
  test("từ thanh điều hướng: chỉ số, đổi tác vụ, bảng số, mục lục neo, sao chép trích dẫn", async ({ browser }) => {
    const { context, page } = await openPage(browser, DESKTOP, "light", {
      permissions: ["clipboard-read", "clipboard-write"],
    });
    await page.goto("/");
    await page.getByRole("navigation", { name: "Điều hướng chính" }).getByRole("link", { name: "Mô hình & đạo đức" }).click();
    await expect(page).toHaveURL(/\/model\/$/);
    await expect(page).toHaveTitle("Mô hình & đạo đức · EduPulse");
    await expect(page.getByRole("link", { name: "Mô hình & đạo đức" }).first()).toHaveAttribute("aria-current", "page");
    await expect(page.getByRole("heading", { name: "Mô hình & đạo đức", level: 1 })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Model card" })).toBeVisible({ timeout: 20_000 });

    // Bốn chỉ số của cảm xúc, rồi chủ đề.
    for (const label of ["Accuracy", "Macro-F1", "ECE (sau hiệu chỉnh)"]) {
      await expect(page.locator("dt", { hasText: label }).first()).toBeVisible();
    }
    await expect(page.getByRole("table", { name: /Chỉ số từng lớp/ })).toContainText("Tiêu cực");
    await page.getByRole("button", { name: "Chủ đề" }).click();
    await expect(page.getByRole("table", { name: /Chỉ số từng lớp/ })).toContainText("Giảng viên");
    await expect(page.getByRole("table", { name: /Ma trận nhầm lẫn/ })).toContainText("Cơ sở vật chất");

    // Bảng số của biểu đồ độ tin cậy.
    await page.getByText("Xem bảng số").click();
    await expect(page.getByRole("table", { name: /Độ tin cậy báo so với tỉ lệ đúng thực tế/ })).toBeVisible();

    // Mục lục neo: bấm thì cuộn tới đúng mục.
    await page.getByRole("navigation", { name: "Mục lục" }).getByRole("link", { name: "Giấy phép & trích dẫn" }).click();
    await expect(page).toHaveURL(/#giay-phep$/);
    await expect(page.getByRole("heading", { name: /Giấy phép, mã nguồn/ })).toBeInViewport();

    // Sao chép BibTeX.
    await page.getByRole("button", { name: "Sao chép trích dẫn BibTeX" }).click();
    const copied = await page.evaluate(() => navigator.clipboard.readText());
    expect(copied).toContain("@inproceedings{nguyen2018uitvsfc");
    expect(copied).toContain("10.18653/v1/2020.findings-emnlp.92");
    await expect(page.getByText("Đã sao chép trích dẫn BibTeX.")).toBeVisible();
    await context.close();
  });

  test("mở thẳng /model (không dấu / cuối) và liên kết neo vẫn tới đúng mục sau khi dữ liệu nạp", async ({ browser }) => {
    const { context, page, problems } = await openPage(browser, DESKTOP, "light");
    await page.goto("/model");
    await expect(page).toHaveURL(/\/model\/$/);
    await page.goto("/model/#giay-phep");
    await expect(page.getByRole("heading", { name: "Model card" })).toBeVisible({ timeout: 20_000 });
    await expect(page.getByRole("heading", { name: /Giấy phép, mã nguồn/ })).toBeInViewport();
    expect(problems).toEqual([]);
    await context.close();
  });

  test("nút mã nguồn mở liên kết ngoài an toàn khi đã cấu hình source_url", async ({ browser }) => {
    const { context, page } = await openPage(browser, DESKTOP, "light");
    await page.goto("/model/");
    await settle(page);
    const config = await (await page.request.get("/api/config")).json();
    const link = page.getByRole("link", { name: /^Mã nguồn( trên GitHub)?$/ });
    if (config.source_url) {
      await expect(link).toHaveAttribute("href", config.source_url);
      await expect(link).toHaveAttribute("target", "_blank");
      await expect(link).toHaveAttribute("rel", /noopener/);
    } else {
      await expect(link).toHaveCount(0);
      await expect(page.getByText(/Liên kết mã nguồn chưa được cấu hình/)).toBeVisible();
    }
    await context.close();
  });
});

test.describe("đường vàng · co giãn (WCAG 1.4.10)", () => {
  // 320px = chuẩn reflow; 720px = trình duyệt 1440 phóng 200%; 1024px = máy tính bảng ngang.
  for (const width of [320, 720, 1024]) {
    test(`không cuộn ngang ở ${width}px: trang 1 có kết quả và trang 4`, async ({ browser }) => {
      const { context, page } = await openPage(browser, { width, height: 800 }, "light");
      await page.goto("/");
      await analyze(page, SENTENCES.highlighted);
      await explainSettled(page);
      await expectNoHorizontalScroll(page);

      await page.goto("/model/");
      await expect(page.getByRole("heading", { name: "Model card" })).toBeVisible({ timeout: 20_000 });
      await expectNoHorizontalScroll(page);
      await context.close();
    });
  }
});
