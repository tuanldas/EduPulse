import { readFile } from "node:fs/promises";

import { expect, test, type Download, type Page } from "@playwright/test";

import { analyzeFeedback, FEEDBACK, leavePage } from "./batch-helpers";
import { SENTENCES } from "./helpers";

// Xuất CSV trên trình duyệt thật: file toàn bộ kết quả, file chỉ nhãn đã sửa, và việc tải xong làm hết cảnh
// báo "còn nhãn sửa chưa xuất". Cần chạy trên bản export do uvicorn phục vụ cùng origin với /api.

test.describe.configure({ timeout: 120_000 });

const BOM = Buffer.from([0xef, 0xbb, 0xbf]);
const PHONE = SENTENCES.onlyPhone;

// Máy chủ giữ tối đa 8 batch: rời trang để pagehide xoá batch của test vừa chạy.
test.afterEach(async ({ page }) => {
  await leavePage(page);
});

async function analyzeFile(page: Page) {
  await page.goto("/batch/");
  await analyzeFeedback(page);
}

async function readCsv(download: Download): Promise<{ bytes: Buffer; lines: string[] }> {
  const path = await download.path();
  const bytes = await readFile(path);
  const lines = bytes.subarray(3).toString("utf-8").split("\r\n");
  // Dòng cuối rỗng do CSV kết thúc bằng CRLF.
  if (lines.at(-1) === "") lines.pop();
  return { bytes, lines };
}

/** Tách một dòng CSV (RFC 4180, không có xuống dòng trong ô): câu góp ý có dấu phẩy nên được đặt trong ngoặc kép. */
function splitCsvLine(line: string): string[] {
  const cells: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (quoted) {
      if (char === '"' && line[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        cell += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      cells.push(cell);
      cell = "";
    } else {
      cell += char;
    }
  }
  cells.push(cell);
  return cells;
}

test("tải toàn bộ kết quả: CSV UTF-8 có BOM, đủ dòng, chỉ có câu đã ẩn thông tin cá nhân", async ({ page }) => {
  await analyzeFile(page);
  const total = FEEDBACK.rows;

  const downloading = page.waitForEvent("download");
  await page.getByRole("button", { name: "Tải toàn bộ kết quả (CSV)" }).click();
  const download = await downloading;
  expect(download.suggestedFilename()).toMatch(/_ket-qua\.csv$/);

  const { bytes, lines } = await readCsv(download);
  expect(bytes.subarray(0, 3).equals(BOM)).toBe(true);
  expect(lines[0]).toContain("Câu (đã ẩn thông tin cá nhân)");
  expect(lines).toHaveLength(total + 1);
  // File tải lên có hai dòng chứa số điện thoại (một dòng chỉ có số): bản xuất không được còn số đó.
  const body = lines.join("\n");
  expect(body).not.toContain(PHONE);
  expect(body).toContain("[SĐT]");
  await expect(page.getByText(`Đã tạo file CSV · ${total} dòng`)).toBeVisible();
});

test("nhãn đã sửa: file riêng chỉ có dòng bị sửa và việc xuất làm hết cảnh báo khi rời trang", async ({ page }) => {
  await analyzeFile(page);

  const select = page.getByLabel(/^Sửa cảm xúc câu/).first();
  const chosen = await select.locator("option").nth(1).getAttribute("value");
  expect(chosen).not.toBeNull();
  await select.selectOption(chosen ?? "");
  const exportCorrected = page.getByRole("button", { name: /Tải nhãn đã sửa · 1 dòng/ });
  await expect(exportCorrected).toBeEnabled();
  await expect(page.getByText(/Còn 1 câu có nhãn đã sửa chưa tải xuống/)).toBeVisible();

  const downloading = page.waitForEvent("download");
  await exportCorrected.click();
  const download = await downloading;
  expect(download.suggestedFilename()).toMatch(/_nhan-da-sua\.csv$/);

  const { bytes, lines } = await readCsv(download);
  expect(bytes.subarray(0, 3).equals(BOM)).toBe(true);
  expect(lines).toHaveLength(2);
  const header = splitCsvLine(lines[0]);
  const row = splitCsvLine(lines[1]);
  // Nhãn cuối cùng là nhãn người duyệt chọn; dự đoán gốc của mô hình vẫn được giữ lại để đo vòng duyệt.
  expect(row[header.indexOf("label_sentiment")]).toBe(chosen);
  expect(row[header.indexOf("pred_sentiment")]).not.toBe(chosen);

  // Đã xuất: hết cảnh báo, rời trang không bị hỏi.
  await expect(page.getByText(/chưa tải xuống/)).toHaveCount(0);
  const dialogs: string[] = [];
  page.on("dialog", (dialog) => {
    dialogs.push(dialog.type());
    void dialog.dismiss();
  });
  await page.goto("about:blank");
  expect(dialogs).toEqual([]);
});
