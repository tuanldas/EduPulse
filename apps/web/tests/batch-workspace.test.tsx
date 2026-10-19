import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { BatchWorkspace } from "@/components/batch/batch-workspace";
import type { AppConfig } from "@/lib/api/types";
import { resetAppConfig } from "@/lib/app-config";
import { discardReviewEdits } from "@/lib/batch/review-edits";
import { clearStoredBatch, saveStoredBatch, storedBatchFromUpload } from "@/lib/batch/stored-batch";
import type { BatchStatus } from "@/lib/batch/types";

import { callsTo, fixture, json, stubApi, type Handler } from "./helpers/api-stub";
import { batchResults, batchStatusDone, batchUpload } from "./helpers/batch-fixtures";

// Chạy cả trang 2 (useBatchFlow + các bước) với phản hồi API đã ghi từ API thật: tải file mẫu, chạy, theo dõi,
// kết quả, khoá thiết lập, phiên kết thúc, khôi phục sau khi chuyển trang.

const ID = "fixture-batch";
const uploaded: BatchStatus = { ...batchStatusDone, state: "uploaded", step: null, done: 0, progress: 0, eta_s: null };
const queued: BatchStatus = { ...uploaded, state: "queued" };

function routes(overrides: Record<string, Handler> = {}) {
  let started = false;
  const table: Record<string, Handler> = {
    "GET /api/config": () => json(fixture<AppConfig>("config.json")),
    "POST /api/batches/sample": () => json(batchUpload, 201),
    [`POST /api/batches/${ID}/run`]: () => {
      started = true;
      return json(queued, 202);
    },
    [`GET /api/batches/${ID}`]: () => json(started ? batchStatusDone : uploaded),
    [`GET /api/batches/${ID}/results`]: () => json(batchResults),
    [`POST /api/batches/${ID}/heartbeat`]: () => new Response(null, { status: 204 }),
    [`DELETE /api/batches/${ID}`]: () => new Response(null, { status: 204 }),
    ...overrides,
  };
  stubApi(table);
}

async function runSample() {
  await userEvent.click(await screen.findByRole("button", { name: /Dùng file mẫu/ }));
  await userEvent.click(await screen.findByRole("button", { name: /^Phân tích 20 câu$/ }));
  await screen.findByText("Đã phân tích xong.", undefined, { timeout: 5000 });
}

beforeEach(() => {
  resetAppConfig();
  clearStoredBatch();
  discardReviewEdits(ID);
  window.sessionStorage.clear();
});

describe("BatchWorkspace", () => {
  it("file mẫu → thiết lập mặc định → chạy → kết quả, gửi đúng cột và ngân sách", async () => {
    routes();
    render(<BatchWorkspace />);
    await runSample();

    expect(callsTo(`POST /api/batches/${ID}/run`)[0].body).toEqual({
      text_col: "Góp ý",
      group_col: "Nhóm (minh hoạ)",
      review_budget: 10,
    });
    expect(await screen.findByRole("heading", { name: "3 · Kết quả & duyệt" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Cần duyệt \(5\)/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "4 · Xuất kết quả" })).toBeInTheDocument();
    // Dấu gạch chéo cuối do `trailingSlash` của bản export thêm vào; môi trường test không có cấu hình đó.
    expect(screen.getByRole("link", { name: /Xem dashboard cho file này/ }).getAttribute("href")).toMatch(
      new RegExp(`^/dashboard/?\\?batch=${ID}$`),
    );
  });

  it("chạy xong thì khoá thiết lập; Bắt đầu lại xoá batch trên máy chủ và về bước 1", async () => {
    routes();
    render(<BatchWorkspace />);
    await runSample();
    expect(screen.getByLabelText("Cột văn bản")).toBeDisabled();
    expect(screen.getByLabelText(/Cột nhóm/)).toBeDisabled();
    // Thanh trượt của Base UI chỉ hiện nút kéo khi đã đo bố cục (jsdom không có), nên kiểm trên phần tử gốc.
    expect(document.querySelector('[data-slot="slider"]')).toHaveAttribute("data-disabled");

    await userEvent.click(screen.getByRole("button", { name: /Bắt đầu lại/ }));
    await waitFor(() => expect(callsTo(`DELETE /api/batches/${ID}`)).toHaveLength(1));
    expect(screen.queryByRole("heading", { name: "2 · Thiết lập" })).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /Kết quả/ })).not.toBeInTheDocument();
    expect(window.sessionStorage.getItem("edupulse.batch")).toBeNull();
  });

  it("sửa nhãn được đếm, gắn đúng chỉ số dòng và bật cảnh báo trước khi rời trang", async () => {
    routes();
    render(<BatchWorkspace />);
    await runSample();

    const first = batchResults.rows.find((row) => row.index === batchResults.review_order[0])!;
    expect(first.sentiment).toBe("negative");
    await userEvent.selectOptions(screen.getByLabelText(`Sửa cảm xúc câu ${first.source_row}`), "positive");
    expect(screen.getByText("Đã sửa nhãn").closest("dl")).toHaveTextContent("1 / 5");
    expect(JSON.parse(window.sessionStorage.getItem(`edupulse.edits.${ID}`) ?? "{}").overrides).toEqual({
      [first.index]: { sentiment: "positive" },
    });
    expect(screen.getByRole("button", { name: /Tải nhãn đã sửa · 1 dòng/ })).toBeEnabled();

    const leave = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(leave);
    expect(leave.defaultPrevented).toBe(true);
  });

  it("máy chủ báo batch không còn khi hỏi tiến độ thì hiện phiên đã kết thúc", async () => {
    routes({
      [`GET /api/batches/${ID}`]: () => json({ detail: { code: "batch_not_found", message: "x" } }, 404),
    });
    render(<BatchWorkspace />);
    await userEvent.click(await screen.findByRole("button", { name: /Dùng file mẫu/ }));
    expect(await screen.findByText("Phiên làm việc đã kết thúc.")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "2 · Thiết lập" })).not.toBeInTheDocument();
  });

  it("máy chủ bận khi tạo file mẫu thì báo bằng tiếng Việt và vẫn dùng được trang", async () => {
    routes({ "POST /api/batches/sample": () => json({ detail: { code: "busy", message: "x" } }, 429) });
    render(<BatchWorkspace />);
    await userEvent.click(await screen.findByRole("button", { name: /Dùng file mẫu/ }));
    expect(await screen.findByText("Máy chủ đang bận, thử lại sau ít phút.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Dùng file mẫu/ })).toBeEnabled();
  });

  it("không có dữ liệu demo thì thay nút file mẫu bằng thông báo", async () => {
    routes({ "GET /api/config": () => json({ ...fixture<AppConfig>("config.json"), demo_available: false }) });
    render(<BatchWorkspace />);
    expect(await screen.findByText("Dữ liệu demo chưa sẵn")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Dùng file mẫu/ })).not.toBeInTheDocument();
  });

  it("quay lại trang sau khi sang dashboard: dựng lại kết quả từ batch đã lưu trong tab", async () => {
    routes({ [`GET /api/batches/${ID}`]: () => json(batchStatusDone) });
    saveStoredBatch({
      ...storedBatchFromUpload(batchUpload),
      run: { textCol: "Góp ý", groupCol: "Nhóm (minh hoạ)", budget: 20 },
    });
    render(<BatchWorkspace />);
    expect(await screen.findByRole("heading", { name: "3 · Kết quả & duyệt" }, { timeout: 5000 })).toBeInTheDocument();
    // Thiết lập đã dùng được nhớ lại và khoá.
    expect(screen.getByLabelText("Cột văn bản")).toHaveValue("Góp ý");
    expect(screen.getByLabelText("Cột văn bản")).toBeDisabled();
    expect(within(screen.getByRole("table")).getAllByRole("row").length).toBeGreaterThan(1);
  });

  it("batch đã lưu mà máy chủ không còn (F5 sau khi tab cũ đóng) thì báo phiên đã kết thúc và cho tải lại", async () => {
    routes({
      [`GET /api/batches/${ID}`]: () => json({ detail: { code: "batch_not_found", message: "x" } }, 404),
    });
    saveStoredBatch({
      ...storedBatchFromUpload(batchUpload),
      run: { textCol: "Góp ý", groupCol: null, budget: 10 },
    });
    render(<BatchWorkspace />);
    expect(await screen.findByText("Phiên làm việc đã kết thúc.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Bắt đầu lại/ }));
    expect(screen.queryByText("Phiên làm việc đã kết thúc.")).not.toBeInTheDocument();
  });

  it("job thất bại thì báo lỗi và cho chạy lại với đúng thiết lập đã chọn", async () => {
    let attempts = 0;
    const failed: BatchStatus = { ...uploaded, state: "failed", error_code: "internal_error" };
    routes({
      [`POST /api/batches/${ID}/run`]: () => {
        attempts += 1;
        return json(queued, 202);
      },
      [`GET /api/batches/${ID}`]: () => json(attempts === 0 ? uploaded : attempts === 1 ? failed : batchStatusDone),
    });
    render(<BatchWorkspace />);
    await userEvent.click(await screen.findByRole("button", { name: /Dùng file mẫu/ }));
    await userEvent.click(await screen.findByRole("button", { name: /^Phân tích 20 câu$/ }));
    expect(await screen.findByText(/Phân tích không thành công\. Hãy thử lại/, undefined, { timeout: 5000 })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Thử lại" }));
    await screen.findByText("Đã phân tích xong.", undefined, { timeout: 5000 });
    expect(callsTo(`POST /api/batches/${ID}/run`)).toHaveLength(2);
    expect(callsTo(`POST /api/batches/${ID}/run`)[1].body).toEqual(callsTo(`POST /api/batches/${ID}/run`)[0].body);
  });
});
