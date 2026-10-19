import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DashboardWorkspace } from "@/components/dashboard/dashboard-workspace";
import type { AppConfig } from "@/lib/api/types";
import { resetAppConfig } from "@/lib/app-config";
import { discardReviewEdits } from "@/lib/batch/review-edits";
import { clearStoredBatch, saveStoredBatch, storedBatchFromUpload } from "@/lib/batch/stored-batch";

import { callsTo, fixture, json, stubApi, type Handler } from "./helpers/api-stub";
import { batchInsights, batchInsightsFiltered, batchUpload } from "./helpers/batch-fixtures";

// Hook điều hướng của Next cần ngữ cảnh router của ứng dụng; test thay bằng bản đọc ?batch= từ biến cục bộ.
const navigation = vi.hoisted(() => ({ search: "", replace: vi.fn() }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: navigation.replace }),
  useSearchParams: () => new URLSearchParams(navigation.search),
}));

const ID = "fixture-batch";
const demoInsights = {
  ...batchInsights,
  source: "demo" as const,
  illustrative_groups: false,
  groups: null,
  heatmap: null,
  kpis: { ...batchInsights.kpis, n_rows: 3166, n_groups: null, n_flagged: null, flagged_share: null },
};

function routes(overrides: Record<string, Handler> = {}, config: Partial<AppConfig> = {}) {
  stubApi({
    "GET /api/config": () => json({ ...fixture<AppConfig>("config.json"), ...config }),
    "POST /api/batches/demo/insights": () => json(demoInsights),
    [`POST /api/batches/${ID}/insights`]: async (request) => {
      const body = (await request.clone().json()) as { groups: string[] | null };
      return json(body.groups === null ? batchInsights : batchInsightsFiltered);
    },
    [`POST /api/batches/${ID}/heartbeat`]: () => new Response(null, { status: 204 }),
    ...overrides,
  });
}

function storeFile() {
  saveStoredBatch({
    ...storedBatchFromUpload(batchUpload),
    run: { textCol: "Góp ý", groupCol: "Nhóm (minh hoạ)", budget: 20 },
  });
}

beforeEach(() => {
  navigation.search = "";
  navigation.replace.mockClear();
  resetAppConfig();
  clearStoredBatch();
  discardReviewEdits(ID);
  window.sessionStorage.clear();
});

describe("DashboardWorkspace", () => {
  it("mặc định xem dữ liệu demo, không cần file nào", async () => {
    routes();
    render(<DashboardWorkspace />);
    expect(await screen.findByRole("heading", { name: "Cảnh báo so với mức nền" })).toBeInTheDocument();
    expect(callsTo("POST /api/batches/demo/insights")[0].body).toEqual({ groups: null });
    // Demo không có nhóm: không có bộ lọc nhóm và có chú thích lý do không có heatmap.
    expect(screen.queryByText(/Lọc theo/)).not.toBeInTheDocument();
    expect(screen.getByText(/Heatmap nhóm × chủ đề chỉ hiện khi file có cột nhóm/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Dữ liệu demo · UIT-VSFC \(3\.166 câu\)/ })).toBeInTheDocument();
    // Không có phiên cần giữ khi xem demo.
    expect(callsTo(`POST /api/batches/${ID}/heartbeat`)).toHaveLength(0);
  });

  it("?batch= chọn file vừa tải: có bộ lọc nhóm và heatmap, gọi đúng batch", async () => {
    navigation.search = `batch=${ID}`;
    routes();
    storeFile();
    render(<DashboardWorkspace />);
    expect(await screen.findByRole("heading", { name: "Tiêu cực theo Nhóm (minh hoạ) × Chủ đề" })).toBeInTheDocument();
    expect(callsTo(`POST /api/batches/${ID}/insights`)[0].body).toEqual({ groups: null });
    expect(screen.getByRole("button", { name: /Tất cả 5 nhóm/ })).toBeInTheDocument();
    expect(screen.getAllByText("Dữ liệu minh hoạ — nhóm gán ngẫu nhiên").length).toBeGreaterThan(0);
  });

  it("lọc theo một nhóm gửi tên nhóm trong thân yêu cầu rồi hiện kết quả của nhóm đó", async () => {
    navigation.search = `batch=${ID}`;
    routes();
    storeFile();
    render(<DashboardWorkspace />);
    await userEvent.click(await screen.findByRole("button", { name: /Tất cả 5 nhóm/ }));
    await userEvent.click(await screen.findByRole("menuitemcheckbox", { name: /Nhóm A/ }));
    await waitFor(() =>
      expect(callsTo(`POST /api/batches/${ID}/insights`).map((call) => call.body)).toContainEqual({
        groups: ["Nhóm A"],
      }),
    );
    expect(await screen.findByText(`${batchInsightsFiltered.kpis.n_rows}`, { selector: "dd" })).toBeInTheDocument();
  });

  it("batch không còn trên máy chủ: báo phiên đã kết thúc và cho xem dữ liệu demo", async () => {
    navigation.search = "batch=het-han";
    routes({
      "POST /api/batches/het-han/insights": () => json({ detail: { code: "batch_not_found", message: "x" } }, 404),
    });
    render(<DashboardWorkspace />);
    expect(await screen.findByText("Phiên làm việc đã kết thúc.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Xem dữ liệu demo/ }));
    expect(navigation.replace).toHaveBeenCalledWith("/dashboard/", { scroll: false });
    expect(screen.queryByRole("heading", { name: "Cảnh báo so với mức nền" })).not.toBeInTheDocument();
  });

  it("phân tích chưa xong: báo chờ ở trang Phân tích hàng loạt", async () => {
    navigation.search = `batch=${ID}`;
    routes({
      [`POST /api/batches/${ID}/insights`]: () => json({ detail: { code: "not_done", message: "x" } }, 409),
    });
    storeFile();
    render(<DashboardWorkspace />);
    expect(await screen.findByText("Phân tích chưa xong.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Phân tích hàng loạt" })).toBeInTheDocument();
  });

  it("thiếu số liệu mức nền thì báo bằng tiếng Việt, không để trống trang", async () => {
    routes({
      "POST /api/batches/demo/insights": () => json({ detail: { code: "insights_unavailable", message: "x" } }, 503),
    });
    render(<DashboardWorkspace />);
    expect(await screen.findByText(/thiếu số liệu mức nền/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Thử lại" })).toBeInTheDocument();
  });

  it("chuyển nguồn sang file vừa tải bằng điều hướng trong ứng dụng", async () => {
    routes();
    storeFile();
    render(<DashboardWorkspace />);
    await screen.findByRole("heading", { name: "Cảnh báo so với mức nền" });
    await userEvent.click(screen.getByRole("button", { name: /File vừa tải/ }));
    expect(navigation.replace).toHaveBeenCalledWith(`/dashboard/?batch=${ID}`, { scroll: false });
  });

  it("chưa có file và không có demo thì nói rõ cần làm gì", async () => {
    routes({}, { demo_available: false });
    render(<DashboardWorkspace />);
    expect(await screen.findByText("Chưa có dữ liệu để xem.")).toBeInTheDocument();
    expect(screen.getByText("Dữ liệu demo chưa sẵn")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Phân tích một file/ })).toBeInTheDocument();
    expect(callsTo("POST /api/batches/demo/insights")).toHaveLength(0);
  });

  it("không có demo nhưng có file trong tab thì xem file đó", async () => {
    routes({}, { demo_available: false });
    storeFile();
    render(<DashboardWorkspace />);
    expect(await screen.findByRole("heading", { name: "Cảnh báo so với mức nền" })).toBeInTheDocument();
    expect(callsTo(`POST /api/batches/${ID}/insights`)).toHaveLength(1);
    expect(screen.getByRole("button", { name: /Dữ liệu demo/ })).toBeDisabled();
  });
});
