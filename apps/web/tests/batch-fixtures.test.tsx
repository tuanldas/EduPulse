import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ResultsSection } from "@/components/batch/results-section";
import { InsightsView } from "@/components/dashboard/insights-view";
import { NO_OVERRIDES, applyChange } from "@/lib/batch/overrides";
import { rowsForTab } from "@/lib/batch/review-rows";
import { parseStoredBatch, storedBatchFromUpload } from "@/lib/batch/stored-batch";

import {
  batchInsights,
  batchInsightsFiltered,
  batchResults,
  batchStatusDone,
  batchUpload,
} from "./helpers/batch-fixtures";

// Test dựa trên fixture ghi từ API thật (scripts/record-batch-fixtures.mjs): 20 câu tự viết, nhóm minh hoạ A–E.

describe("hợp đồng của dữ liệu thật", () => {
  it("danh sách duyệt gồm câu bị cờ và câu kiểm tra, chỉ trỏ tới dòng có thật", () => {
    const { summary, rows, review_order } = batchResults;
    expect(review_order).toHaveLength(summary.n_review);
    expect(summary.n_flagged + summary.n_audit).toBe(summary.n_review);
    const indexes = new Set(rows.map((row) => row.index));
    expect(review_order.every((index) => indexes.has(index))).toBe(true);
    expect(rowsForTab(batchResults, "review")).toHaveLength(summary.n_review);
    expect(rowsForTab(batchResults, "all")).toHaveLength(summary.n_analyzed);
  });

  it("câu có thông tin cá nhân đã được ẩn thành thẻ trước khi tới giao diện", () => {
    const masked = batchResults.rows.filter((row) => row.pii_types.length > 0);
    expect(masked.length).toBe(batchResults.summary.pii_rows);
    for (const row of masked) expect(row.text).toMatch(/\[(TÊN|SĐT|EMAIL|MSSV|CCCD|URL)\]/);
  });

  it("batch vừa tải lên khôi phục được từ bản lưu trong tab, không kèm phần xem trước", () => {
    const stored = storedBatchFromUpload(batchUpload, 4096);
    expect(parseStoredBatch(JSON.parse(JSON.stringify(stored)))).toEqual(stored);
    expect(JSON.stringify(stored)).not.toContain(batchUpload.preview[0][0]);
    expect(stored.illustrativeGroups).toBe(true);
    expect(batchStatusDone.state).toBe("done");
  });
});

describe("ResultsSection (dữ liệu thật)", () => {
  function setup(overrides = NO_OVERRIDES) {
    const onChange = vi.fn();
    const view = render(
      <ResultsSection
        batchId="fixture-batch"
        results={batchResults}
        groupLabel="Nhóm (minh hoạ)"
        xaiMaxWords={25}
        overrides={overrides}
        correctedCount={Object.keys(overrides).length}
        onChange={onChange}
      />,
    );
    return { ...view, onChange };
  }

  function shownRowNumbers(): string[] {
    const body = screen.getAllByRole("rowgroup")[1];
    return within(body)
      .getAllByRole("row")
      .map((row) => within(row).getAllByRole("cell")[0].textContent ?? "");
  }

  it("bốn số liệu lấy từ summary của máy chủ", () => {
    setup();
    const { summary } = batchResults;
    expect(screen.getByText("Số câu").closest("dl")).toHaveTextContent(`${summary.n_analyzed}`);
    expect(screen.getByText("Số câu").closest("dl")).toHaveTextContent(`${summary.n_groups} nhóm`);
    expect(screen.getByText("Cần duyệt").closest("dl")).toHaveTextContent("20,0% số câu (ngân sách 20%)");
    expect(screen.getByText("Đã ẩn thông tin cá nhân").closest("dl")).toHaveTextContent("[TÊN] 1 · [SĐT] 1");
    expect(screen.getByText("Đã sửa nhãn").closest("dl")).toHaveTextContent(`0 / ${summary.n_review}`);
  });

  it("tab Cần duyệt theo đúng thứ tự cố định của máy chủ, dùng số dòng trong file", () => {
    setup();
    const bySourceRow = new Map(batchResults.rows.map((row) => [row.index, row.source_row]));
    expect(shownRowNumbers()).toEqual(batchResults.review_order.map((index) => `${bySourceRow.get(index)}`));
    expect(screen.getByRole("tab", { name: /Cần duyệt \(5\)/ })).toBeInTheDocument();
  });

  it("lý do ngắn gọn theo từng loại, kể cả câu không dấu và mẫu kiểm tra", () => {
    setup();
    const table = screen.getByRole("table");
    expect(within(table).getByText("Câu không dấu")).toBeInTheDocument();
    expect(within(table).getByText("Ngẫu nhiên (kiểm tra)")).toBeInTheDocument();
    expect(within(table).getAllByText(/không chắc/).length).toBeGreaterThan(0);
  });

  it("sửa nhãn gửi chỉ số dòng của batch", async () => {
    const { onChange } = setup();
    const first = batchResults.rows.find((row) => row.index === batchResults.review_order[0])!;
    await userEvent.selectOptions(screen.getByLabelText(`Sửa chủ đề câu ${first.source_row}`), "others");
    expect(onChange).toHaveBeenCalledWith(first.index, { topic: "others" });
  });

  it("nhãn đã sửa được đếm và hiện ở cột Đã sửa của tab Tất cả", async () => {
    const overrides = applyChange(NO_OVERRIDES, batchResults.review_order[0], { sentiment: "negative" });
    setup(overrides);
    expect(screen.getByText("Đã sửa nhãn").closest("dl")).toHaveTextContent(`1 / ${batchResults.summary.n_review}`);
    await userEvent.click(screen.getByRole("tab", { name: /Tất cả kết quả/ }));
    expect(screen.getByText("Cảm xúc → Tiêu cực")).toBeInTheDocument();
    expect(shownRowNumbers()).toHaveLength(batchResults.summary.n_analyzed);
  });

  it("tab Tất cả hiện câu có thông tin cá nhân bằng chip, không lộ thông tin gốc", async () => {
    setup();
    await userEvent.click(screen.getByRole("tab", { name: /Tất cả kết quả/ }));
    expect(screen.getByText("[TÊN]")).toBeInTheDocument();
    expect(screen.getByText("[SĐT]")).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("0912345678");
  });

  it("dòng chỉ còn thông tin cá nhân hoặc ký hiệu hiện riêng, không lẫn vào bảng", async () => {
    const withSkipped = {
      ...batchResults,
      summary: { ...batchResults.summary, n_skipped: 1 },
      skipped: [{ index: 30, source_row: 31, text: "[SĐT]" }],
    };
    render(
      <ResultsSection
        batchId="fixture-batch"
        results={withSkipped}
        groupLabel={null}
        xaiMaxWords={25}
        overrides={NO_OVERRIDES}
        correctedCount={0}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByText("Bỏ qua 1 dòng không có nội dung")).toBeInTheDocument();
    const toggle = screen.getByRole("button", { name: /1 dòng không có nội dung để phân tích/ });
    await userEvent.click(toggle);
    expect(await screen.findByText("31")).toBeInTheDocument();
    expect(screen.getByText(/không có nhãn và không nằm trong danh sách duyệt/)).toBeInTheDocument();
    // Dòng bị bỏ qua không có trong bảng duyệt.
    expect(within(screen.getByRole("table")).queryByText("[SĐT]")).not.toBeInTheDocument();
  });

  it("câu quá dài ghi rõ số từ ở cột Lý do để hiểu vì sao không tô được", async () => {
    setup();
    await userEvent.click(screen.getByRole("tab", { name: /Tất cả kết quả/ }));
    expect(screen.getByText("dài 58 từ")).toBeInTheDocument();
  });
});

describe("InsightsView (dữ liệu thật)", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
  });

  it("file ít câu: không có cảnh báo nào và nói rõ như khung trạng thái mặc định", () => {
    render(<InsightsView data={batchInsights} groupLabel="Nhóm (minh hoạ)" />);
    expect(screen.getByText("Không chủ đề nào vượt mức nền đáng kể.")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Tiêu cực theo Nhóm (minh hoạ) × Chủ đề" })).toBeInTheDocument();
    // Nhóm gán ngẫu nhiên luôn mang nhãn minh hoạ.
    expect(screen.getAllByText("Dữ liệu minh hoạ — nhóm gán ngẫu nhiên").length).toBeGreaterThan(0);
  });

  it("heatmap có đủ các nhóm của file, ô ít câu không tô", () => {
    render(<InsightsView data={batchInsights} groupLabel="Nhóm (minh hoạ)" />);
    for (const group of batchInsights.groups ?? []) {
      expect(screen.getByRole("rowheader", { name: group.name })).toBeInTheDocument();
    }
    const cells = screen.getAllByRole("cell");
    expect(cells.length).toBe((batchInsights.heatmap?.groups.length ?? 0) * 4);
    // Mỗi nhóm chỉ có 4 câu nên mọi ô đều dưới ngưỡng và không có ô nào nêu chênh lệch so với nền.
    expect(cells.every((cell) => cell.getAttribute("style") === null)).toBe(true);
  });

  it("lọc theo một nhóm cho kết quả của riêng nhóm đó", () => {
    render(<InsightsView data={batchInsightsFiltered} groupLabel="Nhóm (minh hoạ)" />);
    expect(screen.getByText("Số câu").closest("dl")).toHaveTextContent(`${batchInsightsFiltered.kpis.n_rows}`);
    expect(batchInsightsFiltered.selected_groups).toEqual(["Nhóm A"]);
  });
});
