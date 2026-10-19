import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ReviewTable, type ReviewTableProps } from "@/components/batch/review-table";
import { NO_OVERRIDES, applyChange } from "@/lib/batch/overrides";
import type { ResultRow } from "@/lib/batch/types";

import { lowConfidence, makeRow } from "./helpers/batch-rows";

// Dựng cả bảng/trang trong jsdom: khi máy bận chạy nhiều tệp song song, 5 giây mặc định không đủ.
vi.setConfig({ testTimeout: 30_000 });

/** `count` dòng có chỉ số KHÔNG bắt đầu từ 0 và không trùng số dòng trong file, như API thật sau khi bỏ dòng trống. */
function rows(count: number): ResultRow[] {
  return Array.from({ length: count }, (_, position) =>
    makeRow({
      index: 100 + position * 2,
      source_row: position + 1,
      text: `Câu số ${position + 1}`,
      source: "flag",
      needs_review: true,
      sentiment: "neutral",
      topic: "lecturer",
      reasons: [lowConfidence("sentiment", "Cảm xúc không chắc — Trung tính 40% · Tích cực 38%")],
    }),
  );
}

function setup(overrides: Partial<ReviewTableProps> = {}) {
  const props: ReviewTableProps = {
    rows: rows(3),
    overrides: NO_OVERRIDES,
    editable: true,
    groupLabel: null,
    xaiMaxWords: 25,
    page: 1,
    onPageChange: vi.fn(),
    onChange: vi.fn(),
    onExplain: vi.fn(),
    ...overrides,
  };
  const view = render(<ReviewTable {...props} />);
  return { ...view, props };
}

/** Cột "#" (số dòng trong file) theo thứ tự đang hiển thị. Duyệt DOM trực tiếp: truy vấn theo role trên bảng 50 dòng rất chậm. */
function shownRowNumbers(): string[] {
  return [...document.querySelectorAll("tbody tr")].map((row) => row.querySelector("td")?.textContent ?? "");
}

describe("ReviewTable", () => {
  it("hiện đúng 50 dòng mỗi trang và nói rõ đang xem dòng nào", () => {
    setup({ rows: rows(120) });
    expect(shownRowNumbers()).toHaveLength(50);
    expect(shownRowNumbers()[0]).toBe("1");
    expect(screen.getByText("Hiển thị 1–50 / 120 dòng")).toBeInTheDocument();
    expect(screen.getByText("Trang 1/3")).toBeInTheDocument();
  });

  it("trang cuối chỉ có phần còn lại và nút Trang sau bị khoá", () => {
    setup({ rows: rows(120), page: 3 });
    expect(shownRowNumbers()).toHaveLength(20);
    expect(shownRowNumbers()[0]).toBe("101");
    expect(screen.getByRole("button", { name: "Trang sau" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Trang trước" })).toBeEnabled();
  });

  it("chuyển trang qua callback, không tự đổi trang", async () => {
    const { props } = setup({ rows: rows(120) });
    await userEvent.click(screen.getByRole("button", { name: "Trang sau" }));
    expect(props.onPageChange).toHaveBeenCalledWith(2);
    expect(shownRowNumbers()[0]).toBe("1");
  });

  it("không có nút phân trang khi chỉ có một trang", () => {
    setup({ rows: rows(50) });
    expect(screen.queryByRole("navigation", { name: "Phân trang" })).not.toBeInTheDocument();
  });

  it("đổi nhãn gửi đúng chỉ số dòng của batch, không phải số dòng trong file hay vị trí", async () => {
    const { props } = setup({ rows: rows(3) });
    await userEvent.selectOptions(screen.getByLabelText("Sửa cảm xúc câu 2"), "negative");
    expect(props.onChange).toHaveBeenCalledWith(102, { sentiment: "negative" });
    await userEvent.selectOptions(screen.getByLabelText("Sửa chủ đề câu 3"), "facility");
    expect(props.onChange).toHaveBeenLastCalledWith(104, { topic: "facility" });
  });

  it("chọn lại dấu gạch thì xoá nhãn đã sửa", async () => {
    const overrides = applyChange(NO_OVERRIDES, 100, { sentiment: "negative" });
    const { props } = setup({ overrides });
    expect(screen.getByLabelText("Sửa cảm xúc câu 1")).toHaveValue("negative");
    await userEvent.selectOptions(screen.getByLabelText("Sửa cảm xúc câu 1"), "");
    expect(props.onChange).toHaveBeenCalledWith(100, { sentiment: null });
  });

  it("không cho chọn lại đúng nhãn của mô hình làm nhãn sửa", () => {
    setup();
    const options = within(screen.getByLabelText("Sửa cảm xúc câu 1")).getAllByRole("option");
    expect(options.map((option) => option.textContent)).toEqual(["—", "Tiêu cực", "Tích cực"]);
  });

  it("sửa nhãn không làm đổi thứ tự dòng", () => {
    const { rerender, props } = setup({ rows: rows(5) });
    const before = shownRowNumbers();
    const overrides = applyChange(applyChange(NO_OVERRIDES, 104, { sentiment: "positive" }), 100, {
      topic: "others",
    });
    rerender(<ReviewTable {...props} overrides={overrides} />);
    expect(shownRowNumbers()).toEqual(before);
    expect(screen.getByLabelText("Sửa cảm xúc câu 3")).toHaveValue("positive");
    expect(screen.getByLabelText("Sửa chủ đề câu 1")).toHaveValue("others");
    expect(screen.getByLabelText("Sửa cảm xúc câu 2")).toHaveValue("");
  });

  it("mở giải thích theo vị trí của dòng trong cả danh sách, kể cả ở trang sau", async () => {
    const { props } = setup({ rows: rows(120), page: 2 });
    await userEvent.click(screen.getByRole("button", { name: "Giải thích câu 52" }));
    expect(props.onExplain).toHaveBeenCalledWith(51);
  });

  it("tab chỉ xem: không có ô sửa, có cột Đã sửa ghi nhãn đã đổi", () => {
    const overrides = applyChange(NO_OVERRIDES, 102, { sentiment: "negative", topic: "facility" });
    setup({ editable: false, overrides });
    expect(screen.queryByLabelText(/^Sửa cảm xúc/)).not.toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Đã sửa" })).toBeInTheDocument();
    expect(screen.getByText("Cảm xúc → Tiêu cực; Chủ đề → Cơ sở vật chất")).toBeInTheDocument();
  });

  it("ghi lý do ngắn và cho xem lý do đầy đủ khi rê chuột", () => {
    setup({ rows: rows(1) });
    const cell = screen.getByText("Cảm xúc không chắc").closest("td");
    expect(cell).toHaveAttribute("title", expect.stringContaining("Trung tính 40%"));
  });

  it("chỉ có cột nhóm khi đã chọn cột nhóm", () => {
    const withGroup = rows(2).map((row) => ({ ...row, group: "Nhóm A" }));
    const { unmount } = setup({ rows: withGroup, groupLabel: "Nhóm (minh hoạ)" });
    expect(screen.getByRole("columnheader", { name: "Nhóm (minh hoạ)" })).toBeInTheDocument();
    expect(screen.getAllByText("Nhóm A")).toHaveLength(2);
    unmount();
    setup({ rows: withGroup, groupLabel: null });
    expect(screen.queryByRole("columnheader", { name: "Nhóm (minh hoạ)" })).not.toBeInTheDocument();
  });

  it("thẻ thông tin cá nhân đã ẩn hiện thành chip, không phải chữ thô", () => {
    const row = makeRow({ index: 7, source_row: 7, text: "Phòng lab [TÊN] rất nóng", pii_types: ["name"] });
    setup({ rows: [row], editable: false });
    expect(screen.getByText("[TÊN]")).toBeInTheDocument();
    expect(screen.getByText("[TÊN]").className).toContain("font-mono");
  });

  it("danh sách rỗng báo rõ thay vì để bảng trống", () => {
    setup({ rows: [] });
    expect(screen.getByText("Không có câu nào để hiển thị.")).toBeInTheDocument();
    expect(screen.getByText("Hiển thị 0–0 / 0 dòng")).toBeInTheDocument();
  });
});
