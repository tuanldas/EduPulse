import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ExplainSheet, type ExplainSheetProps } from "@/components/batch/explain-sheet";
import { failureFromResponse } from "@/lib/api/errors";
import { rowsForTab } from "@/lib/batch/review-rows";

import { batchExplain, batchResults } from "./helpers/batch-fixtures";

const reviewRows = rowsForTab(batchResults, "review");
const allRows = rowsForTab(batchResults, "all");
const { withHighlights, noHighlights } = batchExplain;

/** Trả đúng giải thích đã ghi theo chỉ số dòng, như API thật. */
const load = () =>
  vi.fn((_batchId: string, index: number) =>
    Promise.resolve(index === withHighlights.index ? withHighlights.response : noHighlights.response),
  );

function setup(overrides: Partial<ExplainSheetProps> = {}) {
  const props: ExplainSheetProps = {
    open: true,
    onOpenChange: vi.fn(),
    batchId: "fixture-batch",
    rows: allRows,
    position: allRows.findIndex((row) => row.index === withHighlights.index),
    onPositionChange: vi.fn(),
    xaiMaxWords: 25,
    load: load(),
    ...overrides,
  };
  const view = render(<ExplainSheet {...props} />);
  return { ...view, props };
}

describe("ExplainSheet (dữ liệu ghi từ API thật)", () => {
  it("hiện câu, lý do gắn cờ bằng lời và câu tô từ sau khi tải xong", async () => {
    const { props } = setup();
    const row = allRows.find((candidate) => candidate.index === withHighlights.index)!;
    expect(await screen.findByRole("heading", { name: `Giải thích câu #${row.source_row}` })).toBeInTheDocument();
    expect(screen.getByText("Cần xem xét")).toBeInTheDocument();
    // Đã tải đúng dòng của batch theo chỉ số (index), không phải số dòng trong file.
    await waitFor(() => expect(props.load).toHaveBeenCalledWith("fixture-batch", row.index, expect.any(AbortSignal)));
    expect(await screen.findByText(/Bỏ “/)).toBeInTheDocument();
    expect(document.querySelectorAll("mark").length).toBeGreaterThan(0);
  });

  it("câu rất chắc chắn không có từ nào để tô thì nói rõ thay vì để trống", async () => {
    const position = allRows.findIndex((row) => row.index === noHighlights.index);
    setup({ position });
    expect(await screen.findByText(/Không có từ nào ảnh hưởng đủ lớn/)).toBeInTheDocument();
    expect(document.querySelectorAll("mark")).toHaveLength(0);
  });

  it("ghi số thứ tự của câu trong danh sách đang xem", async () => {
    setup({ rows: reviewRows, position: 2 });
    expect(await screen.findByText(`Câu 3 / ${reviewRows.length} trong danh sách đang xem.`)).toBeInTheDocument();
  });

  it("Câu trước bị khoá ở đầu danh sách, Câu sau báo vị trí kế tiếp", async () => {
    const { props } = setup({ rows: reviewRows, position: 0 });
    await screen.findByRole("heading", { name: /Giải thích câu/ });
    expect(screen.getByRole("button", { name: /Câu trước/ })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: /Câu sau/ }));
    expect(props.onPositionChange).toHaveBeenCalledWith(1);
  });

  it("Câu sau bị khoá ở cuối danh sách", async () => {
    setup({ rows: reviewRows, position: reviewRows.length - 1 });
    await screen.findByRole("heading", { name: /Giải thích câu/ });
    expect(screen.getByRole("button", { name: /Câu sau/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Câu trước/ })).toBeEnabled();
  });

  it("câu quá dài không gọi API và nói rõ vì sao không tô từ", async () => {
    const position = allRows.findIndex((row) => row.index === batchExplain.tooLongIndex);
    const { props } = setup({ position });
    expect(await screen.findByText(/nên không tô từ để giữ tốc độ/)).toBeInTheDocument();
    expect(screen.getByText(/Câu dài 58 từ \(> 25\)/)).toBeInTheDocument();
    expect(props.load).not.toHaveBeenCalled();
  });

  it("đang tải thì có khung chờ cho trình đọc màn hình", async () => {
    setup({ load: vi.fn().mockReturnValue(new Promise(() => {})) });
    await screen.findByRole("heading", { name: /Giải thích câu/ });
    expect(screen.getByLabelText("Đang tính giải thích")).toBeInTheDocument();
  });

  it("lỗi mạng hiện thông báo tiếng Việt và cho thử lại", async () => {
    const flaky = vi
      .fn()
      .mockRejectedValueOnce(failureFromResponse(503, { detail: { code: "busy" } }))
      .mockResolvedValue(withHighlights.response);
    setup({ load: flaky });
    expect(await screen.findByText("Máy chủ đang bận, thử lại sau ít phút.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Thử lại" }));
    expect(await screen.findByText(/Bỏ “/)).toBeInTheDocument();
    expect(flaky).toHaveBeenCalledTimes(2);
  });

  it("mã không có nội dung hiện như thông tin, không phải lỗi đỏ", async () => {
    setup({ load: vi.fn().mockRejectedValue(failureFromResponse(422, { detail: { code: "no_content" } })) });
    expect(await screen.findByText(/không có nội dung để phân tích/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Thử lại" })).not.toBeInTheDocument();
  });

  it("đóng ngăn thì không tải gì", () => {
    const { props } = setup({ open: false });
    expect(props.load).not.toHaveBeenCalled();
    expect(screen.queryByRole("heading", { name: /Giải thích câu/ })).not.toBeInTheDocument();
  });
});
