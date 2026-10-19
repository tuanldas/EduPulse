import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { Callout } from "@/components/shared/callout";
import { ConfidenceBar } from "@/components/shared/confidence-bar";
import { FailureNotice } from "@/components/shared/failure-notice";
import { IllustrativeBadge } from "@/components/shared/illustrative-badge";
import { LabelBadge } from "@/components/shared/label-badge";
import { PiiText, countPiiTags } from "@/components/shared/pii-chip";
import { ProbBars } from "@/components/shared/prob-bars";
import { SegmentedControl } from "@/components/shared/segmented-control";
import { XaiSentence } from "@/components/shared/xai-sentence";
import { failureFromResponse, networkFailure } from "@/lib/api/errors";
import type { HeadExplanation } from "@/lib/api/types";

const HEAD: HeadExplanation = {
  label: "negative",
  probability: 0.93,
  tokens: ["Thầy", "hay", "vào lớp", "trễ", "và", "nghỉ", "không báo", "."].map((text) => ({
    text,
    delta: 0,
    p_without: 0.5,
  })),
  support: [6, 3, 5],
  oppose: [1],
};

describe("XaiSentence", () => {
  it("tô từ ủng hộ theo hạng và từ phản đối, kèm chữ ẩn cho trình đọc màn hình", () => {
    const { container } = render(<XaiSentence head={HEAD} />);
    const marks = container.querySelectorAll("mark");
    expect(marks).toHaveLength(4);
    // Thứ tự trong câu: hay (phản đối), trễ (hạng 2), nghỉ (hạng 3), không báo (hạng 1).
    expect(marks[0]).toHaveTextContent("hay");
    expect(marks[0]).toHaveClass("bg-xai-opp", "italic");
    expect(marks[0]).toHaveTextContent("phản đối");
    expect(marks[1]).toHaveTextContent("trễ");
    expect(marks[1]).toHaveClass("bg-xai-2");
    expect(marks[2]).toHaveTextContent("nghỉ");
    expect(marks[2]).toHaveClass("bg-xai-1", "font-semibold");
    expect(marks[2]).toHaveTextContent("ủng hộ, hạng 3");
    expect(marks[3]).toHaveClass("bg-xai-3");
  });

  it("ghép câu đúng khoảng trắng, dấu chấm dính vào từ cuối", () => {
    const { container } = render(<XaiSentence head={HEAD} />);
    expect(container.textContent?.replace(/\s*\(.*?\)/g, "").replace(/[123−]/g, "")).toBe(
      "Thầy hay vào lớp trễ và nghỉ không báo.",
    );
  });

  it("câu không có từ nào được tô thì không có thẻ mark", () => {
    const { container } = render(<XaiSentence head={{ ...HEAD, support: [], oppose: [] }} />);
    expect(container.querySelectorAll("mark")).toHaveLength(0);
  });
});

describe("thông tin cá nhân đã ẩn", () => {
  it("bọc các thẻ thành chip và giữ nguyên phần còn lại", () => {
    const { container } = render(
      <p>
        <PiiText text="Cô [TÊN] dạy rất nhiệt tình, em xin số [SĐT] để hỏi bài." />
      </p>,
    );
    expect(container.textContent).toBe("Cô [TÊN] dạy rất nhiệt tình, em xin số [SĐT] để hỏi bài.");
    const chips = container.querySelectorAll("span.font-mono");
    expect([...chips].map((chip) => chip.textContent)).toEqual(["[TÊN]", "[SĐT]"]);
  });

  it("đếm số lần xuất hiện của từng thẻ theo thứ tự cố định", () => {
    expect(countPiiTags("Em [SĐT] và [TÊN], thêm [TÊN] nữa.")).toEqual([
      { tag: "[TÊN]", count: 2 },
      { tag: "[SĐT]", count: 1 },
    ]);
    expect(countPiiTags("Không có gì cả")).toEqual([]);
  });
});

describe("thông báo lỗi", () => {
  it("lỗi mạng có nút Thử lại gọi lại hàm được truyền vào", async () => {
    const retry = vi.fn();
    render(<FailureNotice failure={networkFailure()} onRetry={retry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Không kết nối được máy chủ");
    await userEvent.click(screen.getByRole("button", { name: /Thử lại/ }));
    expect(retry).toHaveBeenCalledTimes(1);
  });

  it("lỗi do dữ liệu nhập thì không gợi ý thử lại", () => {
    render(
      <FailureNotice
        failure={failureFromResponse(422, { detail: { code: "text_empty" } })}
        onRetry={() => {}}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Hãy nhập một câu góp ý.");
    expect(screen.queryByRole("button")).toBeNull();
  });
});

describe("SegmentedControl", () => {
  const options = [
    { value: "sentiment", label: "Cảm xúc" },
    { value: "topic", label: "Chủ đề" },
  ] as const;

  it("chọn mục khác thì báo giá trị mới", async () => {
    const onChange = vi.fn();
    render(<SegmentedControl value="sentiment" onValueChange={onChange} options={options} ariaLabel="Tác vụ" />);
    await userEvent.click(screen.getByRole("button", { name: "Chủ đề" }));
    expect(onChange).toHaveBeenCalledWith("topic");
  });

  it("bấm lại mục đang chọn không làm giá trị rỗng", async () => {
    const onChange = vi.fn();
    render(<SegmentedControl value="sentiment" onValueChange={onChange} options={options} ariaLabel="Tác vụ" />);
    await userEvent.click(screen.getByRole("button", { name: "Cảm xúc" }));
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("thành phần hiển thị nhãn", () => {
  it("LabelBadge ghi tên Việt và độ tin cậy", () => {
    render(<LabelBadge task="sentiment" label="negative" confidence={0.9312} />);
    expect(screen.getByText("Tiêu cực · 93%")).toBeInTheDocument();
  });

  it("ProbBars xếp lớp xác suất cao nhất lên đầu và in đậm", () => {
    render(<ProbBars task="topic" probs={{ lecturer: 0.58, facility: 0.31, training_program: 0.07, others: 0.04 }} />);
    const items = screen.getAllByRole("listitem");
    expect(items.map((item) => item.textContent)).toEqual([
      "Giảng viên58%",
      "Cơ sở vật chất31%",
      "Chương trình7%",
      "Khác4%",
    ]);
  });

  it("ConfidenceBar nói rõ khi dưới ngưỡng cờ", () => {
    const { rerender } = render(<ConfidenceBar value={0.58} flagged />);
    expect(screen.getByText("dưới ngưỡng cờ")).toBeInTheDocument();
    rerender(<ConfidenceBar value={0.93} />);
    expect(screen.getByText(/đạt ngưỡng/)).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "93");
  });

  it("Callout đặt vai trò theo ngữ cảnh", () => {
    render(
      <Callout tone="warn" title="Câu có vẻ được gõ không dấu." role="status">
        Hãy gõ lại có dấu nếu được.
      </Callout>,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Câu có vẻ được gõ không dấu. Hãy gõ lại có dấu nếu được.");
  });

  it("IllustrativeBadge co theo khung và xuống dòng được (không bị cắt hay làm trang cuộn ngang)", () => {
    render(<IllustrativeBadge />);
    const badge = screen.getByText("Dữ liệu minh hoạ — nhóm gán ngẫu nhiên");
    expect(badge).toHaveClass("max-w-full", "shrink", "whitespace-normal");
    expect(badge).not.toHaveClass("whitespace-nowrap");
    expect(badge).not.toHaveClass("shrink-0");
  });
});
