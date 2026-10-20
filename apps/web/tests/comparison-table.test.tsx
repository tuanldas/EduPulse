import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ComparisonTable } from "@/components/model/comparison-table";
import type { ModelCard } from "@/lib/api/types";
import { formatDecimal, formatScore } from "@/lib/format";

import { fixture } from "./helpers/api-stub";

const card = fixture<ModelCard>("model-card.json");

type Architecture = ModelCard["architectures"][number];

// Tên và số liệu theo đúng dạng bundle phát hành ghi vào model card.
const stl: Architecture = {
  name: "Đơn nhiệm (STL), sơ bộ: validation, 1 seed",
  seeds: [42],
  sentiment_macro_f1: { mean: 0.8701, std: 0, n: 1 },
  topic_macro_f1: { mean: 0.8154, std: 0, n: 1 },
};
const mtl: Architecture = {
  name: "Đa nhiệm (MTL)",
  seeds: [1, 2, 3],
  sentiment_macro_f1: { mean: 0.85, std: 0.012, n: 3 },
  topic_macro_f1: { mean: 0.82, std: 0.008, n: 3 },
};

const literal = (text: string) => new RegExp(text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));

function rowOf(name: RegExp): HTMLElement {
  const table = screen.getByRole("table", { name: "So sánh với mô hình cơ sở" });
  return within(table).getByRole("row", { name });
}

describe("bảng so sánh với mô hình cơ sở", () => {
  it("mỗi mô hình cơ sở một dòng; dòng đang dùng được đánh dấu và in đậm", () => {
    const baselines = card.baselines.map((baseline, index) => ({ ...baseline, deployed: index === 0 }));
    render(<ComparisonTable baselines={baselines} architectures={[]} />);

    const table = screen.getByRole("table", { name: "So sánh với mô hình cơ sở" });
    expect(within(table).getAllByRole("row")).toHaveLength(baselines.length + 1);
    const deployed = rowOf(literal(baselines[0].name));
    expect(within(deployed).getByText("đang dùng")).toBeInTheDocument();
    expect(deployed).toHaveAttribute("data-highlight", "true");
    expect(within(deployed).getByText(formatScore(baselines[0].sentiment_accuracy))).toHaveClass("font-semibold");
    if (baselines[0].sentiment_ece != null) {
      expect(within(deployed).getByText(formatDecimal(baselines[0].sentiment_ece, 3))).toBeInTheDocument();
    }
  });

  it("mô hình nhiều seed chỉ có Macro-F1 dạng trung bình ± độ lệch chuẩn, ô còn lại ghi dấu gạch", () => {
    render(<ComparisonTable baselines={[]} architectures={[mtl]} />);
    const row = rowOf(/Đa nhiệm \(MTL\)/);
    expect(within(row).getByText(`${formatScore(mtl.sentiment_macro_f1.mean)} ± ${formatScore(mtl.sentiment_macro_f1.std)}`)).toBeInTheDocument();
    expect(within(row).getByText(`${formatScore(mtl.topic_macro_f1.mean)} ± ${formatScore(mtl.topic_macro_f1.std)}`)).toBeInTheDocument();
    expect(within(row).getAllByText("—")).toHaveLength(3);
  });

  it("hàng chỉ có 1 seed ghi một giá trị, không kèm ± 0,0", () => {
    render(<ComparisonTable baselines={[]} architectures={[stl]} />);
    const row = rowOf(/Đơn nhiệm \(STL\)/);
    expect(within(row).getByText(formatScore(stl.sentiment_macro_f1.mean))).toBeInTheDocument();
    expect(within(row).getByText(formatScore(stl.topic_macro_f1.mean))).toBeInTheDocument();
    expect(row).not.toHaveTextContent("±");
  });

  it("ghi số seed sau tên khi tên chưa nói, không lặp khi tên đã có", () => {
    render(<ComparisonTable baselines={[]} architectures={[stl, mtl]} />);
    expect(rowOf(/Đa nhiệm \(MTL\)/)).toHaveTextContent("Đa nhiệm (MTL) (3 seed)");
    // Tên của bản phát hành đã ghi "1 seed" nên không thêm "(1 seed)" nữa.
    const stlRow = rowOf(/Đơn nhiệm \(STL\)/);
    expect(stlRow).toHaveTextContent("Đơn nhiệm (STL), sơ bộ: validation, 1 seed");
    expect(stlRow).not.toHaveTextContent("1 seed (1 seed)");
    expect(within(stlRow).queryByText("(1 seed)")).toBeNull();
  });

  it("cột tên xuống dòng được để tên dài không đẩy các cột số ra ngoài khung bảng", () => {
    render(<ComparisonTable baselines={[]} architectures={[stl]} />);
    const nameCell = within(rowOf(/Đơn nhiệm \(STL\)/)).getAllByRole("cell")[0];
    expect(nameCell).not.toHaveClass("whitespace-nowrap");
  });

  it("không dựng bảng khi không có dòng nào", () => {
    const { container } = render(<ComparisonTable baselines={[]} architectures={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});
