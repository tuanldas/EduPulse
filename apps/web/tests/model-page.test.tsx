import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { defaultBudgetRow } from "@/components/model/derive";
import { ModelPage } from "@/components/model/model-page";
import type { AppConfig, ModelCard } from "@/lib/api/types";
import { resetAppConfig } from "@/lib/app-config";
import { formatDecimal, formatNumber, formatPercent, formatScore } from "@/lib/format";
import { labelMeta } from "@/lib/labels";
import { regroupReliability } from "@/lib/reliability";

import { fixture, json, stubApi } from "./helpers/api-stub";

// `model-card.json` và `config.json` do `pnpm record:fixtures` ghi từ API thật. Mọi con số mong đợi
// được tính từ chính fixture (không ghi cứng) nên ghi lại fixture bằng bundle khác vẫn đúng.
const card = fixture<ModelCard>("model-card.json");
const config = fixture<AppConfig>("config.json");

const parseVi = (text: string | null) => Number((text ?? "").replace(",", "."));

function serve(cardResponse: () => Response | Promise<Response> = () => json(card)) {
  stubApi({
    "GET /api/model-card": cardResponse,
    "GET /api/config": () => json(config),
  });
}

/** Ô chỉ số có nhãn `label` (MetricCard dùng `dl`). Hộp "Đọc các chỉ số thế nào?" cũng có `dt` cùng tên, nằm sau lưới chỉ số. */
function metric(label: string): HTMLElement {
  const [term] = screen.getAllByText(label, { selector: "dt" });
  return term.closest("dl") as HTMLElement;
}

beforeEach(() => {
  resetAppConfig();
});

describe("trang Mô hình & đạo đức", () => {
  it("hiện bốn chỉ số của cảm xúc từ model card, kèm nhãn tập đánh giá theo bản", async () => {
    serve();
    render(<ModelPage />);
    const splitName = card.metrics.split === "test" ? "test" : "validation";
    await screen.findByRole("heading", { name: `Hiệu năng trên tập ${splitName}` });

    const sentiment = card.metrics.sentiment;
    expect(within(metric("Accuracy")).getByText(formatPercent(sentiment.accuracy, 1))).toBeInTheDocument();
    expect(within(metric("Macro-F1")).getByText(formatScore(sentiment.macro_f1))).toBeInTheDocument();
    expect(within(metric("ECE (sau hiệu chỉnh)")).getByText(formatDecimal(sentiment.ece, 3))).toBeInTheDocument();

    const budget = defaultBudgetRow(card);
    expect(budget).toBeDefined();
    const caught = metric(`Lỗi được bắt ở ngân sách duyệt ${formatNumber(budget!.budget)}%`);
    expect(within(caught).getByText(formatPercent(budget!.efficiency.caught_sentiment_errors))).toBeInTheDocument();

    // Bản smoke có thông báo riêng; bản release (số trên test) thì không.
    if (card.stage === "smoke") {
      expect(screen.getByText(/Bản smoke · số trên tập validation/)).toBeInTheDocument();
    } else {
      expect(screen.queryByText(/Bản smoke/)).toBeNull();
    }
  });

  it("chuyển sang Chủ đề thì đổi chỉ số, bảng từng lớp và ma trận nhầm lẫn", async () => {
    serve();
    render(<ModelPage />);
    const user = userEvent.setup();
    await screen.findAllByText("Accuracy", { selector: "dt" });
    await user.click(screen.getByRole("button", { name: "Chủ đề" }));

    const topic = card.metrics.topic;
    expect(within(metric("Accuracy")).getByText(formatPercent(topic.accuracy, 1))).toBeInTheDocument();
    const perClass = screen.getByRole("table", { name: /Chỉ số từng lớp/ });
    for (const item of topic.per_class) {
      expect(within(perClass).getByText(labelMeta("topic", item.label).vi)).toBeInTheDocument();
    }
    expect(within(perClass).getAllByRole("row")).toHaveLength(topic.per_class.length + 1);
    const matrix = screen.getByRole("table", { name: /Ma trận nhầm lẫn/ });
    expect(within(matrix).getAllByRole("row")).toHaveLength(topic.per_class.length + 1);
  });

  it("ma trận nhầm lẫn của cảm xúc: mỗi hàng cộng lại khoảng 100%", async () => {
    serve();
    render(<ModelPage />);
    const matrix = await screen.findByRole("table", { name: /Ma trận nhầm lẫn/ });
    const rows = within(matrix).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(card.metrics.sentiment.per_class.length);
    for (const row of rows) {
      const cells = within(row).getAllByRole("cell");
      const total = cells.reduce((sum, cell) => sum + parseVi(cell.textContent), 0);
      expect(total).toBeGreaterThan(99.5);
      expect(total).toBeLessThan(100.5);
    }
  });

  it("model card hiện mục đích, giới hạn và ghi chú triển khai từ bundle", async () => {
    serve();
    render(<ModelPage />);
    await screen.findByRole("heading", { name: "Model card" });
    await screen.findByText(card.intended_use);
    expect(screen.getByText(card.out_of_scope)).toBeInTheDocument();
    for (const limitation of card.limitations) {
      expect(screen.getByText(limitation)).toBeInTheDocument();
    }
    expect(screen.getByText(card.notes.x86)).toBeInTheDocument();
    expect(screen.getByText(`${card.model_version} · ${card.bundle_version}`)).toBeInTheDocument();
  });

  it("biểu đồ độ tin cậy có bảng số tương đương, mở bằng 'Xem bảng số'", async () => {
    serve();
    render(<ModelPage />);
    const user = userEvent.setup();
    await screen.findByRole("img", { name: /Biểu đồ độ tin cậy/ });
    await user.click(screen.getByText("Xem bảng số"));
    const table = await screen.findByRole("table", { name: /Độ tin cậy báo so với tỉ lệ đúng thực tế/ });
    const populated = regroupReliability(card.metrics.sentiment.reliability).filter((bin) => bin.n > 0);
    expect(within(table).getAllByRole("row")).toHaveLength(populated.length + 1);
  });

  it("mục giấy phép: nút mã nguồn dùng source_url của /api/config", async () => {
    serve();
    render(<ModelPage />);
    const link = await screen.findByRole("link", { name: /Mã nguồn/ });
    expect(link).toHaveAttribute("href", config.source_url);
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
  });

  it("chưa cấu hình source_url thì không có nút mã nguồn và nói rõ", async () => {
    stubApi({
      "GET /api/model-card": () => json(card),
      "GET /api/config": () => json({ ...config, source_url: "" }),
    });
    render(<ModelPage />);
    await screen.findByText(/Liên kết mã nguồn chưa được cấu hình/);
    expect(screen.queryByRole("link", { name: /Mã nguồn trên/ })).toBeNull();
  });

  it("sao chép BibTeX vào clipboard", async () => {
    serve();
    render(<ModelPage />);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Sao chép trích dẫn BibTeX" }));
    const copied = await navigator.clipboard.readText();
    expect(copied).toContain("@inproceedings{nguyen2018uitvsfc");
    expect(copied).toContain("10.1109/KSE.2018.8573337");
    expect(copied).toContain("10.18653/v1/2020.findings-emnlp.92");
  });

  it("chưa có model card cho bản đang chạy: báo rõ, các mục còn lại vẫn hiện", async () => {
    serve(() => json({ detail: { code: "model_card_unavailable", message: "x" } }, 404));
    render(<ModelPage />);
    await screen.findByText(/Chưa có model card cho bản mô hình đang chạy/);
    expect(screen.getByRole("heading", { name: "Quyền riêng tư & vai trò của con người" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Giấy phép, mã nguồn/ })).toBeInTheDocument();
  });

  it("lỗi máy chủ: báo bằng tiếng Việt và thử lại được", async () => {
    let attempt = 0;
    serve(() => {
      attempt += 1;
      return attempt === 1 ? json({ detail: { code: "internal_error", message: "x" } }, 500) : json(card);
    });
    render(<ModelPage />);
    const user = userEvent.setup();
    expect(await screen.findByRole("alert")).toHaveTextContent(/Máy chủ gặp lỗi/);
    await user.click(screen.getByRole("button", { name: /Thử lại/ }));
    await waitFor(() => expect(screen.getAllByText("Accuracy", { selector: "dt" }).length).toBeGreaterThan(0));
    expect(attempt).toBe(2);
  });
});
