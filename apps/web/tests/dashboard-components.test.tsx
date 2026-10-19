import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AlertCards } from "@/components/dashboard/alert-cards";
import { DeltaHeatmap } from "@/components/dashboard/delta-heatmap";
import { InsightsView } from "@/components/dashboard/insights-view";
import { KeywordChips } from "@/components/dashboard/keyword-chips";
import { KpiRow } from "@/components/dashboard/kpi-row";
import { Recommendations } from "@/components/dashboard/recommendations";
import { RepresentativeSentences } from "@/components/dashboard/representative-sentences";
import { SentimentByTopic } from "@/components/dashboard/sentiment-by-topic";
import { NO_QUOTE_FILTER } from "@/lib/dashboard/insight-view";

import { makeInsights, makeKpis, makeQuote, makeRecommendation, makeTopicStat } from "./helpers/insights";

// Dựng cả bảng/trang trong jsdom: khi máy bận chạy nhiều tệp song song, 5 giây mặc định không đủ.
vi.setConfig({ testTimeout: 30_000 });

describe("KpiRow", () => {
  it("file vừa tải: bốn ô Số câu, Tiêu cực, Tích cực, Cần duyệt kèm chênh lệch so với nền", () => {
    render(<KpiRow kpis={makeKpis()} />);
    expect(screen.getByText("1.000")).toBeInTheDocument();
    expect(screen.getByText("6 nhóm")).toBeInTheDocument();
    expect(screen.getByText("54,1%")).toBeInTheDocument();
    expect(screen.getByText("7,5 điểm so với nền")).toBeInTheDocument();
    expect(screen.getByText("40,4%")).toBeInTheDocument();
    expect(screen.getByText("9,0 điểm so với nền")).toBeInTheDocument();
    expect(screen.getByText("Cần duyệt")).toBeInTheDocument();
    expect(screen.getByText("100 câu bị gắn cờ")).toBeInTheDocument();
  });

  it("dữ liệu demo không có bước gắn cờ nên ô thứ tư là Trung tính", () => {
    render(
      <KpiRow
        kpis={makeKpis({
          n_rows: 3166,
          n_groups: null,
          n_flagged: null,
          flagged_share: null,
          shares: { negative: 0.445, neutral: 0.053, positive: 0.502 },
          delta_points: { negative: -2.1, neutral: 1, positive: 1.1 },
        })}
      />,
    );
    expect(screen.queryByText("Cần duyệt")).not.toBeInTheDocument();
    expect(screen.getByText("Trung tính")).toBeInTheDocument();
    expect(screen.getByText("3.166")).toBeInTheDocument();
    expect(screen.getByText("2,1 điểm dưới nền")).toBeInTheDocument();
  });
});

describe("AlertCards", () => {
  const topics = makeInsights().topics;

  it("không có cảnh báo thì nói rõ và liệt kê tiêu cực so với nền từng chủ đề", () => {
    render(<AlertCards alerts={[]} topics={topics} illustrativeGroups={false} onSeeQuotes={vi.fn()} />);
    expect(screen.getByText("Không chủ đề nào vượt mức nền đáng kể.")).toBeInTheDocument();
    expect(screen.getByText(/Giảng viên 38,0% \(nền 35,6\)/)).toBeInTheDocument();
    expect(screen.getByText(/Chương trình 93,3% \(78,0\)/)).toBeInTheDocument();
  });

  it("thẻ cảnh báo viết thành lời, in đậm số chính và nêu căn cứ", () => {
    render(<AlertCards alerts={makeInsights().alerts} topics={topics} illustrativeGroups={false} onSeeQuotes={vi.fn()} />);
    expect(screen.getByRole("heading", { name: "Chương trình: tỉ lệ tiêu cực cao hơn mức nền" })).toBeInTheDocument();
    expect(screen.getByText("93%").tagName).toBe("B");
    expect(screen.getByText("Khoảng tin cậy 95%: ít nhất 89%")).toBeInTheDocument();
    expect(screen.getByText("n = 195")).toBeInTheDocument();
    // Dưới 10% ghi một chữ số thập phân.
    expect(screen.getByText("Khoảng tin cậy 95%: ít nhất 7,4%")).toBeInTheDocument();
    // Tối đa ba từ khoá nổi bật.
    expect(screen.getByText("Nổi bật: máy chiếu, phòng máy, quạt")).toBeInTheDocument();
    expect(screen.getByText("Nhóm: Nhóm A")).toBeInTheDocument();
    expect(screen.getByText("Tổng hợp theo nhóm, không gắn với cá nhân.")).toBeInTheDocument();
  });

  it("cảnh báo theo nhóm gán ngẫu nhiên phải ghi rõ là dữ liệu minh hoạ, cảnh báo cả tập thì không cần", () => {
    const { rerender } = render(
      <AlertCards alerts={makeInsights().alerts} topics={topics} illustrativeGroups onSeeQuotes={vi.fn()} />,
    );
    // Chỉ cảnh báo riêng một nhóm (cảnh báo thứ hai trong dữ liệu mẫu) mang nhãn.
    expect(screen.getAllByText("Dữ liệu minh hoạ — nhóm gán ngẫu nhiên")).toHaveLength(1);
    rerender(<AlertCards alerts={makeInsights().alerts} topics={topics} illustrativeGroups={false} onSeeQuotes={vi.fn()} />);
    expect(screen.queryByText("Dữ liệu minh hoạ — nhóm gán ngẫu nhiên")).not.toBeInTheDocument();
  });

  it("'Xem câu liên quan' báo đúng cảnh báo được chọn", async () => {
    const onSeeQuotes = vi.fn();
    const alerts = makeInsights().alerts;
    render(<AlertCards alerts={alerts} topics={topics} illustrativeGroups={false} onSeeQuotes={onSeeQuotes} />);
    await userEvent.click(screen.getAllByRole("button", { name: /Xem câu liên quan/ })[1]);
    expect(onSeeQuotes).toHaveBeenCalledWith(alerts[1]);
  });

  it("phân biệt chủ đề bình thường với chủ đề chưa đủ câu và ghi rõ cảnh báo chỉ ở một nhóm", () => {
    render(<AlertCards alerts={makeInsights().alerts} topics={topics} illustrativeGroups={false} onSeeQuotes={vi.fn()} />);
    expect(screen.getByText("Giảng viên (toàn tập):")).toBeInTheDocument();
    expect(screen.getByText(/trong mức bình thường \(38%\)/)).toBeInTheDocument();
    expect(screen.getByText(/chưa đủ câu để kết luận \(n = 9\)/)).toBeInTheDocument();
  });
});

describe("SentimentByTopic", () => {
  it("mỗi hàng có mô tả bằng chữ đủ số liệu và mức nền cho người dùng trình đọc màn hình", () => {
    render(<SentimentByTopic topics={makeInsights().topics} />);
    expect(
      screen.getByText("Cơ sở vật chất: 96% tiêu cực, 1% trung tính, 3% tích cực. Mức nền tiêu cực 95,4%. 90 câu."),
    ).toBeInTheDocument();
  });

  it("chủ đề chưa có câu nào không vẽ cột", () => {
    render(
      <SentimentByTopic
        topics={[makeTopicStat({ topic: "facility", n: 0, sentiment_shares: { negative: 0, neutral: 0, positive: 0 } })]}
      />,
    );
    expect(screen.getByText("Cơ sở vật chất: chưa có câu nào.")).toBeInTheDocument();
    expect(screen.getByText("Chưa có câu nào")).toBeInTheDocument();
  });
});

describe("DeltaHeatmap", () => {
  const heatmap = makeInsights().heatmap!;

  it("có tiêu đề cột rút gọn và tiêu đề hàng là tên nhóm", () => {
    render(<DeltaHeatmap heatmap={heatmap} groupLabel="Môn học" illustrative={false} />);
    expect(screen.getByRole("heading", { name: "Tiêu cực theo Môn học × Chủ đề" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "CSVC" })).toBeInTheDocument();
    expect(screen.getByRole("rowheader", { name: "Nhóm B" })).toBeInTheDocument();
  });

  it("ô đủ câu được tô màu và ghi chênh lệch so với nền", () => {
    render(<DeltaHeatmap heatmap={heatmap} groupLabel="Môn học" illustrative={false} />);
    const cell = screen.getByRole("cell", {
      name: "Nhóm A, Giảng viên: 58% tiêu cực, +22 điểm so với mức nền, 120 câu.",
    });
    expect(cell).toHaveTextContent("58%");
    expect(cell).toHaveTextContent("+22");
    // Chênh ≥ 20 điểm là bậc đậm nhất của thang màu.
    expect(cell.getAttribute("style")).toMatch(/181, 64, 31/);
  });

  it("ô ít câu không tô và không nêu chênh lệch, ô không có câu ghi dấu gạch", () => {
    render(<DeltaHeatmap heatmap={heatmap} groupLabel="Môn học" illustrative={false} />);
    const sparse = screen.getByRole("cell", {
      name: "Nhóm A, Cơ sở vật chất: 92% tiêu cực, chỉ 12 câu nên chưa đủ để so với mức nền.",
    });
    expect(sparse).toHaveTextContent("92%");
    expect(sparse).toHaveTextContent("n 12");
    expect(sparse.textContent).not.toContain("+");
    expect(sparse.getAttribute("style")).toBeNull();
    const empty = screen.getByRole("cell", { name: "Nhóm A, Khác: chưa có câu nào." });
    expect(empty).toHaveTextContent("–");
  });

  it("nhóm gán ngẫu nhiên luôn mang nhãn dữ liệu minh hoạ", () => {
    const { rerender } = render(<DeltaHeatmap heatmap={heatmap} groupLabel="Nhóm" illustrative />);
    expect(screen.getByText("Dữ liệu minh hoạ — nhóm gán ngẫu nhiên")).toBeInTheDocument();
    rerender(<DeltaHeatmap heatmap={heatmap} groupLabel="Nhóm" illustrative={false} />);
    expect(screen.queryByText("Dữ liệu minh hoạ — nhóm gán ngẫu nhiên")).not.toBeInTheDocument();
  });

  it("chú giải nêu ngưỡng số câu lấy từ dữ liệu", () => {
    render(<DeltaHeatmap heatmap={{ ...heatmap, min_n: 25 }} groupLabel="Nhóm" illustrative={false} />);
    expect(screen.getByText(/< 25 câu \(không tô\)/)).toBeInTheDocument();
  });
});

describe("KeywordChips", () => {
  const keywords = makeInsights().keywords;

  it("xếp khối CSVC, Chương trình, Giảng viên và mỗi từ kèm số câu", () => {
    render(<KeywordChips keywords={keywords} selected={null} onSelect={vi.fn()} />);
    const groups = screen.getAllByRole("group").map((group) => group.getAttribute("aria-label"));
    expect(groups).toEqual([
      "Từ khoá chủ đề Cơ sở vật chất",
      "Từ khoá chủ đề Chương trình",
      "Từ khoá chủ đề Giảng viên",
    ]);
    expect(screen.getByRole("button", { name: /^máy chiếu\s*19 câu$/ })).toBeInTheDocument();
  });

  it("chọn một từ báo cả chủ đề; chọn lại từ đang chọn thì bỏ chọn", async () => {
    const onSelect = vi.fn();
    const { rerender } = render(<KeywordChips keywords={keywords} selected={null} onSelect={onSelect} />);
    await userEvent.click(screen.getByRole("button", { name: /^thực hành\s*34 câu$/ }));
    expect(onSelect).toHaveBeenCalledWith({ topic: "training_program", term: "thực hành" });

    rerender(<KeywordChips keywords={keywords} selected={{ topic: "training_program", term: "thực hành" }} onSelect={onSelect} />);
    const pressed = screen.getByRole("button", { name: /^thực hành\s*34 câu$/ });
    expect(pressed).toHaveAttribute("aria-pressed", "true");
    await userEvent.click(pressed);
    expect(onSelect).toHaveBeenLastCalledWith(null);
  });
});

describe("RepresentativeSentences", () => {
  function setup(overrides: Partial<Parameters<typeof RepresentativeSentences>[0]> = {}) {
    const props = {
      quotes: makeInsights().quotes,
      tab: "facility" as const,
      onTabChange: vi.fn(),
      filter: NO_QUOTE_FILTER,
      onFilterChange: vi.fn(),
      ...overrides,
    };
    return { ...render(<RepresentativeSentences {...props} />), props };
  }

  it("in đậm từ khoá có trong câu bằng thẻ mark", () => {
    const { container } = setup();
    const marked = [...container.querySelectorAll("mark")].map((mark) => mark.textContent);
    expect(marked).toEqual(["Máy chiếu", "mờ", "máy chiếu"]);
  });

  it("lọc theo từ khoá và cho bỏ lọc bằng chip", async () => {
    const { props } = setup({ filter: { term: "máy chiếu", group: null } });
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    await userEvent.click(screen.getByRole("button", { name: "Bỏ lọc từ khoá máy chiếu" }));
    expect(props.onFilterChange).toHaveBeenCalledWith({ term: null, group: null });
  });

  it("lọc theo nhóm và ghi rõ nhóm đang lọc", () => {
    setup({ filter: { term: null, group: "Nhóm B" } });
    expect(screen.getByText("Nhóm: Nhóm B")).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
  });

  it("báo rõ khi bộ lọc không khớp câu nào", () => {
    setup({ filter: { term: "quạt", group: null } });
    expect(screen.getByText("Không có câu nào về Cơ sở vật chất khớp bộ lọc này.")).toBeInTheDocument();
  });

  it("chỉ hiện 5 câu rồi cho xem thêm", async () => {
    const many = Array.from({ length: 7 }, (_, index) => makeQuote({ text: `Câu ${index + 1} về máy chiếu.`, matched: [] }));
    setup({ quotes: [{ topic: "facility", quotes: many }] });
    expect(screen.getAllByRole("listitem")).toHaveLength(5);
    await userEvent.click(screen.getByRole("button", { name: "Xem thêm (2 câu)" }));
    expect(screen.getAllByRole("listitem")).toHaveLength(7);
    expect(screen.queryByRole("button", { name: /Xem thêm/ })).not.toBeInTheDocument();
  });

  it("đổi tab báo chủ đề được chọn", async () => {
    const { props } = setup();
    await userEvent.click(screen.getByRole("tab", { name: "Chương trình" }));
    expect(props.onTabChange).toHaveBeenCalledWith("training_program");
  });

  it("câu nhóm Khác chưa có thì nói rõ thay vì để trống", async () => {
    setup({ tab: "others" });
    expect(screen.getByText("Chưa có câu tiêu biểu nào về Khác.")).toBeInTheDocument();
  });
});

describe("Recommendations", () => {
  it("hiện hành động, căn cứ và nhãn chủ đề rút gọn; luật lạ vẫn hiện được", () => {
    render(
      <Recommendations
        recommendations={[makeRecommendation(), makeRecommendation({
            id: "luat_moi",
            topic: "lecturer",
            action: "Việc mới",
            evidence: "Căn cứ: 5 câu nhắc “nhanh”.",
            note: "Tổng hợp theo nhóm, không gắn với cá nhân.",
          })]}
      />,
    );
    expect(screen.getByText("Bảo trì thiết bị trình chiếu và âm thanh ở phòng học")).toBeInTheDocument();
    expect(screen.getByText(/Căn cứ: 22 câu tiêu cực/)).toBeInTheDocument();
    expect(screen.getByText("CSVC")).toBeInTheDocument();
    expect(screen.getByText("Việc mới")).toBeInTheDocument();
    expect(screen.getByText("Tổng hợp theo nhóm, không gắn với cá nhân.")).toBeInTheDocument();
  });

  it("luôn nhắc đây chỉ là gợi ý và không dùng để đánh giá cá nhân", () => {
    render(<Recommendations recommendations={[]} />);
    expect(screen.getByText(/Chưa có khuyến nghị nào/)).toBeInTheDocument();
    expect(screen.getByText("chỉ để gợi ý")).toBeInTheDocument();
    expect(screen.getByText(/Không dùng để đánh giá cá nhân giảng viên/)).toBeInTheDocument();
  });
});

describe("InsightsView", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
  });

  function selectedTab() {
    return screen.getAllByRole("tab").find((tab) => tab.getAttribute("aria-selected") === "true");
  }

  it("mặc định mở câu tiêu biểu của Cơ sở vật chất", () => {
    render(<InsightsView data={makeInsights()} groupLabel="Môn học" />);
    expect(selectedTab()).toHaveTextContent("Cơ sở vật chất");
  });

  it("'Xem câu liên quan' của cảnh báo theo chủ đề mở đúng tab và cuộn tới câu tiêu biểu", async () => {
    render(<InsightsView data={makeInsights()} groupLabel="Môn học" />);
    await userEvent.click(screen.getAllByRole("button", { name: /Xem câu liên quan/ })[0]);
    expect(selectedTab()).toHaveTextContent("Chương trình");
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
  });

  it("cảnh báo của một nhóm mở tab và lọc câu theo nhóm đó, bỏ lọc được", async () => {
    render(<InsightsView data={makeInsights()} groupLabel="Môn học" />);
    await userEvent.click(screen.getAllByRole("button", { name: /Xem câu liên quan/ })[1]);
    expect(selectedTab()).toHaveTextContent("Giảng viên");
    expect(screen.getByText("Nhóm: Nhóm A", { selector: "span.inline-flex" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Bỏ lọc nhóm Nhóm A" }));
    expect(screen.queryByText("Đang lọc:")).not.toBeInTheDocument();
  });

  it("chọn từ khoá chuyển sang chủ đề của từ đó; tự đổi tab thì bỏ lọc từ khoá của chủ đề khác", async () => {
    render(<InsightsView data={makeInsights()} groupLabel="Môn học" />);
    await userEvent.click(screen.getByRole("button", { name: /^thực hành\s*34 câu$/ }));
    expect(selectedTab()).toHaveTextContent("Chương trình");
    expect(screen.getByRole("button", { name: "Bỏ lọc từ khoá thực hành" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Cơ sở vật chất" }));
    expect(screen.queryByRole("button", { name: "Bỏ lọc từ khoá thực hành" })).not.toBeInTheDocument();
    expect(within(screen.getByRole("tabpanel")).getAllByRole("listitem").length).toBeGreaterThan(0);
  });

  it("không có cột nhóm thì không có heatmap và có chú thích lý do", () => {
    render(<InsightsView data={makeInsights({ heatmap: null, groups: null, alerts: [] })} groupLabel="nhóm" />);
    expect(screen.queryByRole("table", { name: /theo nhóm/ })).not.toBeInTheDocument();
    expect(screen.getByText(/Heatmap nhóm × chủ đề chỉ hiện khi file có cột nhóm/)).toBeInTheDocument();
  });
});

