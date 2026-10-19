import { describe, expect, it } from "vitest";

import {
  calmTopics,
  filterQuotes,
  groupSummary,
  heatmapRows,
  kpiDelta,
  NO_QUOTE_FILTER,
  topicBars,
  TOPIC_SHORT,
} from "@/lib/dashboard/insight-view";

import { makeAlert, makeHeatmap, makeQuote, makeTopicStat } from "./helpers/insights";

describe("kpiDelta", () => {
  it("tiêu cực cao hơn nền: đỏ, mũi tên lên", () => {
    expect(kpiDelta("negative", 7.46)).toEqual({
      tone: "bad",
      direction: "up",
      points: 7.5,
      relation: "so với nền",
    });
  });

  it("tích cực thấp hơn nền: đỏ, mũi tên xuống", () => {
    expect(kpiDelta("positive", -9.04)).toEqual({
      tone: "bad",
      direction: "down",
      points: 9,
      relation: "so với nền",
    });
  });

  it("lệch có lợi chỉ hiển thị trung tính, không khen", () => {
    expect(kpiDelta("negative", -2.1)).toEqual({
      tone: "neutral",
      direction: "none",
      points: 2.1,
      relation: "dưới nền",
    });
    expect(kpiDelta("positive", 3)).toEqual({
      tone: "neutral",
      direction: "up",
      points: 3,
      relation: "trên nền",
    });
  });

  it("làm tròn về 0 thì coi là ngang mức nền", () => {
    expect(kpiDelta("negative", 0.04)).toEqual({
      tone: "neutral",
      direction: "none",
      points: 0,
      relation: "ngang mức nền",
    });
    expect(kpiDelta("positive", -0.02).relation).toBe("ngang mức nền");
    expect(kpiDelta("positive", 0).tone).toBe("neutral");
  });
});

describe("topicBars", () => {
  it("giữ thứ tự tiêu cực → trung tính → tích cực và vị trí vạch nền của từng hàng", () => {
    const bars = topicBars([
      makeTopicStat({ topic: "lecturer", baseline_negative_rate: 0.356 }),
      makeTopicStat({
        topic: "facility",
        n: 90,
        baseline_negative_rate: 0.954,
        sentiment_shares: { negative: 0.956, neutral: 0.011, positive: 0.033 },
      }),
    ]);
    expect(bars.map((bar) => bar.baseline)).toEqual([0.356, 0.954]);
    expect(bars[0].segments.map((segment) => segment.sentiment)).toEqual([
      "negative",
      "neutral",
      "positive",
    ]);
    expect(bars[1].n).toBe(90);
  });

  it("chỉ ghi số trong đoạn đủ rộng", () => {
    const [bar] = topicBars([
      makeTopicStat({ sentiment_shares: { negative: 0.956, neutral: 0.011, positive: 0.033 } }),
    ]);
    expect(bar.segments.map((segment) => segment.showLabel)).toEqual([true, false, false]);
  });

  it("đánh dấu chủ đề chưa có câu", () => {
    const [bar] = topicBars([
      makeTopicStat({ n: 0, sentiment_shares: { negative: 0, neutral: 0, positive: 0 } }),
    ]);
    expect(bar.empty).toBe(true);
    expect(bar.segments.every((segment) => !segment.showLabel)).toBe(true);
  });
});

describe("calmTopics", () => {
  const topics = [
    makeTopicStat({ topic: "lecturer", n: 600 }),
    makeTopicStat({ topic: "training_program", n: 195 }),
    makeTopicStat({ topic: "facility", n: 9 }),
    makeTopicStat({ topic: "others", n: 115 }),
  ];

  it("bỏ chủ đề đã có cảnh báo cấp toàn tập", () => {
    const calm = calmTopics(topics, [makeAlert({ topic: "training_program", group: null })]);
    expect(calm.map((item) => item.topic)).toEqual(["lecturer", "facility", "others"]);
  });

  it("cảnh báo riêng một nhóm không che chủ đề khỏi danh sách bình thường", () => {
    const calm = calmTopics(topics, [makeAlert({ topic: "lecturer", group: "Nhóm A" })]);
    expect(calm.map((item) => item.topic)).toContain("lecturer");
  });

  it("chủ đề ít câu là chưa đủ câu chứ không phải bình thường", () => {
    const calm = calmTopics(topics, []);
    expect(calm.find((item) => item.topic === "facility")?.reason).toBe("few");
    expect(calm.find((item) => item.topic === "lecturer")?.reason).toBe("enough");
  });
});

describe("heatmapRows", () => {
  const rows = heatmapRows(makeHeatmap());

  it("mỗi hàng là một nhóm, mỗi cột là một chủ đề", () => {
    expect(rows.map((row) => row.group)).toEqual(["Nhóm A", "Nhóm B"]);
    expect(rows.every((row) => row.cells.length === 4)).toBe(true);
  });

  it("ô đủ câu được tô và mang chênh lệch so với nền", () => {
    expect(rows[0].cells[0]).toEqual({ kind: "shaded", n: 120, rate: 0.58, delta: 22.4 });
  });

  it("ô thiếu câu không tô và không nêu chênh lệch, dù API có gửi số", () => {
    expect(rows[0].cells[2]).toEqual({ kind: "sparse", n: 12, rate: 0.92, delta: null });
    expect(rows[1].cells[2]).toEqual({ kind: "sparse", n: 10, rate: 1, delta: null });
  });

  it("ô không có câu nào là ô trống", () => {
    expect(rows[0].cells[3]).toEqual({ kind: "empty", n: 0, rate: null, delta: null });
  });

  it("thiếu ô trong dữ liệu thì coi là trống thay vì vỡ", () => {
    const heatmap = makeHeatmap();
    heatmap.cells[1] = [];
    expect(heatmapRows(heatmap)[1].cells.map((cell) => cell.kind)).toEqual([
      "empty",
      "empty",
      "empty",
      "empty",
    ]);
  });
});

describe("filterQuotes", () => {
  const quotes = [
    makeQuote({ text: "a", matched: ["máy chiếu", "mờ"], group: "Nhóm A" }),
    makeQuote({ text: "b", matched: ["mờ"], group: "Nhóm B" }),
    makeQuote({ text: "c", matched: [], group: null }),
  ];

  it("không lọc thì giữ nguyên", () => {
    expect(filterQuotes(quotes, NO_QUOTE_FILTER)).toEqual(quotes);
  });

  it("lọc theo từ khoá có trong câu", () => {
    expect(filterQuotes(quotes, { term: "mờ", group: null }).map((q) => q.text)).toEqual([
      "a",
      "b",
    ]);
    expect(filterQuotes(quotes, { term: "máy chiếu", group: null }).map((q) => q.text)).toEqual([
      "a",
    ]);
  });

  it("lọc theo nhóm và kết hợp với từ khoá", () => {
    expect(filterQuotes(quotes, { term: null, group: "Nhóm B" }).map((q) => q.text)).toEqual(["b"]);
    expect(filterQuotes(quotes, { term: "máy chiếu", group: "Nhóm B" })).toEqual([]);
  });
});

describe("groupSummary", () => {
  it("không chọn gì hoặc chọn hết là tất cả", () => {
    expect(groupSummary([], 6)).toBe("Tất cả 6 nhóm");
    expect(groupSummary(["A", "B", "C"], 3)).toBe("Tất cả 3 nhóm");
  });

  it("một hoặc hai nhóm thì ghi tên", () => {
    expect(groupSummary(["Nhập môn lập trình"], 6)).toBe("Nhập môn lập trình");
    expect(groupSummary(["A", "B"], 6)).toBe("A, B");
  });

  it("từ ba nhóm trở lên thì ghi số lượng", () => {
    expect(groupSummary(["A", "B", "C"], 6)).toBe("3/6 nhóm");
  });
});

describe("TOPIC_SHORT", () => {
  it("rút gọn chủ đề dài nhất cho chỗ hẹp", () => {
    expect(TOPIC_SHORT.facility).toBe("CSVC");
    expect(TOPIC_SHORT.lecturer).toBe("Giảng viên");
  });
});
