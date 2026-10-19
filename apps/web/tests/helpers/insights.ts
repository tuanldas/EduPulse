import type { InsightsResponse, TopicLabel } from "@/lib/batch/types";

type TopicStat = InsightsResponse["topics"][number];
type Alert = InsightsResponse["alerts"][number];
type Heatmap = NonNullable<InsightsResponse["heatmap"]>;
type Quote = InsightsResponse["quotes"][number]["quotes"][number];

// Số liệu tự dựng cho test của logic hiển thị, không phải kết quả của mô hình.

export function makeTopicStat(overrides: Partial<TopicStat> = {}): TopicStat {
  return {
    topic: "lecturer",
    n: 100,
    share: 0.5,
    baseline_share: 0.5,
    sentiment_shares: { negative: 0.4, neutral: 0.1, positive: 0.5 },
    negative_n: 40,
    baseline_negative_rate: 0.356,
    negative_ci_low: 0.31,
    ...overrides,
  };
}

export function makeAlert(overrides: Partial<Alert> = {}): Alert {
  return {
    kind: "negative_rate",
    topic: "training_program",
    group: null,
    title: "Chương trình: tỉ lệ tiêu cực cao hơn mức nền",
    summary: "93% góp ý về Chương trình là tiêu cực (182/195 câu).",
    evidence: { n: 195, k: 182, rate: 0.933, ci_low: 0.89, baseline: 0.78, delta_points: 15.3 },
    keywords: ["thực hành"],
    note: null,
    highlight: null,
    ...overrides,
  };
}

export function makeHeatmap(): Heatmap {
  const topics: TopicLabel[] = ["lecturer", "training_program", "facility", "others"];
  return {
    groups: ["Nhóm A", "Nhóm B"],
    topics,
    baseline_negative_rate: [0.356, 0.78, 0.954, 0.402],
    min_n: 20,
    cells: [
      [
        { n: 120, negative_rate: 0.58, delta_points: 22.4, shaded: true },
        { n: 30, negative_rate: 0.9, delta_points: 12, shaded: true },
        { n: 12, negative_rate: 0.92, delta_points: -3.4, shaded: false },
        { n: 0, negative_rate: null, delta_points: null, shaded: false },
      ],
      [
        { n: 110, negative_rate: 0.35, delta_points: -0.6, shaded: true },
        { n: 48, negative_rate: 0.98, delta_points: 20, shaded: true },
        { n: 10, negative_rate: 1, delta_points: 4.6, shaded: false },
        { n: 22, negative_rate: 0.41, delta_points: 0.8, shaded: true },
      ],
    ],
  };
}

export function makeQuote(overrides: Partial<Quote> = {}): Quote {
  return {
    text: "Máy chiếu quá mờ.",
    sentiment: "negative",
    confidence: 0.97,
    group: "Nhóm A",
    matched: ["máy chiếu", "mờ"],
    ...overrides,
  };
}
