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

type Kpis = InsightsResponse["kpis"];
type Recommendation = InsightsResponse["recommendations"][number];

export function makeKpis(overrides: Partial<Kpis> = {}): Kpis {
  return {
    n_rows: 1000,
    n_groups: 6,
    shares: { negative: 0.541, neutral: 0.055, positive: 0.404 },
    baseline_shares: { negative: 0.466, neutral: 0.04, positive: 0.494 },
    delta_points: { negative: 7.46, neutral: 1.5, positive: -9.04 },
    n_flagged: 100,
    flagged_share: 0.1,
    ...overrides,
  };
}

export function makeRecommendation(overrides: Partial<Recommendation> = {}): Recommendation {
  return {
    id: "facility_projector",
    topic: "facility",
    action: "Bảo trì thiết bị trình chiếu và âm thanh ở phòng học",
    evidence: "Căn cứ: 22 câu tiêu cực về Cơ sở vật chất nhắc “máy chiếu”, “mờ”.",
    n: 22,
    terms: ["máy chiếu", "mờ"],
    note: null,
    ...overrides,
  };
}

/** Phản hồi insight đầy đủ cho test `InsightsView`: có nhóm, heatmap, ba cảnh báo, từ khoá, câu và khuyến nghị. */
export function makeInsights(overrides: Partial<InsightsResponse> = {}): InsightsResponse {
  return {
    source: "batch",
    illustrative_groups: false,
    groups: [
      { name: "Nhóm A", n: 60 },
      { name: "Nhóm B", n: 40 },
    ],
    selected_groups: null,
    kpis: makeKpis(),
    topics: [
      makeTopicStat({ topic: "lecturer", n: 600, sentiment_shares: { negative: 0.38, neutral: 0.03, positive: 0.59 } }),
      makeTopicStat({
        topic: "training_program",
        n: 195,
        baseline_negative_rate: 0.78,
        sentiment_shares: { negative: 0.933, neutral: 0.031, positive: 0.036 },
      }),
      makeTopicStat({
        topic: "facility",
        n: 90,
        baseline_negative_rate: 0.954,
        sentiment_shares: { negative: 0.956, neutral: 0.011, positive: 0.033 },
      }),
      makeTopicStat({
        topic: "others",
        n: 9,
        baseline_negative_rate: 0.402,
        sentiment_shares: { negative: 0.4, neutral: 0.3, positive: 0.3 },
      }),
    ],
    alerts: [
      makeAlert(),
      makeAlert({
        topic: "lecturer",
        group: "Nhóm A",
        title: "Giảng dạy ở Nhóm A: tiêu cực cao hơn mức nền",
        summary: "58% góp ý về giảng dạy ở nhóm này là tiêu cực (70/120 câu), cao hơn khoảng 23 điểm.",
        evidence: { n: 120, k: 70, rate: 0.583, ci_low: 0.49, baseline: 0.356, delta_points: 22.7 },
        note: "Tổng hợp theo nhóm, không gắn với cá nhân.",
      }),
      makeAlert({
        kind: "topic_share",
        topic: "facility",
        title: "Cơ sở vật chất được nhắc nhiều hơn thường lệ",
        summary: "9% góp ý nói về Cơ sở vật chất (90/1.000 câu), gấp đôi mức thường gặp 4,4%.",
        evidence: { n: 1000, k: 90, rate: 0.09, ci_low: 0.074, baseline: 0.044, delta_points: 4.6 },
        keywords: ["máy chiếu", "phòng máy", "quạt", "nóng"],
      }),
    ],
    heatmap: makeHeatmap(),
    keywords: [
      {
        topic: "facility",
        n_negative: 86,
        keywords: [
          { term: "máy chiếu", n: 19, score: 9 },
          { term: "mờ", n: 9, score: 8 },
        ],
      },
      { topic: "training_program", n_negative: 182, keywords: [{ term: "thực hành", n: 34, score: 7 }] },
      { topic: "lecturer", n_negative: 228, keywords: [{ term: "nhanh", n: 15, score: 6 }] },
    ],
    quotes: [
      {
        topic: "facility",
        quotes: [
          makeQuote({ text: "Máy chiếu quá mờ, thiết bị cũ.", matched: ["máy chiếu", "mờ"], group: "Nhóm A" }),
          makeQuote({ text: "Phòng học cần rèm để máy chiếu rõ.", matched: ["máy chiếu"], group: "Nhóm B" }),
          makeQuote({ text: "Quạt hỏng, phòng nóng.", matched: [], group: "Nhóm A" }),
        ],
      },
      {
        topic: "training_program",
        quotes: [makeQuote({ text: "Môn này nặng lý thuyết, thiếu thực hành.", matched: ["thực hành"], group: "Nhóm B" })],
      },
      { topic: "lecturer", quotes: [] },
      { topic: "others", quotes: [] },
    ],
    recommendations: [makeRecommendation(), makeRecommendation({ id: "chua_biet", topic: "lecturer", action: "Việc mới" })],
    ...overrides,
  };
}
