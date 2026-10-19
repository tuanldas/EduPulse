import type { InsightsResponse, SentimentLabel, TopicLabel } from "@/lib/batch/types";
import { LABEL_ORDER } from "@/lib/labels";

type Topics = InsightsResponse["topics"];
type Alerts = InsightsResponse["alerts"];
type Heatmap = NonNullable<InsightsResponse["heatmap"]>;
type Quotes = InsightsResponse["quotes"][number]["quotes"];

// Luật cảnh báo của máy chủ, chỉ để viết lại thành chú thích cho người đọc; giá trị thật nằm ở API.
/** Số câu tối thiểu của một chủ đề (hoặc ô nhóm × chủ đề) để được xét cảnh báo. */
export const ALERT_MIN_N = 20;
/** Cận dưới khoảng tin cậy 95% phải cao hơn mức nền ít nhất chừng này điểm. */
export const ALERT_MARGIN_POINTS = 10;

// --- KPI ---------------------------------------------------------------------------------------

export type KpiMetric = "negative" | "positive";

export interface KpiDelta {
  /** `bad`: lệch theo hướng đáng chú ý (tô đỏ); `neutral`: lệch có lợi hoặc không lệch (xám). */
  tone: "bad" | "neutral";
  direction: "up" | "down" | "none";
  /** Độ lệch tuyệt đối đã làm tròn, đơn vị điểm phần trăm. */
  points: number;
  /** Cụm chữ đứng sau số điểm. */
  relation: "so với nền" | "dưới nền" | "trên nền" | "ngang mức nền";
}

/**
 * Chip so với mức nền ở KPI. Chỉ tô đỏ khi lệch theo hướng xấu (tiêu cực cao hơn nền, tích cực thấp
 * hơn nền); lệch có lợi hiển thị trung tính, không khen: dashboard mô tả dữ liệu, không phán xét.
 */
export function kpiDelta(metric: KpiMetric, deltaPoints: number, digits = 1): KpiDelta {
  const factor = 10 ** digits;
  const points = Math.round(Math.abs(deltaPoints) * factor) / factor;
  if (points === 0) return { tone: "neutral", direction: "none", points: 0, relation: "ngang mức nền" };

  const above = deltaPoints > 0;
  const bad = metric === "negative" ? above : !above;
  if (bad) {
    return { tone: "bad", direction: above ? "up" : "down", points, relation: "so với nền" };
  }
  return {
    tone: "neutral",
    direction: metric === "negative" ? "none" : "up",
    points,
    relation: metric === "negative" ? "dưới nền" : "trên nền",
  };
}

// --- Cột chồng cảm xúc theo chủ đề -------------------------------------------------------------

/** Đoạn có tỉ lệ nhỏ hơn mức này không ghi số trong ô (chật), số vẫn có trong chú thích văn bản. */
export const SEGMENT_LABEL_MIN_SHARE = 0.07;

export interface BarSegment {
  sentiment: SentimentLabel;
  share: number;
  showLabel: boolean;
}

export interface TopicBar {
  topic: TopicLabel;
  n: number;
  segments: BarSegment[];
  /** Vị trí vạch mức nền tiêu cực, tỉ lệ 0..1 tính từ mép trái. */
  baseline: number;
  /** Chủ đề chưa có câu nào: không vẽ cột. */
  empty: boolean;
}

export function topicBars(topics: Topics): TopicBar[] {
  return topics.map((stat) => ({
    topic: stat.topic,
    n: stat.n,
    empty: stat.n === 0,
    baseline: stat.baseline_negative_rate,
    segments: LABEL_ORDER.sentiment.map((sentiment) => {
      const share = stat.sentiment_shares[sentiment];
      return { sentiment, share, showLabel: share >= SEGMENT_LABEL_MIN_SHARE };
    }),
  }));
}

// --- Chủ đề không có cảnh báo ------------------------------------------------------------------

export interface CalmTopic {
  topic: TopicLabel;
  n: number;
  /** Tỉ lệ tiêu cực của chủ đề, 0..1. */
  negativeRate: number;
  baselineRate: number;
  /** `enough`: đủ câu mà không vượt nền; `few`: chưa đủ câu để kết luận. */
  reason: "enough" | "few";
}

/**
 * Chủ đề không có cảnh báo cấp toàn tập. Phải tách "bình thường" khỏi "chưa đủ câu": thiếu câu
 * không có nghĩa là ổn, nên giao diện không được viết "trong mức bình thường" cho trường hợp đó.
 */
export function calmTopics(topics: Topics, alerts: Alerts, minN: number = ALERT_MIN_N): CalmTopic[] {
  const alerted = new Set(
    alerts.filter((alert) => alert.group === null).map((alert) => alert.topic),
  );
  return topics
    .filter((stat) => !alerted.has(stat.topic))
    .map((stat) => ({
      topic: stat.topic,
      n: stat.n,
      negativeRate: stat.sentiment_shares.negative,
      baselineRate: stat.baseline_negative_rate,
      reason: stat.n >= minN ? "enough" : "few",
    }));
}

// --- Heatmap -----------------------------------------------------------------------------------

export type CellKind =
  /** Tô màu theo chênh so với nền. */
  | "shaded"
  /** Có câu nhưng chưa đủ (n < min_n): không tô, viền nét đứt, vẫn ghi tỉ lệ và n. */
  | "sparse"
  /** Không có câu nào. */
  | "empty";

export interface CellView {
  kind: CellKind;
  n: number;
  /** Tỉ lệ tiêu cực 0..1; null khi n = 0. */
  rate: number | null;
  /** Chênh so với nền của chủ đề, điểm phần trăm; chỉ có khi `kind` là `shaded`. */
  delta: number | null;
}

export interface HeatmapRow {
  group: string;
  cells: CellView[];
}

export function heatmapRows(heatmap: Heatmap): HeatmapRow[] {
  return heatmap.groups.map((group, rowIndex) => ({
    group,
    cells: heatmap.topics.map((_, columnIndex): CellView => {
      const cell = heatmap.cells[rowIndex]?.[columnIndex];
      if (cell === undefined || cell.n === 0 || cell.negative_rate === null) {
        return { kind: "empty", n: cell?.n ?? 0, rate: null, delta: null };
      }
      if (!cell.shaded || cell.delta_points === null) {
        return { kind: "sparse", n: cell.n, rate: cell.negative_rate, delta: null };
      }
      return { kind: "shaded", n: cell.n, rate: cell.negative_rate, delta: cell.delta_points };
    }),
  }));
}

// --- Từ khoá và câu tiêu biểu ------------------------------------------------------------------

/** Thứ tự khối từ khoá trên trang (chủ đề "Khác" không có từ khoá). */
export const KEYWORD_TOPIC_ORDER = [
  "facility",
  "training_program",
  "lecturer",
] as const satisfies readonly TopicLabel[];

export interface QuoteFilter {
  term: string | null;
  group: string | null;
}

export const NO_QUOTE_FILTER: QuoteFilter = { term: null, group: null };

export function filterQuotes(quotes: Quotes, filter: QuoteFilter): Quotes {
  return quotes.filter(
    (quote) =>
      (filter.term === null || quote.matched.includes(filter.term)) &&
      (filter.group === null || quote.group === filter.group),
  );
}
