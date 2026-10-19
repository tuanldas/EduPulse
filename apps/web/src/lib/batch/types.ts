import type { components } from "@/lib/api/schema";
import type { SentimentLabel, TopicLabel } from "@/lib/api/types";
import { LABEL_ORDER } from "@/lib/labels";

export type {
  ExplainResponse,
  FlagReason,
  PiiType,
  SentimentLabel,
  TopicLabel,
} from "@/lib/api/types";

type Schemas = components["schemas"];

// Kiểu của phân tích hàng loạt và dashboard: bí danh từ schema OpenAPI, không khai báo lại hình dạng.

export type BatchUpload = Schemas["BatchUpload"];
export type BatchWarning = Schemas["BatchWarning"];
export type BatchStatus = Schemas["BatchStatus"];
export type BatchState = BatchStatus["state"];
export type BatchStep = NonNullable<BatchStatus["step"]>;

export type ResultsResponse = Schemas["ResultsResponse"];
export type ResultsSummary = Schemas["ResultsSummary"];
export type ResultRow = Schemas["ResultRow"];
export type SkippedRow = Schemas["SkippedRow"];

export type RunRequest = Schemas["RunRequest"];
export type ExportRequest = Schemas["ExportRequest"];
export type LabelOverride = Schemas["LabelOverride"];

export type ReviewBudget = RunRequest["review_budget"];

export type BudgetOption = Schemas["BudgetOption"];
export type BudgetEfficiency = Schemas["BudgetEfficiency"];

export type InsightsResponse = Schemas["InsightsResponse"];

// Thứ tự cố định của nhãn (khớp `edupulse.labels`), dùng chung với phần còn lại của giao diện.
export const SENTIMENT_LABELS: readonly SentimentLabel[] = LABEL_ORDER.sentiment;
export const TOPIC_LABELS: readonly TopicLabel[] = LABEL_ORDER.topic;

export function isSentimentLabel(value: unknown): value is SentimentLabel {
  return typeof value === "string" && SENTIMENT_LABELS.some((label) => label === value);
}

export function isTopicLabel(value: unknown): value is TopicLabel {
  return typeof value === "string" && TOPIC_LABELS.some((label) => label === value);
}

/** Id của batch chỉ đọc dựng sẵn từ dữ liệu demo (khớp `DEMO_BATCH_ID` ở API). */
export const DEMO_BATCH_ID = "demo";

/** Batch đang chờ hoặc đang chạy: chỉ lúc này mới cần hỏi tiến độ. */
export function isActiveState(state: BatchState): boolean {
  return state === "queued" || state === "running";
}
