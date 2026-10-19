// Các số mô tả suy ra từ model card (không có số nào được ghi cứng): lớp yếu nhất, cách lớp yếu bị
// đoán nhầm, hàng ngân sách mặc định. Thuần logic để test được.
import type { ModelCard } from "@/lib/api/types";

type TaskMetrics = ModelCard["metrics"]["sentiment"];
type BudgetRow = ModelCard["review_budget"][number];

export interface WeakestClass {
  label: string;
  f1: number;
  recall: number;
  /** Tỉ lệ số câu của lớp này trong tập đánh giá (0..1). */
  share: number;
}

/** Lớp có F1 thấp nhất trong số các lớp có câu; bằng nhau thì lấy lớp đứng trước. */
export function weakestClass(task: TaskMetrics): WeakestClass | undefined {
  const total = task.per_class.reduce((sum, item) => sum + item.support, 0);
  const present = task.per_class.filter((item) => item.support > 0);
  if (present.length === 0 || total === 0) return undefined;
  const weakest = present.reduce((best, item) => (item.f1 < best.f1 ? item : best));
  return { label: weakest.label, f1: weakest.f1, recall: weakest.recall, share: weakest.support / total };
}

export interface Misclassification {
  /** Lớp bị đoán nhầm thành và tỉ lệ (0..1), nhiều nhất trước. */
  into: { label: string; share: number }[];
}

/**
 * Câu của lớp `label` bị đoán nhầm thành những lớp nào (ma trận chuẩn hoá theo hàng). Chỉ giữ ô nhầm
 * từ `minShare` trở lên để câu mô tả không liệt kê toàn số rất nhỏ.
 */
export function misclassifiedInto(task: TaskMetrics, label: string, minShare = 0.05): Misclassification {
  const labels = task.per_class.map((item) => item.label);
  const row = task.confusion[labels.indexOf(label)] ?? [];
  const into = row
    .map((share, column) => ({ label: labels[column], share }))
    .filter((cell) => cell.label !== label && cell.share >= minShare)
    .sort((a, b) => b.share - a.share);
  return { into };
}

/** Hàng ngân sách duyệt mặc định (đánh dấu `default`), nếu model card có. */
export function defaultBudgetRow(card: Pick<ModelCard, "review_budget">): BudgetRow | undefined {
  return card.review_budget.find((row) => row.default);
}

/** Tỉ lệ lỗi được bắt ở ngân sách của một tác vụ. */
export function caughtErrors(row: BudgetRow, task: "sentiment" | "topic"): number {
  return task === "sentiment" ? row.efficiency.caught_sentiment_errors : row.efficiency.caught_topic_errors;
}
