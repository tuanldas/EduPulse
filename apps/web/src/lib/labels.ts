// Ánh xạ nhãn tiếng Anh của API sang thuật ngữ tiếng Việt cố định, icon và màu ngữ nghĩa
// (docs/design-guidelines.md mục 2.2 và 8). Màu không bao giờ là tín hiệu duy nhất: luôn có icon + chữ.
import type { IconName } from "@/components/shared/icon-paths";
import type { PiiType, SentimentLabel, TopicLabel } from "@/lib/api/types";

export type Task = "sentiment" | "topic";

/** Khoá màu ngữ nghĩa; lớp CSS tương ứng ở `lib/colors.ts`. */
export type Tone = "neg" | "neu" | "pos" | "gv" | "ct" | "csvc" | "khac" | "flag";

export interface LabelMeta {
  vi: string;
  icon: IconName;
  tone: Tone;
}

export const SENTIMENT_META: Record<SentimentLabel, LabelMeta> = {
  negative: { vi: "Tiêu cực", icon: "sentiment_dissatisfied", tone: "neg" },
  neutral: { vi: "Trung tính", icon: "sentiment_neutral", tone: "neu" },
  positive: { vi: "Tích cực", icon: "sentiment_satisfied", tone: "pos" },
};

export const TOPIC_META: Record<TopicLabel, LabelMeta> = {
  lecturer: { vi: "Giảng viên", icon: "co_present", tone: "gv" },
  training_program: { vi: "Chương trình", icon: "menu_book", tone: "ct" },
  facility: { vi: "Cơ sở vật chất", icon: "domain", tone: "csvc" },
  others: { vi: "Khác", icon: "more_horiz", tone: "khac" },
};

/** Thứ tự nhãn của từng tác vụ, khớp `edupulse.labels` (thứ tự lớp của bundle). */
export const LABEL_ORDER: { sentiment: readonly SentimentLabel[]; topic: readonly TopicLabel[] } = {
  sentiment: ["negative", "neutral", "positive"],
  topic: ["lecturer", "training_program", "facility", "others"],
};

export const TASK_META: Record<Task, { vi: string; icon: IconName }> = {
  sentiment: { vi: "Cảm xúc", icon: "mood" },
  topic: { vi: "Chủ đề", icon: "category" },
};

const UNKNOWN: LabelMeta = { vi: "", icon: "help", tone: "neu" };

/** Nhãn lạ (API thêm lớp mới) vẫn hiển thị được bằng chính tên của nó. */
export function labelMeta(task: Task, label: string): LabelMeta {
  const table: Record<string, LabelMeta> = task === "sentiment" ? SENTIMENT_META : TOPIC_META;
  return table[label] ?? { ...UNKNOWN, vi: label };
}

/** Thẻ thay thế thông tin cá nhân trong câu đã ẩn (`display_text`). */
export const PII_TAG: Record<PiiType, string> = {
  name: "[TÊN]",
  phone: "[SĐT]",
  email: "[EMAIL]",
  student_id: "[MSSV]",
  national_id: "[CCCD]",
  url: "[URL]",
};

/** Thứ tự hiển thị các loại thông tin cá nhân. */
export const PII_ORDER: readonly PiiType[] = [
  "name",
  "phone",
  "email",
  "student_id",
  "national_id",
  "url",
];
