import {
  isSentimentLabel,
  isTopicLabel,
  type LabelOverride,
  type SentimentLabel,
  type TopicLabel,
} from "./types";

/** Nhãn người duyệt đã sửa của một câu; trường vắng nghĩa là giữ nhãn của mô hình. */
export interface LabelPatch {
  sentiment?: SentimentLabel;
  topic?: TopicLabel;
}

/** Chỉ số dòng của batch → nhãn đã sửa. Không bao giờ chứa mục rỗng. */
export type Overrides = Readonly<Record<number, LabelPatch>>;

/** `null` xoá nhãn đã sửa của trường đó, `undefined` giữ nguyên. */
export interface LabelChange {
  sentiment?: SentimentLabel | null;
  topic?: TopicLabel | null;
}

export const NO_OVERRIDES: Overrides = Object.freeze({});

/** Trả bản mới (không sửa bản cũ); mục không còn trường nào thì bị bỏ. */
export function applyChange(current: Overrides, index: number, change: LabelChange): Overrides {
  const patch: LabelPatch = { ...current[index] };
  if (change.sentiment !== undefined) {
    if (change.sentiment === null) delete patch.sentiment;
    else patch.sentiment = change.sentiment;
  }
  if (change.topic !== undefined) {
    if (change.topic === null) delete patch.topic;
    else patch.topic = change.topic;
  }
  const next: Record<number, LabelPatch> = { ...current };
  if (patch.sentiment === undefined && patch.topic === undefined) delete next[index];
  else next[index] = patch;
  return next;
}

/** Danh sách gửi lên API khi xuất file, sắp theo chỉ số dòng để kết quả ổn định. */
export function toOverrideList(overrides: Overrides): LabelOverride[] {
  return Object.keys(overrides)
    .map(Number)
    .sort((a, b) => a - b)
    .map((index) => {
      const patch = overrides[index];
      const item: LabelOverride = { index };
      if (patch.sentiment !== undefined) item.sentiment = patch.sentiment;
      if (patch.topic !== undefined) item.topic = patch.topic;
      return item;
    });
}

/** Số câu có nhãn đã sửa; có `within` thì chỉ đếm trong tập chỉ số đó (vd. danh sách duyệt). */
export function countOverrides(overrides: Overrides, within?: ReadonlySet<number>): number {
  let count = 0;
  for (const key of Object.keys(overrides)) {
    if (within === undefined || within.has(Number(key))) count += 1;
  }
  return count;
}

/** Đọc dữ liệu từ sessionStorage: bỏ mọi mục sai hình dạng thay vì tin nó. */
export function parseOverrides(raw: unknown): Overrides {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return NO_OVERRIDES;
  const result: Record<number, LabelPatch> = {};
  for (const [key, value] of Object.entries(raw)) {
    const index = Number(key);
    if (!Number.isInteger(index) || index < 0) continue;
    if (typeof value !== "object" || value === null) continue;
    const { sentiment, topic } = value as Record<string, unknown>;
    const patch: LabelPatch = {};
    if (isSentimentLabel(sentiment)) patch.sentiment = sentiment;
    if (isTopicLabel(topic)) patch.topic = topic;
    if (patch.sentiment !== undefined || patch.topic !== undefined) result[index] = patch;
  }
  return result;
}
