import type { FlagReason } from "@/lib/api/types";
import { formatPercent } from "@/lib/format";
import { labelMeta, TASK_META } from "@/lib/labels";

export interface ReasonParts {
  /** Phần in đậm đầu câu lý do, vd. "Chủ đề không chắc". */
  title: string;
  /** Phần giải thích đi sau, vd. "— mô hình phân vân giữa Giảng viên (58%) và Cơ sở vật chất (31%)." */
  detail: string;
}

/**
 * Lý do gắn cờ bằng lời, ghép từ dữ liệu có cấu trúc của API (đầu ra không chắc + hai lớp cao nhất)
 * để giọng văn thống nhất; mã lạ thì dùng nguyên `message` của máy chủ.
 */
export function reasonParts(reason: FlagReason): ReasonParts {
  if (reason.code === "no_diacritics") {
    return { title: "Câu không dấu", detail: "(luôn gắn cờ, kể cả khi độ tin cậy cao)" };
  }
  if (reason.code === "low_confidence" && reason.task) {
    const names = reason.candidates.map(
      (candidate) => `${labelMeta(reason.task as "sentiment" | "topic", candidate.label).vi} (${formatPercent(candidate.probability)})`,
    );
    const detail =
      names.length >= 2
        ? `— mô hình phân vân giữa ${names[0]} và ${names[1]}.`
        : names.length === 1
          ? `— mô hình nghiêng về ${names[0]}.`
          : "";
    return { title: `${TASK_META[reason.task].vi} không chắc`, detail };
  }
  return { title: reason.message, detail: "" };
}

/** Đầu ra (cảm xúc/chủ đề) nào có độ tin cậy dưới ngưỡng cờ, theo các lý do của API. */
export function belowThreshold(reasons: readonly FlagReason[], task: "sentiment" | "topic"): boolean {
  return reasons.some((reason) => reason.code === "low_confidence" && reason.task === task);
}
