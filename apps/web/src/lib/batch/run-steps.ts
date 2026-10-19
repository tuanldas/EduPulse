import type { BatchStatus, BatchStep } from "./types";

/**
 * Dòng của danh sách tiến trình (bám wireframe). API chỉ có 4 bước và `prepare` gộp "ẩn thông tin
 * cá nhân" với "tách từ" (cùng một đường tiền xử lý), nên hai dòng đầu cùng đổi trạng thái khi
 * `prepare` đổi; `flag` và `explain` chung một dòng cuối.
 */
export type StepLine = "pii" | "segment" | "predict" | "flag";

export type StepLineState = "pending" | "active" | "done" | "failed";

export const STEP_LINES: readonly StepLine[] = ["pii", "segment", "predict", "flag"];

const API_ORDER: Record<BatchStep, number> = { prepare: 0, predict: 1, flag: 2, explain: 3 };

/** Khoảng bước API (gồm hai đầu) mà mỗi dòng bao phủ. */
const LINE_RANGE: Record<StepLine, readonly [number, number]> = {
  pii: [0, 0],
  segment: [0, 0],
  predict: [1, 1],
  flag: [2, 3],
};

export function stepLineStates(status: BatchStatus | null): Record<StepLine, StepLineState> {
  const states = {} as Record<StepLine, StepLineState>;
  const fill = (state: StepLineState) => {
    for (const line of STEP_LINES) states[line] = state;
    return states;
  };

  if (status === null) return fill("pending");
  if (status.state === "done") return fill("done");
  if (status.step === null) return fill("pending");

  const current = API_ORDER[status.step];
  const reached = status.state === "failed" ? "failed" : "active";
  for (const line of STEP_LINES) {
    const [from, to] = LINE_RANGE[line];
    if (current > to) states[line] = "done";
    else if (current >= from) states[line] = reached;
    else states[line] = "pending";
  }
  return states;
}
