// Màu cho biểu đồ và nhãn. Màu giao diện nằm ở `app/globals.css` (biến CSS theo theme); file này
// giữ (1) lớp Tailwind theo khoá màu ngữ nghĩa và (2) các thang màu tường minh cho confusion
// matrix và heatmap, vốn KHÔNG đổi theo theme (docs/design-guidelines.md mục 2.2).
// Mã hex chỉ được phép nằm ở hai nơi này.
import type { Tone } from "@/lib/labels";

/** Tên lớp đầy đủ (không ghép chuỗi) để Tailwind nhìn thấy khi quét mã nguồn. */
export const TONE_CLASS: Record<Tone, { badge: string; text: string; solid: string }> = {
  neg: { badge: "bg-neg-bg text-neg-fg", text: "text-neg-fg", solid: "bg-neg" },
  neu: { badge: "bg-neu-bg text-neu-fg", text: "text-neu-fg", solid: "bg-neu" },
  pos: { badge: "bg-pos-bg text-pos-fg", text: "text-pos-fg", solid: "bg-pos" },
  gv: { badge: "bg-gv-bg text-gv-fg", text: "text-gv-fg", solid: "bg-gv" },
  ct: { badge: "bg-ct-bg text-ct-fg", text: "text-ct-fg", solid: "bg-ct" },
  csvc: { badge: "bg-csvc-bg text-csvc-fg", text: "text-csvc-fg", solid: "bg-csvc" },
  khac: { badge: "bg-khac-bg text-khac-fg", text: "text-khac-fg", solid: "bg-khac" },
  flag: { badge: "bg-flag-bg text-flag-fg", text: "text-flag-fg", solid: "bg-flag" },
};

/** Giá trị CSS của màu nền đặc (SVG, style nội tuyến). */
export function toneVar(tone: Tone): string {
  return `var(--${tone})`;
}

const DARK_TEXT = "#14171E";
const LIGHT_TEXT = "#FFFFFF";

export interface CellColor {
  background: string;
  color: string;
}

// Thang tuần tự primary 10 bậc, dùng cho confusion matrix. Chữ tối ở bậc 1–7, chữ trắng ở 8–10.
export const SEQUENTIAL_SCALE = [
  "#F0F3FF",
  "#DCE3FF",
  "#C7D2FF",
  "#AEBDFC",
  "#97A8F7",
  "#8193F1",
  "#6A7BE5",
  "#5664D3",
  "#444EBB",
  "#343A9F",
] as const;

/** Bậc màu của một tỉ lệ 0..1: mười khoảng đều nhau, 0 vẫn là bậc 1 (`ceil(p·10)`, tối thiểu 1). */
export function sequentialStep(ratio: number): number {
  const clamped = Math.min(1, Math.max(0, Number.isFinite(ratio) ? ratio : 0));
  return Math.min(SEQUENTIAL_SCALE.length, Math.max(1, Math.ceil(clamped * SEQUENTIAL_SCALE.length)));
}

export function sequentialColor(ratio: number): CellColor {
  const step = sequentialStep(ratio);
  return {
    background: SEQUENTIAL_SCALE[step - 1],
    color: step >= 8 ? LIGHT_TEXT : DARK_TEXT,
  };
}

// Heatmap "chênh so với mức nền" (điểm %): ≤ 0 · 0–5 · 5–10 · 10–15 · 15–20 · ≥ 20.
export const DELTA_SCALE = ["#EEF0F2", "#FBEAE4", "#F5C9B8", "#EDA286", "#E07656", "#B5401F"] as const;

/** Bậc (0..5) của một chênh lệch tính bằng điểm phần trăm. */
export function deltaStep(points: number): number {
  if (!Number.isFinite(points) || points <= 0) return 0;
  return Math.min(DELTA_SCALE.length - 1, Math.floor(points / 5) + 1);
}

export function deltaColor(points: number): CellColor {
  const step = deltaStep(points);
  return {
    background: DELTA_SCALE[step],
    color: step === DELTA_SCALE.length - 1 ? LIGHT_TEXT : DARK_TEXT,
  };
}
