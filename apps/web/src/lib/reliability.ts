// Biểu đồ độ tin cậy (reliability diagram): gom các khoảng mịn của model card thành khoảng rộng
// 0,1 để cột đủ rộng ghi được số câu, và tính các số mô tả. Logic thuần, không phụ thuộc React.
import type { ModelCard } from "@/lib/api/types";

type ReliabilityBin = ModelCard["metrics"]["sentiment"]["reliability"][number];

export interface DisplayBin {
  lower: number;
  upper: number;
  /** Số câu trong khoảng. */
  n: number;
  /** Độ tin cậy trung bình (có trọng số theo số câu); null khi khoảng rỗng. */
  confidence: number | null;
  /** Tỉ lệ đoán đúng thực tế (có trọng số theo số câu); null khi khoảng rỗng. */
  accuracy: number | null;
}

const round3 = (value: number): number => Math.round(value * 1000) / 1000;

/**
 * Gom khoảng mịn (vd. 15 khoảng) thành các khoảng rộng `width`. Mỗi khoảng mịn thuộc về khoảng rộng
 * chứa điểm giữa của nó, nên tổng số câu được bảo toàn.
 */
export function regroupReliability(bins: readonly ReliabilityBin[], width = 0.1): DisplayBin[] {
  const count = Math.ceil(1 / width - 1e-9);
  const groups = Array.from({ length: count }, (_, index) => ({
    lower: round3(index * width),
    upper: round3(Math.min(1, (index + 1) * width)),
    n: 0,
    confidenceSum: 0,
    hitSum: 0,
  }));
  for (const bin of bins) {
    if (bin.n <= 0 || bin.confidence === null || bin.accuracy === null) continue;
    const middle = (bin.lower + bin.upper) / 2;
    const index = Math.min(count - 1, Math.max(0, Math.floor(middle / width + 1e-9)));
    const group = groups[index];
    group.n += bin.n;
    group.confidenceSum += bin.confidence * bin.n;
    group.hitSum += bin.accuracy * bin.n;
  }
  return groups.map((group) => ({
    lower: group.lower,
    upper: group.upper,
    n: group.n,
    confidence: group.n > 0 ? group.confidenceSum / group.n : null,
    accuracy: group.n > 0 ? group.hitSum / group.n : null,
  }));
}

/** Bỏ các khoảng rỗng ở đầu (độ tin cậy thấp không xuất hiện) để trục hoành không có vùng trống. */
export function trimLeadingEmpty(bins: readonly DisplayBin[]): DisplayBin[] {
  const first = bins.findIndex((bin) => bin.n > 0);
  return first === -1 ? [] : bins.slice(first);
}

/**
 * Ví dụ để đọc biểu đồ: khoảng nhiều câu nhất trong số các khoảng có điểm trên chưa chạm trần
 * (≤ 0,9), vì khoảng sát 1 gần như luôn đúng và không minh hoạ được việc hiệu chỉnh.
 */
export function calibrationExample(bins: readonly DisplayBin[]): DisplayBin | undefined {
  const candidates = bins.filter((bin) => bin.n > 0 && bin.upper <= 0.9 + 1e-9);
  if (candidates.length === 0) return undefined;
  return candidates.reduce((best, bin) => (bin.n > best.n ? bin : best));
}
