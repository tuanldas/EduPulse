import explainJson from "../fixtures/batch-explain.json";
import insightsFilteredJson from "../fixtures/batch-insights-filtered.json";
import insightsJson from "../fixtures/batch-insights.json";
import resultsJson from "../fixtures/batch-results.json";
import statusJson from "../fixtures/batch-status-done.json";
import uploadJson from "../fixtures/batch-upload.json";

import type {
  BatchStatus,
  BatchUpload,
  ExplainResponse,
  InsightsResponse,
  ResultsResponse,
} from "@/lib/batch/types";

// Fixture ghi từ API thật bằng `scripts/record-batch-fixtures.mjs` trên 20 câu tự viết (không có câu
// UIT-VSFC). JSON chỉ suy ra kiểu `string` cho các giá trị nhãn nên mỗi fixture được ép về kiểu của API đúng
// một lần ở đây; hình dạng thật vẫn được kiểm bằng chính các test dùng chúng.

export const batchUpload = uploadJson as unknown as BatchUpload;
export const batchStatusDone = statusJson as unknown as BatchStatus;
export const batchResults = resultsJson as unknown as ResultsResponse;
export const batchInsights = insightsJson as unknown as InsightsResponse;
export const batchInsightsFiltered = insightsFilteredJson as unknown as InsightsResponse;

interface RecordedExplanation {
  index: number;
  response: ExplainResponse;
}

export const batchExplain = explainJson as unknown as {
  /** Giải thích có từ được tô (câu mô hình không chắc). */
  withHighlights: RecordedExplanation;
  /** Giải thích mà không từ nào đủ ảnh hưởng để tô (câu rất chắc chắn). */
  noHighlights: RecordedExplanation;
  /** Chỉ số của một dòng quá dài để giải thích. */
  tooLongIndex: number;
};
