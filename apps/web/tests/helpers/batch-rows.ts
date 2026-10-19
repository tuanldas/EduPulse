import type { FlagReason, ResultRow, ResultsResponse, ResultsSummary } from "@/lib/batch/types";

// Câu tự viết cho test, không lấy từ UIT-VSFC.

export function lowConfidence(task: "sentiment" | "topic", message: string): FlagReason {
  return { code: "low_confidence", task, candidates: [], message };
}

export function makeRow(overrides: Partial<ResultRow> = {}): ResultRow {
  return {
    index: 0,
    source_row: 1,
    text: "Cô giảng bài rất dễ hiểu và nhiệt tình.",
    n_words: 9,
    no_diacritics: false,
    pii_types: [],
    sentiment: "positive",
    sentiment_confidence: 0.97,
    topic: "lecturer",
    topic_confidence: 0.96,
    needs_review: false,
    reasons: [],
    source: "none",
    in_review: false,
    group: null,
    has_explanation: false,
    explainable: true,
    ...overrides,
  };
}

export function makeSummary(overrides: Partial<ResultsSummary> = {}): ResultsSummary {
  return {
    n_rows: 4,
    n_analyzed: 4,
    n_skipped: 0,
    n_flagged: 2,
    flagged_share: 0.5,
    n_audit: 1,
    n_review: 3,
    budget: 10,
    budget_target: 1,
    n_no_diacritics: 0,
    pii_rows: 0,
    pii_counts: [],
    n_truncated: 0,
    n_explained: 2,
    n_groups: null,
    elapsed_s: 1.5,
    illustrative_groups: false,
    ...overrides,
  };
}

/** Bốn dòng: 0 và 2 bị cờ, 3 là mẫu kiểm tra, 1 không nằm trong danh sách duyệt. */
export function makeResults(): ResultsResponse {
  const rows: ResultRow[] = [
    makeRow({
      index: 0,
      source_row: 11,
      text: "Phòng máy hơi cũ nhưng thầy dạy nhiệt tình.",
      sentiment: "neutral",
      sentiment_confidence: 0.44,
      topic: "facility",
      topic_confidence: 0.52,
      needs_review: true,
      reasons: [lowConfidence("sentiment", "Cảm xúc không chắc")],
      source: "flag",
      in_review: true,
    }),
    makeRow({ index: 1, source_row: 12, text: "Thầy dạy dễ hiểu." }),
    makeRow({
      index: 2,
      source_row: 13,
      text: "Môn này nhiều lý thuyết quá.",
      sentiment: "negative",
      sentiment_confidence: 0.51,
      topic: "training_program",
      needs_review: true,
      reasons: [lowConfidence("sentiment", "Cảm xúc không chắc")],
      source: "flag",
      in_review: true,
    }),
    makeRow({
      index: 3,
      source_row: 14,
      text: "Slide đầy đủ.",
      source: "audit",
      in_review: true,
    }),
  ];
  return {
    summary: makeSummary(),
    rows,
    review_order: [0, 2, 3],
    skipped: [],
  };
}
