import { useSyncExternalStore } from "react";

import { createSessionStore } from "./session-store";
import type { BatchUpload, BatchWarning, ReviewBudget } from "./types";

/** Lựa chọn khi chạy phân tích; giữ lại để dựng lại trang khi quay về từ dashboard. */
export interface StoredRun {
  textCol: string;
  groupCol: string | null;
  budget: ReviewBudget;
}

/**
 * Batch gần nhất của tab này, để chuyển trang (Phân tích hàng loạt ↔ Dashboard) mà không mất phiên.
 * Cố ý KHÔNG lưu `preview`: đó là nguyên văn các dòng đầu của file, chưa ẩn thông tin cá nhân.
 */
export interface StoredBatch {
  batchId: string;
  filename: string;
  nRows: number;
  columns: string[];
  suggestedTextCol: string | null;
  suggestedGroupCol: string | null;
  illustrativeGroups: boolean;
  warnings: BatchWarning[];
  run: StoredRun | null;
}

const KEY = "edupulse.batch";
const BUDGETS = [5, 10, 20, 30] as const satisfies readonly ReviewBudget[];
const WARNING_CODES = [
  "encoding_fallback",
  "blank_rows_skipped",
  "other_sheets_ignored",
  "columns_renamed",
] as const satisfies readonly BatchWarning["code"][];

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isWarningCode(value: unknown): value is BatchWarning["code"] {
  return (WARNING_CODES as readonly unknown[]).includes(value);
}

function isBudget(value: unknown): value is ReviewBudget {
  return (BUDGETS as readonly unknown[]).includes(value);
}

function parseWarnings(raw: unknown): BatchWarning[] {
  if (!Array.isArray(raw)) return [];
  const warnings: BatchWarning[] = [];
  for (const item of raw) {
    if (typeof item !== "object" || item === null) continue;
    const { code, message, count } = item as Record<string, unknown>;
    if (!isWarningCode(code) || !isString(message)) continue;
    warnings.push({ code, message, count: typeof count === "number" ? count : null });
  }
  return warnings;
}

function parseRun(raw: unknown): StoredRun | null {
  if (typeof raw !== "object" || raw === null) return null;
  const { textCol, groupCol, budget } = raw as Record<string, unknown>;
  if (!isString(textCol) || !isBudget(budget)) return null;
  const group = groupCol ?? null;
  if (group !== null && !isString(group)) return null;
  return { textCol, groupCol: group, budget };
}

export function parseStoredBatch(raw: unknown): StoredBatch | null {
  if (typeof raw !== "object" || raw === null) return null;
  const value = raw as Record<string, unknown>;
  if (!isString(value.batchId) || value.batchId === "") return null;
  if (!isString(value.filename)) return null;
  if (typeof value.nRows !== "number" || !Number.isFinite(value.nRows)) return null;
  if (!Array.isArray(value.columns) || !value.columns.every(isString)) return null;
  const suggestedText = value.suggestedTextCol ?? null;
  const suggestedGroup = value.suggestedGroupCol ?? null;
  if (suggestedText !== null && !isString(suggestedText)) return null;
  if (suggestedGroup !== null && !isString(suggestedGroup)) return null;
  return {
    batchId: value.batchId,
    filename: value.filename,
    nRows: value.nRows,
    columns: value.columns,
    suggestedTextCol: suggestedText,
    suggestedGroupCol: suggestedGroup,
    illustrativeGroups: value.illustrativeGroups === true,
    warnings: parseWarnings(value.warnings),
    run: parseRun(value.run),
  };
}

export const storedBatchStore = createSessionStore<StoredBatch | null>(KEY, null, parseStoredBatch);

export function storedBatchFromUpload(upload: BatchUpload): StoredBatch {
  return {
    batchId: upload.batch_id,
    filename: upload.filename,
    nRows: upload.n_rows,
    columns: upload.columns,
    suggestedTextCol: upload.suggested_text_col,
    suggestedGroupCol: upload.suggested_group_col,
    illustrativeGroups: upload.illustrative_groups,
    warnings: upload.warnings,
    run: null,
  };
}

export function useStoredBatch(): StoredBatch | null {
  return useSyncExternalStore(
    storedBatchStore.subscribe,
    storedBatchStore.getSnapshot,
    storedBatchStore.getServerSnapshot,
  );
}
