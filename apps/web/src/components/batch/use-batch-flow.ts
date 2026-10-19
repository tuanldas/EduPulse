"use client";

import { useEffect, useRef, useState } from "react";

import { toFailure } from "@/lib/api/client";
import { isAbortError, type ApiFailure } from "@/lib/api/errors";
import { useAppConfig, type AppConfigState } from "@/lib/app-config";
import {
  createSampleBatch,
  deleteBatch,
  getBatchResults,
  getBatchStatus,
  runBatch,
  sendClose,
  sendHeartbeat,
  uploadBatch,
} from "@/lib/batch/batch-api";
import { checkUploadFile } from "@/lib/batch/file-checks";
import { discardReviewEdits, useReviewEdits, type ReviewEditsApi } from "@/lib/batch/review-edits";
import {
  clearStoredBatch,
  saveStoredBatch,
  storedBatchFromUpload,
  storedBatchStore,
  useStoredBatch,
  type StoredBatch,
  type StoredRun,
} from "@/lib/batch/stored-batch";
import type { BatchStatus, BatchUpload, ResultsResponse } from "@/lib/batch/types";
import { useAsyncData, type UseAsyncData } from "@/lib/batch/use-async-data";
import { useBatchPolling } from "@/lib/batch/use-batch-polling";
import { useBatchSession } from "@/lib/batch/use-batch-session";

import type { RunPhase } from "./run-progress";
import type { UploadActivity } from "./upload-step";

/** Chu kỳ heartbeat dự phòng khi chưa nạp được `/api/config` (khớp `HEARTBEAT_S` của API). */
const FALLBACK_HEARTBEAT_S = 60;

const CONFIRM_DISCARD_EDITS =
  "Bạn còn nhãn đã sửa chưa tải xuống. Tiếp tục sẽ làm mất chúng. Tiếp tục?";

export type FlowPhase =
  /** Chưa có file. */
  | "empty"
  /** Có file đang giữ trên máy chủ nhưng chưa biết trạng thái (vừa tải lại trang). */
  | "checking"
  /** File đã tải lên, chờ chọn cột và ngân sách. */
  | "setup"
  | "running"
  | "failed"
  | "done"
  /** Máy chủ không còn batch (hết hạn, đã đóng). */
  | "ended"
  /** Mất kết nối liên tục khi hỏi tiến độ. */
  | "unreachable";

export interface BatchFlow {
  configState: AppConfigState;
  stored: StoredBatch | null;
  phase: FlowPhase;
  status: BatchStatus | null;
  /** Trạng thái cho hộp tiến trình; null khi chưa chạy lần nào. */
  runPhase: RunPhase | null;
  preview: readonly (readonly string[])[] | null;
  activity: UploadActivity;
  uploadFailure: ApiFailure | null;
  runFailure: ApiFailure | null;
  pollFailure: ApiFailure | null;
  starting: boolean;
  results: UseAsyncData<ResultsResponse>;
  edits: ReviewEditsApi;
  actions: {
    uploadFile: (file: File) => void;
    loadSample: () => void;
    removeFile: () => void;
    run: (run: StoredRun) => void;
    retryRun: () => void;
    reset: () => void;
    retryPolling: () => void;
  };
}

function runPhaseOf(status: BatchStatus | null, run: StoredRun | null): RunPhase | null {
  if (status === null) return run === null ? null : "checking";
  switch (status.state) {
    case "queued":
      return "queued";
    case "running":
      return "running";
    case "done":
      return "done";
    case "failed":
      return "failed";
    case "uploaded":
      return null;
  }
}

/**
 * Toàn bộ vòng đời của trang Phân tích hàng loạt: tải file → chạy nền → theo dõi → kết quả → xuất,
 * cộng việc giữ phiên (heartbeat, `close` khi đóng tab). Trạng thái hiển thị được SUY RA từ batch đã
 * lưu trong tab và câu trả lời của máy chủ, nên tải lại trang hay quay về từ dashboard đều dựng lại đúng.
 */
export function useBatchFlow(): BatchFlow {
  const configState = useAppConfig();
  const config = configState.status === "ready" ? configState.config : undefined;

  const stored = useStoredBatch();
  const batchId = stored?.batchId ?? null;
  const edits = useReviewEdits(batchId);
  const polling = useBatchPolling({ batchId, fetchStatus: getBatchStatus });

  const [activity, setActivity] = useState<UploadActivity>("idle");
  const [uploadFailure, setUploadFailure] = useState<ApiFailure | null>(null);
  const [runFailure, setRunFailure] = useState<ApiFailure | null>(null);
  const [starting, setStarting] = useState(false);
  const [preview, setPreview] = useState<{ batchId: string; rows: readonly (readonly string[])[] } | null>(null);
  const [endedFor, setEndedFor] = useState<string | null>(null);
  const inflight = useRef<AbortController | null>(null);

  useEffect(() => () => inflight.current?.abort(), []);

  // --- Suy ra giai đoạn ----------------------------------------------------------------------
  const pollState = polling.state;
  const status = pollState.phase === "polling" || pollState.phase === "settled" ? pollState.status : null;
  const ended = stored !== null && (endedFor === stored.batchId || pollState.phase === "gone");

  let phase: FlowPhase;
  if (stored === null) phase = "empty";
  else if (ended) phase = "ended";
  else if (pollState.phase === "error") phase = "unreachable";
  else if (status === null) phase = stored.run === null ? "setup" : "checking";
  else if (status.state === "uploaded") phase = "setup";
  else if (status.state === "done") phase = "done";
  else if (status.state === "failed") phase = "failed";
  else phase = "running";

  const results = useAsyncData(phase === "done" ? batchId : null, (signal) =>
    getBatchResults(batchId ?? "", signal),
  );
  const resultsGone = results.status === "error" && toFailure(results.failure).status === 404;
  if (resultsGone && phase === "done") phase = "ended";

  const alive = phase !== "empty" && phase !== "ended";
  useBatchSession({
    batchId: alive ? batchId : null,
    heartbeatS: config?.heartbeat_s ?? FALLBACK_HEARTBEAT_S,
    sendHeartbeat,
    sendClose,
    onGone: () => setEndedFor(batchId),
    confirmLeave: edits.unexported,
  });

  // --- Thao tác ------------------------------------------------------------------------------
  function confirmDiscard(): boolean {
    return !edits.unexported || window.confirm(CONFIRM_DISCARD_EDITS);
  }

  /** Giải phóng batch cũ trên máy chủ và xoá nhãn sửa của nó; lỗi không quan trọng (TTL sẽ dọn). */
  function release(previous: StoredBatch | null) {
    if (previous === null) return;
    void deleteBatch(previous.batchId).catch(() => undefined);
    discardReviewEdits(previous.batchId);
  }

  function adopt(upload: BatchUpload, sizeBytes: number | null) {
    const previous = storedBatchStore.getSnapshot();
    saveStoredBatch(storedBatchFromUpload(upload, sizeBytes));
    setPreview({ batchId: upload.batch_id, rows: upload.preview });
    setEndedFor(null);
    setRunFailure(null);
    release(previous);
  }

  async function intake(kind: UploadActivity, request: (signal: AbortSignal) => Promise<BatchUpload>, size: number | null) {
    const controller = new AbortController();
    inflight.current = controller;
    setUploadFailure(null);
    setActivity(kind);
    try {
      adopt(await request(controller.signal), size);
    } catch (error) {
      if (!isAbortError(error)) setUploadFailure(toFailure(error));
    } finally {
      setActivity("idle");
    }
  }

  function uploadFile(file: File) {
    const invalid = checkUploadFile(file, config?.limits.max_upload_mb);
    if (invalid !== null) {
      setUploadFailure(invalid);
      return;
    }
    if (!confirmDiscard()) return;
    void intake("uploading", (signal) => uploadBatch(file, signal), file.size);
  }

  function loadSample() {
    if (!confirmDiscard()) return;
    void intake("sampling", (signal) => createSampleBatch(signal), null);
  }

  function removeFile() {
    if (!confirmDiscard()) return;
    release(stored);
    clearStoredBatch();
    setPreview(null);
    setUploadFailure(null);
    setRunFailure(null);
    setEndedFor(null);
  }

  async function run(selection: StoredRun) {
    if (stored === null) return;
    setStarting(true);
    setRunFailure(null);
    try {
      const initial = await runBatch(stored.batchId, {
        text_col: selection.textCol,
        group_col: selection.groupCol,
        review_budget: selection.budget,
      });
      // Đọc lại bản đang lưu: lúc chờ phản hồi người dùng có thể đã đổi file.
      const current = storedBatchStore.getSnapshot();
      if (current !== null && current.batchId === stored.batchId) saveStoredBatch({ ...current, run: selection });
      polling.restart(initial);
    } catch (error) {
      if (isAbortError(error)) return;
      const failure = toFailure(error);
      if (failure.status === 404) setEndedFor(stored.batchId);
      else if (failure.code === "already_running") polling.restart();
      setRunFailure(failure);
    } finally {
      setStarting(false);
    }
  }

  function retryRun() {
    if (stored?.run) void run(stored.run);
  }

  function reset() {
    release(stored);
    clearStoredBatch();
    setPreview(null);
    setUploadFailure(null);
    setRunFailure(null);
    setEndedFor(null);
  }

  return {
    configState,
    stored,
    phase,
    status,
    runPhase: runPhaseOf(status, stored?.run ?? null),
    preview: preview !== null && preview.batchId === batchId ? preview.rows : null,
    activity,
    uploadFailure,
    runFailure,
    pollFailure: pollState.phase === "error" ? toFailure(pollState.failure) : null,
    starting,
    results,
    edits,
    actions: {
      uploadFile,
      loadSample,
      removeFile,
      run: (selection) => void run(selection),
      retryRun,
      reset,
      retryPolling: () => polling.restart(),
    },
  };
}

