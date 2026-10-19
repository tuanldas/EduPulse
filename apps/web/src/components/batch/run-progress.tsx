"use client";

import { useState, type ReactNode } from "react";

import { Callout } from "@/components/shared/callout";
import { Icon, type IconName } from "@/components/shared/icon";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Progress, ProgressLabel } from "@/components/ui/progress";
import { jobFailureMessage } from "@/lib/batch/batch-errors";
import { stepLineStates, STEP_LINES, type StepLine, type StepLineState } from "@/lib/batch/run-steps";
import type { BatchStatus } from "@/lib/batch/types";
import { formatEta, formatNumber, formatPercent } from "@/lib/format";
import { cn } from "@/lib/utils";

export type RunPhase = "checking" | "queued" | "running" | "done" | "failed";

function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "inline-block size-[18px] shrink-0 animate-spin rounded-full border-[2.5px] border-grid border-t-primary motion-reduce:animate-none",
        className,
      )}
    />
  );
}

function StepMark({ state }: { state: StepLineState }) {
  const base: { icon: IconName; className: string } =
    state === "done"
      ? { icon: "check_circle", className: "text-pos-fg" }
      : state === "failed"
        ? { icon: "error", className: "text-err-fg" }
        : { icon: "radio_button_unchecked", className: "text-muted-foreground" };
  if (state === "active") return <Spinner className="mx-px" />;
  return <Icon name={base.icon} size="sm" className={base.className} />;
}

interface RunProgressProps {
  phase: RunPhase;
  status: BatchStatus | null;
  /** Ngân sách duyệt đã chọn (%), để ghi vào dòng "Gắn cờ theo ngân sách …". */
  budget: number;
  /** Số câu cờ thấp nhất được giải thích sẵn (`limits.auto_xai_top`). */
  autoXaiTop: number | undefined;
  nRows: number;
  /** Thời gian chạy (giây), có khi đã xong. */
  elapsedS: number | null;
  onRetry: () => void;
}

function stepLabel(line: StepLine, status: BatchStatus | null, budget: number, autoXaiTop: number | undefined): ReactNode {
  switch (line) {
    case "pii":
      if (status?.pii_rows === null || status?.pii_rows === undefined) return "Ẩn thông tin cá nhân";
      return status.pii_rows === 0
        ? "Ẩn thông tin cá nhân · chưa phát hiện câu nào có thông tin cá nhân"
        : `Ẩn thông tin cá nhân · ${formatNumber(status.pii_rows)} câu có thông tin bị ẩn`;
    case "segment":
      return "Tách từ (RDRSegmenter)";
    case "predict":
      return "Dự đoán cảm xúc và chủ đề";
    case "flag":
      return `Gắn cờ theo ngân sách ${formatPercent(budget / 100)}${
        autoXaiTop ? ` · tự giải thích ${formatNumber(autoXaiTop)} câu cờ có độ tin cậy thấp nhất` : ""
      }`;
  }
}

function progressText(status: BatchStatus): string | null {
  if (status.step === "predict") {
    const eta = status.eta_s === null ? "đang ước tính thời gian" : `còn ${formatEta(status.eta_s)}`;
    return `${formatNumber(status.done)}/${formatNumber(status.total)} câu · ${eta}`;
  }
  if (status.step === "explain") {
    return `Đã giải thích ${formatNumber(status.done)}/${formatNumber(status.total)} câu`;
  }
  return null;
}

/** Hộp trạng thái khi chạy (st.status): danh sách bước + thanh tiến độ "640/1.000 câu · còn khoảng 25 giây". */
export function RunProgress({ phase, status, budget, autoXaiTop, nRows, elapsedS, onRetry }: RunProgressProps) {
  const [userOpen, setUserOpen] = useState<boolean | null>(null);
  const open = userOpen ?? phase !== "done";
  const states = stepLineStates(phase === "queued" || phase === "checking" ? null : status);
  const failed = phase === "failed";
  const text = status !== null && phase === "running" ? progressText(status) : null;

  const title =
    phase === "done"
      ? `Đã phân tích ${formatNumber(nRows)} câu${elapsedS === null ? "" : ` trong ${formatNumber(elapsedS, 1)} giây`}`
      : failed
        ? "Phân tích không thành công"
        : phase === "queued"
          ? "Đang chờ đến lượt…"
          : phase === "checking"
            ? "Đang kiểm tra trạng thái…"
            : `Đang phân tích ${formatNumber(nRows)} câu…`;

  return (
    <div className="flex flex-col gap-3">
      <Collapsible open={open} onOpenChange={setUserOpen} className="rounded-lg border border-border">
        <CollapsibleTrigger className="group flex min-h-11 w-full items-center gap-2.5 rounded-lg px-3.5 text-left text-[15px] outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
          {phase === "done" ? (
            <Icon name="check_circle" size="sm" className="text-pos-fg" />
          ) : failed ? (
            <Icon name="error" size="sm" className="text-err-fg" />
          ) : (
            <Spinner />
          )}
          <b role="status" className="font-semibold">
            {title}
          </b>
          <Icon
            name="keyboard_arrow_down"
            className="ml-auto text-muted-foreground transition-transform group-data-[panel-open]:rotate-180"
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="border-t border-divider px-4 pt-2 pb-4">
          <ol className="flex flex-col gap-2">
            {STEP_LINES.map((line) => (
              <li key={line} className={cn("flex flex-col gap-1.5", states[line] === "pending" && "text-muted-foreground")}>
                <div className="flex items-center gap-2 text-[15px]">
                  <StepMark state={states[line]} />
                  <span>{stepLabel(line, status, budget, autoXaiTop)}</span>
                </div>
                {states[line] === "active" && text !== null ? (
                  <Progress value={Math.round((status?.progress ?? 0) * 100)} className="pl-[26px]">
                    <ProgressLabel className="tabular-nums">{text}</ProgressLabel>
                  </Progress>
                ) : null}
              </li>
            ))}
          </ol>
        </CollapsibleContent>
      </Collapsible>

      {failed ? (
        <Callout
          tone="err"
          role="alert"
          action={
            <Button variant="outline" size="sm" onClick={onRetry}>
              <Icon name="refresh" size="sm" />
              Thử lại
            </Button>
          }
        >
          {jobFailureMessage(status?.error_code ?? null)}
        </Callout>
      ) : null}
    </div>
  );
}
