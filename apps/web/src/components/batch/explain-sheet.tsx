"use client";

import { useState } from "react";

import { Callout } from "@/components/shared/callout";
import { FailureNotice } from "@/components/shared/failure-notice";
import { FlagBadge } from "@/components/shared/flag-badge";
import { Icon } from "@/components/shared/icon";
import { PiiText } from "@/components/shared/pii-chip";
import { SegmentedControl } from "@/components/shared/segmented-control";
import { XaiLegend, XaiSentence } from "@/components/shared/xai-sentence";
import { XaiTable } from "@/components/shared/xai-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Sheet, SheetClose, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { toFailure } from "@/lib/api/client";
import type { HeadExplanation } from "@/lib/api/types";
import { explainRow } from "@/lib/batch/batch-api";
import { batchFailureMessage } from "@/lib/batch/batch-errors";
import type { ExplainResponse, FlagReason, ResultRow } from "@/lib/batch/types";
import { useAsyncData } from "@/lib/batch/use-async-data";
import { formatNumber, formatPercent } from "@/lib/format";
import { labelMeta, TASK_META, type Task } from "@/lib/labels";
import { explainRows } from "@/lib/xai";

type LoadExplanation = (batchId: string, index: number, signal: AbortSignal) => Promise<ExplainResponse>;

function ReasonLine({ reason }: { reason: FlagReason }) {
  if (reason.code === "no_diacritics") {
    return (
      <span>
        <b className="font-semibold">Câu không dấu</b> (luôn gắn cờ, kể cả khi độ tin cậy cao)
      </span>
    );
  }
  const task = reason.task;
  const title = task ? `${TASK_META[task].vi} không chắc` : reason.message;
  const candidates = task
    ? reason.candidates.map((candidate) => `${labelMeta(task, candidate.label).vi} ${formatPercent(candidate.probability)}`)
    : [];
  return (
    <span>
      <b className="font-semibold">{title}</b>
      {candidates.length > 0 ? ` — ${candidates.join(" · ")}` : ""}
    </span>
  );
}

function FlagSummary({ row }: { row: ResultRow }) {
  if (row.needs_review) {
    return (
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
        <FlagBadge />
        <span className="text-foreground">
          Lý do:{" "}
          {row.reasons.map((reason, index) => (
            <span key={`${reason.code}-${index}`}>
              {index > 0 ? "; " : ""}
              <ReasonLine reason={reason} />
            </span>
          ))}
        </span>
      </div>
    );
  }
  if (row.source === "audit") {
    return (
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Badge variant="gray">
          <Icon name="fact_check" size="xs" />
          Mẫu kiểm tra ngẫu nhiên
        </Badge>
        <span className="text-muted-foreground">Không bị gắn cờ; chọn ngẫu nhiên để đo độ chính xác.</span>
      </div>
    );
  }
  return null;
}

/** Câu "Bỏ “từ” làm xác suất “Nhãn” giảm từ 51% xuống 30%." cho từ ủng hộ mạnh nhất. */
function EffectSentence({ head, labelVi }: { head: HeadExplanation; labelVi: string }) {
  const [top] = explainRows(head);
  if (!top) {
    return (
      <Callout tone="info" icon="manage_search" role="status">
        Không có từ nào ảnh hưởng đủ lớn đến nhãn “{labelVi}” để tô nổi.
      </Callout>
    );
  }
  const verb = top.pWithout < top.pFull ? "giảm" : "tăng";
  return (
    <p className="text-sm leading-[1.55] text-muted-foreground">
      Bỏ “{top.text}” làm xác suất “{labelVi}” {verb} từ {formatPercent(top.pFull)} xuống {formatPercent(top.pWithout)}.
      Đây là giải thích gần đúng: chỉ tô các từ ảnh hưởng nhiều nhất.
    </p>
  );
}

function ExplainBody({ head, task, row }: { head: HeadExplanation; task: Task; row: ResultRow }) {
  const labelVi = labelMeta(task, head.label).vi;
  const hasHighlights = head.support.length + head.oppose.length > 0;
  return (
    <div className="flex flex-col gap-3">
      <XaiSentence head={head} originalText={row.text} className="text-[19px]" />
      {hasHighlights ? <XaiLegend label={labelVi} /> : null}
      <EffectSentence head={head} labelVi={labelVi} />
      {hasHighlights ? (
        <Collapsible>
          <CollapsibleTrigger className="group inline-flex items-center gap-1.5 rounded-md text-sm text-link outline-none hover:underline focus-visible:ring-3 focus-visible:ring-ring/50">
            Xem bảng xác suất khi bỏ từng từ
            <Icon
              name="keyboard_arrow_down"
              size="sm"
              className="transition-transform group-data-[panel-open]:rotate-180"
            />
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-3">
            <XaiTable head={head} labelVi={labelVi} />
          </CollapsibleContent>
        </Collapsible>
      ) : null}
    </div>
  );
}

function ExplainSkeleton() {
  return (
    <div className="flex flex-col gap-3" aria-busy="true" aria-label="Đang tính giải thích">
      <Skeleton className="h-16 w-full" />
      <Skeleton className="h-4 w-3/4" />
      <Skeleton className="h-4 w-1/2" />
    </div>
  );
}

export interface ExplainSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  batchId: string;
  /** Các dòng của tab đang xem, để Câu trước/Câu sau đi theo đúng thứ tự trên bảng. */
  rows: readonly ResultRow[];
  /** Vị trí của câu đang xem trong `rows`; null khi chưa chọn câu nào. */
  position: number | null;
  onPositionChange: (position: number) => void;
  /** `limits.xai_max_words`: câu dài hơn không tô từ. */
  xaiMaxWords: number;
  /** Cách lấy giải thích; mặc định gọi API (có tham số để test không cần mạng). */
  load?: LoadExplanation;
}

/**
 * Ngăn kéo bên phải giải thích một câu (`Sheet side="right"`): giữ bảng duyệt vẫn thấy được bên
 * cạnh. Câu đã có giải thích sẵn trả về ngay, câu khác được tính khi mở (khoảng 1 giây).
 */
export function ExplainSheet({
  open,
  onOpenChange,
  batchId,
  rows,
  position,
  onPositionChange,
  xaiMaxWords,
  load = explainRow,
}: ExplainSheetProps) {
  const [task, setTask] = useState<Task>("sentiment");
  const row = position === null ? null : (rows[position] ?? null);
  const index = row?.index ?? -1;
  const key = open && row !== null && row.explainable ? `${batchId}:${index}` : null;
  const explanation = useAsyncData(key, (signal) => load(batchId, index, signal));

  const options =
    row === null
      ? []
      : (["sentiment", "topic"] as const).map((value) => ({
          value,
          icon: TASK_META[value].icon,
          label: `${TASK_META[value].vi} · ${labelMeta(value, row[value]).vi}`,
        }));

  const failure = explanation.status === "error" ? toFailure(explanation.failure) : null;
  const total = rows.length;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-full gap-0 overflow-y-auto data-[side=right]:w-full data-[side=right]:sm:max-w-[560px]"
      >
        <SheetHeader className="gap-1 pr-14">
          <SheetTitle className="text-xl leading-7 font-semibold">
            Giải thích câu #{row === null ? "" : formatNumber(row.source_row)}
          </SheetTitle>
          <SheetDescription>
            {position === null ? "" : `Câu ${formatNumber(position + 1)} / ${formatNumber(total)} trong danh sách đang xem.`}
          </SheetDescription>
        </SheetHeader>

        {row !== null ? (
          <div className="flex flex-1 flex-col gap-4 px-4 pb-4">
            <FlagSummary row={row} />
            <SegmentedControl value={task} onValueChange={setTask} options={options} ariaLabel="Chọn tác vụ để giải thích" />
            <div aria-live="polite" className="flex flex-col gap-3">
              {!row.explainable ? (
                <>
                  <p className="text-[17px] leading-[1.8]">
                    <PiiText text={row.text} />
                  </p>
                  <Callout tone="info" icon="short_text" role="status">
                    <b className="font-semibold">
                      Câu dài {formatNumber(row.n_words)} từ (&gt; {formatNumber(xaiMaxWords)})
                    </b>{" "}
                    nên không tô từ để giữ tốc độ. Kết quả phân loại vẫn dùng được; câu dài không tự bị gắn cờ.
                  </Callout>
                </>
              ) : null}
              {row.explainable && explanation.status === "loading" ? <ExplainSkeleton /> : null}
              {failure !== null ? (
                failure.code === "no_content" || failure.code === "xai_too_long" ? (
                  <Callout tone="info" icon="short_text" role="status">
                    {batchFailureMessage(failure)}
                  </Callout>
                ) : (
                  <FailureNotice failure={failure} onRetry={explanation.reload} />
                )
              ) : null}
              {explanation.status === "ready" ? (
                <ExplainBody head={explanation.data[task]} task={task} row={row} />
              ) : null}
            </div>
          </div>
        ) : null}

        <div className="mt-auto flex flex-wrap items-center justify-end gap-2 border-t border-divider p-4">
          <Button
            variant="ghost"
            disabled={position === null || position <= 0}
            onClick={() => position !== null && onPositionChange(position - 1)}
          >
            <Icon name="arrow_back" size="sm" />
            Câu trước
          </Button>
          <Button
            variant="outline"
            disabled={position === null || position >= total - 1}
            onClick={() => position !== null && onPositionChange(position + 1)}
          >
            Câu sau
            <Icon name="arrow_forward" size="sm" />
          </Button>
          <SheetClose render={<Button />}>Đóng</SheetClose>
        </div>
      </SheetContent>
    </Sheet>
  );
}
