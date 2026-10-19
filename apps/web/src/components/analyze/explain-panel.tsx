"use client";

import { useState } from "react";

import { Callout } from "@/components/shared/callout";
import { FailureNotice } from "@/components/shared/failure-notice";
import { SegmentedControl } from "@/components/shared/segmented-control";
import { SubHeading } from "@/components/shared/section-heading";
import { XaiLegend, XaiSentence } from "@/components/shared/xai-sentence";
import { XaiTable } from "@/components/shared/xai-table";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { AnalyzeResponse, HeadExplanation } from "@/lib/api/types";
import { formatNumber, formatPoints } from "@/lib/format";
import { labelMeta, TASK_META, type Task } from "@/lib/labels";
import { explainRows } from "@/lib/xai";

import { LongSentenceNotice } from "./state-notices";
import type { ExplainState } from "./use-analysis";

interface ExplainPanelProps {
  analysis: AnalyzeResponse;
  explain: ExplainState;
  /** `limits.xai_max_words` của `/api/config`, để ghi vào thông báo câu dài. */
  maxWords: number | undefined;
  onRetry: () => void;
}

function ExplainSkeleton() {
  return (
    <div className="flex flex-col gap-4" aria-busy="true" aria-label="Đang tính giải thích">
      <Skeleton className="h-12 w-full" />
      <Skeleton className="h-4 w-3/4" />
      <div className="grid grid-cols-[minmax(0,3fr)_minmax(0,2fr)] gap-4 max-[640px]:grid-cols-1">
        <Skeleton className="h-36" />
        <Skeleton className="h-24" />
      </div>
    </div>
  );
}

/** Câu "Cách đọc" dưới bảng: nêu từ ảnh hưởng nhất và mức giảm/tăng xác suất khi bỏ từ đó. */
function HowToRead({ head, labelVi }: { head: HeadExplanation; labelVi: string }) {
  const [top] = explainRows(head);
  const direction = top && top.changePoints < 0 ? "giảm" : "tăng";
  const supportCount = head.support.length;
  const opposeCount = head.oppose.length;
  return (
    <p className="text-sm leading-[1.55] text-muted-foreground">
      Cách đọc: bỏ lần lượt từng từ khỏi câu rồi xem xác suất “{labelVi}” thay đổi bao nhiêu điểm %.{" "}
      {top ? (
        <>
          Bỏ “{top.text}” làm xác suất {direction} {formatPoints(Math.abs(top.changePoints)).replace("+", "")} điểm, đây là
          từ ảnh hưởng nhiều nhất.{" "}
        </>
      ) : null}
      Đây là giải thích gần đúng: chỉ tô {formatNumber(supportCount)} từ ủng hộ mạnh nhất
      {opposeCount > 0 ? ` và ${formatNumber(opposeCount)} từ phản đối` : ""}.
    </p>
  );
}

function ExplainBody({ head, task, originalText }: { head: HeadExplanation; task: Task; originalText: string }) {
  const labelVi = labelMeta(task, head.label).vi;
  const hasHighlights = head.support.length + head.oppose.length > 0;
  return (
    <div className="flex flex-col gap-4">
      <XaiSentence head={head} originalText={originalText} />
      {hasHighlights ? (
        <>
          <XaiLegend label={labelVi} />
          <div className="grid grid-cols-[minmax(0,3fr)_minmax(0,2fr)] items-start gap-4 max-[640px]:grid-cols-1">
            <XaiTable head={head} labelVi={labelVi} />
            <HowToRead head={head} labelVi={labelVi} />
          </div>
        </>
      ) : (
        <Callout tone="info" icon="manage_search" role="status">
          Không có từ nào ảnh hưởng đủ lớn đến nhãn “{labelVi}” để tô nổi: bỏ từng từ riêng lẻ không làm xác suất đổi
          đáng kể.
        </Callout>
      )}
    </div>
  );
}

/** Khối "Vì sao mô hình đoán như vậy?": chọn Cảm xúc/Chủ đề, câu tô từ, chú giải, bảng bỏ-từng-từ. */
export function ExplainPanel({ analysis, explain, maxWords, onRetry }: ExplainPanelProps) {
  const [task, setTask] = useState<Task>("sentiment");
  const options = (["sentiment", "topic"] as const).map((value) => ({
    value,
    icon: TASK_META[value].icon,
    label: `${TASK_META[value].vi} · ${labelMeta(value, analysis[value].label).vi}`,
  }));

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SubHeading icon="manage_search">Vì sao mô hình đoán như vậy?</SubHeading>
        {explain.status === "ready" ? (
          <SegmentedControl value={task} onValueChange={setTask} options={options} ariaLabel="Chọn tác vụ để giải thích" />
        ) : null}
      </div>
      <div>
        {explain.status === "loading" ? <ExplainSkeleton /> : null}
        {explain.status === "skipped" ? <LongSentenceNotice nWords={analysis.n_words} maxWords={maxWords} /> : null}
        {explain.status === "error" ? <FailureNotice failure={explain.failure} onRetry={onRetry} /> : null}
        {explain.status === "ready" ? (
          <ExplainBody head={explain.data[task]} task={task} originalText={analysis.display_text} />
        ) : null}
      </div>
    </Card>
  );
}
