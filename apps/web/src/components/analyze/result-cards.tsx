import Link from "next/link";

import { ConfidenceBar } from "@/components/shared/confidence-bar";
import { FlagBadge } from "@/components/shared/flag-badge";
import { Icon } from "@/components/shared/icon";
import { ProbBars } from "@/components/shared/prob-bars";
import { Caption, SectionHeading } from "@/components/shared/section-heading";
import { Card } from "@/components/ui/card";
import { TONE_CLASS } from "@/lib/colors";
import { formatPercent } from "@/lib/format";
import { labelMeta, TASK_META, type Task } from "@/lib/labels";
import type { AnalyzeResponse } from "@/lib/api/types";
import { cn } from "@/lib/utils";

import { belowThreshold, reasonParts } from "./flag-reason";

interface ResultHeaderProps {
  analysis: AnalyzeResponse;
  /** Ngân sách duyệt mặc định (%), từ `/api/config`; chưa biết thì ẩn dòng chú thích ngưỡng. */
  defaultBudget: number | undefined;
}

/** "Kết quả" + cờ "Cần xem xét" kèm lý do bằng lời, hoặc "Đủ tin cậy". */
export function ResultHeader({ analysis, defaultBudget }: ResultHeaderProps) {
  const { flag } = analysis;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <SectionHeading id="ket-qua">Kết quả</SectionHeading>
        {flag.needs_review ? (
          <>
            <FlagBadge size="lg" />
            <span className="text-sm">
              Lý do:{" "}
              {flag.reasons.map((reason, index) => {
                const parts = reasonParts(reason);
                return (
                  <span key={`${reason.code}-${index}`}>
                    {index > 0 ? "; " : ""}
                    <b className="font-semibold">{parts.title}</b> {parts.detail}
                  </span>
                );
              })}
            </span>
          </>
        ) : (
          <span className="inline-flex items-center gap-1.5 text-sm text-muted-foreground">
            <Icon name="check_circle" size="xs" />
            Đủ tin cậy, không cần xem xét
          </span>
        )}
      </div>
      {defaultBudget ? (
        <Caption>
          <Icon name="tune" size="xs" /> Ngưỡng cờ theo ngân sách duyệt{" "}
          <b className="font-semibold text-foreground">{formatPercent(defaultBudget / 100)}</b> (mặc định) ·{" "}
          <Link href="/batch/">đổi ngân sách ở trang Phân tích hàng loạt</Link>
        </Caption>
      ) : null}
    </div>
  );
}

interface TaskCardProps {
  task: Task;
  label: string;
  confidence: number;
  probs: Readonly<Record<string, number>>;
  flagged: boolean;
}

/** Thẻ kết quả của một tác vụ: nhãn lớn + %, độ tin cậy đã hiệu chỉnh, xác suất từng lớp. */
function TaskCard({ task, label, confidence, probs, flagged }: TaskCardProps) {
  const meta = labelMeta(task, label);
  const tone = TONE_CLASS[meta.tone];
  const taskMeta = TASK_META[task];
  return (
    <Card aria-label={taskMeta.vi} role="group">
      <div className="flex items-center gap-1.5 text-xs font-semibold tracking-[0.05em] text-muted-foreground uppercase">
        <Icon name={taskMeta.icon} size="xs" />
        {taskMeta.vi}
      </div>
      <div className="flex items-center justify-between gap-3">
        <div className={cn("flex items-center gap-2 text-[28px] leading-[1.2] font-bold", tone.text)}>
          <Icon name={meta.icon} className="size-7" />
          {meta.vi}
        </div>
        <div className="text-[28px] leading-[1.2] font-bold tabular-nums">{formatPercent(confidence)}</div>
      </div>
      <ConfidenceBar value={confidence} flagged={flagged} />
      <ProbBars task={task} probs={probs} />
    </Card>
  );
}

/** Hai thẻ Cảm xúc | Chủ đề và dòng giải thích ý nghĩa độ tin cậy đã hiệu chỉnh. */
export function ResultCards({ analysis }: { analysis: AnalyzeResponse }) {
  const { sentiment, topic, flag } = analysis;
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4 max-[640px]:grid-cols-1">
        <TaskCard
          task="sentiment"
          label={sentiment.label}
          confidence={sentiment.confidence}
          probs={sentiment.probs}
          flagged={belowThreshold(flag.reasons, "sentiment")}
        />
        <TaskCard
          task="topic"
          label={topic.label}
          confidence={topic.confidence}
          probs={topic.probs}
          flagged={belowThreshold(flag.reasons, "topic")}
        />
      </div>
      <Caption>
        <Icon name="info" size="xs" /> Độ tin cậy đã hiệu chỉnh: trong những câu được báo khoảng 90%, chừng 9/10 câu được
        đoán đúng. Câu dưới ngưỡng được gắn cờ để người duyệt.
      </Caption>
    </div>
  );
}
