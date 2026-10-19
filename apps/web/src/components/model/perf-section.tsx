"use client";

import { useState } from "react";

import { Callout } from "@/components/shared/callout";
import { MetricCard } from "@/components/shared/metric-card";
import { SegmentedControl } from "@/components/shared/segmented-control";
import { Caption, SectionHeading } from "@/components/shared/section-heading";
import type { ModelCard } from "@/lib/api/types";
import { formatDecimal, formatNumber, formatPercent, formatScore } from "@/lib/format";
import { labelMeta, TASK_META, type Task } from "@/lib/labels";

import { ComparisonTable } from "./comparison-table";
import { ConfusionMatrix } from "./confusion-matrix";
import { caughtErrors, defaultBudgetRow, misclassifiedInto, weakestClass } from "./derive";
import { MetricsGuide } from "./metrics-guide";
import { PerClassTable } from "./per-class-table";
import { ReliabilityChart } from "./reliability-chart";

/** Câu mô tả lớp yếu nhất, ghép hoàn toàn từ số trong model card. */
function WeakestClassNote({ task, metrics, splitName }: { task: Task; metrics: ModelCard["metrics"]["sentiment"]; splitName: string }) {
  const weakest = weakestClass(metrics);
  if (!weakest) return null;
  const name = labelMeta(task, weakest.label).vi;
  return (
    <Caption>
      <b className="font-semibold text-foreground">{name}</b> yếu nhất: F1 {formatDecimal(weakest.f1)}, recall{" "}
      {formatDecimal(weakest.recall)}, chỉ chiếm {formatPercent(weakest.share, 1)} số câu {splitName}.
    </Caption>
  );
}

function ConfusionNote({ task, metrics }: { task: Task; metrics: ModelCard["metrics"]["sentiment"] }) {
  const weakest = weakestClass(metrics);
  if (!weakest) return null;
  const { into } = misclassifiedInto(metrics, weakest.label);
  if (into.length === 0) return null;
  const name = labelMeta(task, weakest.label).vi;
  const parts = into.map((cell) => `${formatPercent(cell.share)} thành ${labelMeta(task, cell.label).vi}`);
  return (
    <Caption>
      Câu thuộc lớp “{name}” chỉ {formatPercent(weakest.recall)} được đoán đúng; {parts.join(", ")}.
    </Caption>
  );
}

/**
 * Mục "Hiệu năng": chọn Cảm xúc/Chủ đề, 4 chỉ số chính, bảng từng lớp, ma trận nhầm lẫn, so sánh với mô
 * hình cơ sở, độ tin cậy. Tên tập (test/validation) và nhãn bản smoke lấy từ `stage` và `metrics.split`.
 */
export function PerfSection({ card }: { card: ModelCard }) {
  const [task, setTask] = useState<Task>("sentiment");
  const metrics = card.metrics[task];
  const isTest = card.metrics.split === "test";
  const splitName = isTest ? "test" : "validation";
  const budget = defaultBudgetRow(card);
  const options = (["sentiment", "topic"] as const).map((value) => ({
    value,
    icon: TASK_META[value].icon,
    label: TASK_META[value].vi,
  }));

  return (
    <section aria-labelledby="hieu-nang" className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SectionHeading id="hieu-nang" icon="speed" className="mt-4">
          Hiệu năng trên tập {splitName}
        </SectionHeading>
        <SegmentedControl value={task} onValueChange={setTask} options={options} ariaLabel="Chọn tác vụ" />
      </div>

      {card.stage === "smoke" ? (
        <Callout tone="info" icon="model_training" title="Bản smoke · số trên tập validation." role="note">
          Đây là bản chạy thử để kiểm tra toàn tuyến; số liệu tính trên tập validation, chưa phải kết quả cuối cùng trên tập
          test.
        </Callout>
      ) : null}

      <Caption>
        UIT-VSFC {splitName}, {formatNumber(card.metrics.n)} câu
        {isTest ? ", chỉ đánh giá một lần sau khi chốt mô hình." : "."}
      </Caption>

      <div className="grid grid-cols-4 gap-4 max-[1100px]:grid-cols-2 max-[640px]:grid-cols-1">
        <MetricCard label="Accuracy" value={formatPercent(metrics.accuracy, 1)} caption="tỉ lệ câu đoán đúng" />
        <MetricCard label="Macro-F1" value={formatScore(metrics.macro_f1)} caption="trung bình đều các lớp" />
        <MetricCard label="ECE (sau hiệu chỉnh)" value={formatDecimal(metrics.ece, 3)} caption="càng gần 0 càng tốt" />
        {budget ? (
          <MetricCard
            label={`Lỗi được bắt ở ngân sách duyệt ${formatNumber(budget.budget)}%`}
            value={formatPercent(caughtErrors(budget, task))}
            caption={`phần không bị cờ đúng cả 2 nhãn ${formatPercent(budget.efficiency.accept_accuracy_both, 1)}`}
          />
        ) : (
          <MetricCard label="Lỗi được bắt ở ngân sách duyệt" value="—" caption="model card chưa có số ngân sách duyệt" />
        )}
      </div>

      <div className="grid grid-cols-2 gap-6 max-[900px]:grid-cols-1">
        <div className="flex min-w-0 flex-col gap-3">
          <div className="text-[15px] font-semibold">Chỉ số từng lớp</div>
          <PerClassTable task={task} metrics={metrics} splitName={splitName} />
          <WeakestClassNote task={task} metrics={metrics} splitName={splitName} />
        </div>
        <div className="flex min-w-0 flex-col gap-3">
          <div className="text-[15px] font-semibold">
            Confusion matrix <span className="text-sm font-normal text-muted-foreground">(% theo hàng nhãn thật)</span>
          </div>
          <ConfusionMatrix task={task} metrics={metrics} />
          <ConfusionNote task={task} metrics={metrics} />
        </div>
      </div>

      {card.baselines.length > 0 || card.architectures.length > 0 ? (
        <div className="flex min-w-0 flex-col gap-3">
          <div className="text-[15px] font-semibold">So sánh với mô hình cơ sở</div>
          <ComparisonTable baselines={card.baselines} architectures={card.architectures} />
          <Caption>
            Đa nhiệm: 1 mô hình cho cả 2 việc (cảm xúc và chủ đề), nên dùng ít bộ nhớ và thời gian hơn hai mô hình đơn nhiệm.
            Macro-F1 ở các dòng nhiều seed là trung bình ± độ lệch chuẩn.
          </Caption>
        </div>
      ) : null}

      <div className="grid grid-cols-2 items-start gap-6 max-[900px]:grid-cols-1">
        <div className="flex min-w-0 flex-col gap-3">
          <div className="text-[15px] font-semibold">Độ tin cậy có khớp thực tế không?</div>
          <ReliabilityChart metrics={metrics} />
        </div>
        <MetricsGuide metrics={metrics} card={card} />
      </div>
    </section>
  );
}
