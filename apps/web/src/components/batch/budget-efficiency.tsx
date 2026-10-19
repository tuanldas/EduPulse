import type { ReactNode } from "react";

import { Icon } from "@/components/shared/icon";
import { Caption } from "@/components/shared/section-heading";
import { Card } from "@/components/ui/card";
import { acceptGainPoints, optionFor, sortedBudgets } from "@/lib/batch/budget";
import type { BudgetOption, ReviewBudget } from "@/lib/batch/types";
import { formatDecimal, formatPercent } from "@/lib/format";

function Stat({ label, value, children }: { label: string; value: string; children?: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <div className="text-sm">{label}</div>
      <div className="text-[2rem] leading-[1.25] font-semibold tabular-nums">{value}</div>
      {children}
    </div>
  );
}

interface BudgetEfficiencyProps {
  options: readonly BudgetOption[];
  selected: ReviewBudget;
}

/**
 * "Hiệu quả ước tính" của ngân sách đang chọn, tra từ bảng số đo trên tập validation (`/api/config`),
 * nên cập nhật ngay khi kéo thanh trượt mà không phải chạy mô hình.
 */
export function BudgetEfficiency({ options, selected }: BudgetEfficiencyProps) {
  const current = optionFor(options, selected);
  if (current === undefined) return null;
  const { efficiency } = current;
  const gain = acceptGainPoints(efficiency);
  const others = sortedBudgets(options).filter((option) => option.budget !== selected);

  return (
    <Card>
      <div className="flex items-center gap-1.5 text-xs font-semibold tracking-[0.05em] text-muted-foreground uppercase">
        <Icon name="target" size="xs" />
        Hiệu quả ước tính ở mức {formatPercent(selected / 100)} · số đo trên tập validation
      </div>
      <div className="grid grid-cols-3 gap-4 @max-[620px]:grid-cols-1">
        <Stat label="Lỗi cảm xúc được bắt" value={formatPercent(efficiency.caught_sentiment_errors)} />
        <Stat label="Lỗi chủ đề được bắt" value={formatPercent(efficiency.caught_topic_errors)} />
        <Stat label="Phần không cờ đúng cả 2 nhãn" value={formatPercent(efficiency.accept_accuracy_both, 1)}>
          {gain > 0 ? (
            <span className="inline-flex w-fit items-center gap-0.5 rounded-full bg-ok-bg py-px pr-2 pl-1 text-[13px] font-medium text-ok-fg">
              <Icon name="arrow_upward" size="xs" />
              {formatDecimal(gain, 1)} điểm
            </span>
          ) : null}
        </Stat>
      </div>
      <Caption>
        “Lỗi được bắt” = trong các câu mô hình đoán sai, bao nhiêu % nằm trong nhóm bị cờ. Không gắn cờ: đúng cả 2 nhãn{" "}
        {formatPercent(efficiency.all_accuracy_both, 1)}.
        {others.length > 0 ? (
          <>
            {" "}
            Các mức khác:{" "}
            {others
              .map(
                (option) =>
                  `${formatPercent(option.budget / 100)} → ${formatPercent(option.efficiency.caught_sentiment_errors)} · ${formatPercent(option.efficiency.caught_topic_errors)} · ${formatPercent(option.efficiency.accept_accuracy_both, 1)}`,
              )
              .join(" | ")}
            .
          </>
        ) : null}{" "}
        Đây là ước tính, kết quả trên file thật có thể khác.
      </Caption>
    </Card>
  );
}
