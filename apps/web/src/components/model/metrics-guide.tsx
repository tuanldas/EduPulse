import { SubHeading } from "@/components/shared/section-heading";
import { Card } from "@/components/ui/card";
import type { ModelCard } from "@/lib/api/types";
import { formatDecimal, formatNumber, formatPercent } from "@/lib/format";

import { defaultBudgetRow } from "./derive";

type TaskMetrics = ModelCard["metrics"]["sentiment"];

interface MetricsGuideProps {
  /** Chỉ số của tác vụ đang xem, dùng cho câu ví dụ về ECE. */
  metrics: TaskMetrics;
  card: Pick<ModelCard, "review_budget">;
}

/** Hộp "Đọc các chỉ số thế nào?": định nghĩa ngắn kèm ví dụ bằng số thật của model card. */
export function MetricsGuide({ metrics, card }: MetricsGuideProps) {
  const budget = defaultBudgetRow(card);
  return (
    <Card>
      <SubHeading icon="school" className="text-[17px]">
        Đọc các chỉ số thế nào?
      </SubHeading>
      <dl className="grid grid-cols-[130px_minmax(0,1fr)] gap-y-3 text-sm leading-[1.55] max-[640px]:grid-cols-1 max-[640px]:gap-y-0.5">
        <dt className="font-semibold">Accuracy</dt>
        <dd className="max-[640px]:mb-3">
          Tỉ lệ câu đoán đúng. Dễ hiểu nhưng bị lớp đông (tiêu cực, tích cực) chi phối.
        </dd>
        <dt className="font-semibold">Macro-F1</dt>
        <dd className="max-[640px]:mb-3">
          Trung bình đều điểm F1 của từng lớp, nên lớp ít câu như Trung tính, Khác được tính ngang lớp đông. Chỉ số chính
          để so mô hình.
        </dd>
        <dt className="font-semibold">ECE</dt>
        <dd className="max-[640px]:mb-3">
          Độ lệch trung bình giữa “độ tin cậy mô hình báo” và “tỉ lệ đúng thực tế”. {formatDecimal(metrics.ece, 3)} nghĩa
          là lệch khoảng {formatNumber(metrics.ece * 100)} điểm %.
        </dd>
        <dt className="font-semibold">Độ tin cậy</dt>
        <dd className="max-[640px]:mb-3">
          Xác suất đã hiệu chỉnh mô hình gán cho <b className="font-semibold">một câu</b>. Khác với “khoảng tin cậy 95%” ở
          thẻ cảnh báo, vốn nói về tỉ lệ của <b className="font-semibold">cả nhóm câu</b>.
        </dd>
        <dt className="font-semibold">Ngân sách duyệt</dt>
        <dd>
          Tỉ lệ câu có độ tin cậy thấp nhất được gắn cờ cho người duyệt.
          {budget ? (
            <>
              {" "}
              Ở {formatNumber(budget.budget)}% (≈ {formatNumber(budget.budget * 10)}/1.000 câu), nhóm bị cờ chứa khoảng{" "}
              {formatPercent(budget.efficiency.caught_sentiment_errors)} lỗi cảm xúc và{" "}
              {formatPercent(budget.efficiency.caught_topic_errors)} lỗi chủ đề; phần không cờ đúng cả 2 nhãn{" "}
              {formatPercent(budget.efficiency.accept_accuracy_both, 1)} (không gắn cờ:{" "}
              {formatPercent(budget.efficiency.all_accuracy_both, 1)}).
            </>
          ) : null}
        </dd>
      </dl>
    </Card>
  );
}
