import { Icon, type IconName } from "@/components/shared/icon";
import { MetricCard } from "@/components/shared/metric-card";
import { TONE_CLASS } from "@/lib/colors";
import { kpiDelta, type KpiDelta, type KpiMetric } from "@/lib/dashboard/insight-view";
import type { InsightsResponse } from "@/lib/batch/types";
import { formatDecimal, formatNumber, formatPercent } from "@/lib/format";
import { SENTIMENT_META } from "@/lib/labels";
import { cn } from "@/lib/utils";

const DELTA_ICON: Record<KpiDelta["direction"], IconName> = {
  up: "arrow_upward",
  down: "arrow_downward",
  none: "remove",
};

/** Chip "↑ 7,5 điểm so với nền": đỏ chỉ khi lệch theo hướng xấu; mũi tên và chữ luôn có, không chỉ màu. */
function DeltaChip({ delta }: { delta: KpiDelta }) {
  const text =
    delta.relation === "ngang mức nền" ? delta.relation : `${formatDecimal(delta.points, 1)} điểm ${delta.relation}`;
  return (
    <span
      className={cn(
        "inline-flex w-fit items-center gap-0.5 rounded-full py-px pr-2 pl-1 text-[13px] font-medium",
        delta.tone === "bad" ? TONE_CLASS.neg.badge : TONE_CLASS.neu.badge,
      )}
    >
      <Icon name={DELTA_ICON[delta.direction]} size="xs" />
      {text}
    </span>
  );
}

function SentimentTitle({ sentiment }: { sentiment: KpiMetric | "neutral" }) {
  const meta = SENTIMENT_META[sentiment];
  return (
    <span className="inline-flex items-center gap-1.5">
      <Icon name={meta.icon} size="xs" className={TONE_CLASS[meta.tone].text} />
      {meta.vi}
    </span>
  );
}

interface KpiRowProps {
  kpis: InsightsResponse["kpis"];
}

/**
 * Bốn chỉ số đầu trang. Dữ liệu demo không có bước gắn cờ nên ô thứ tư là "Trung tính"; file vừa tải
 * thì ô thứ tư là "Cần duyệt". Tỉ lệ tính trên NHÃN DỰ ĐOÁN (không phản ánh nhãn đã sửa).
 */
export function KpiRow({ kpis }: KpiRowProps) {
  const flagged = kpis.n_flagged !== null && kpis.flagged_share !== null;
  return (
    <div className="grid grid-cols-4 gap-4 @max-[800px]:grid-cols-2 @max-[620px]:grid-cols-1">
      <MetricCard
        label="Số câu"
        value={formatNumber(kpis.n_rows)}
        caption={kpis.n_groups !== null ? `${formatNumber(kpis.n_groups)} nhóm` : undefined}
      />
      <MetricCard label={<SentimentTitle sentiment="negative" />} value={formatPercent(kpis.shares.negative, 1)}>
        <dd>
          <DeltaChip delta={kpiDelta("negative", kpis.delta_points.negative)} />
        </dd>
      </MetricCard>
      {flagged ? (
        <>
          <MetricCard label={<SentimentTitle sentiment="positive" />} value={formatPercent(kpis.shares.positive, 1)}>
            <dd>
              <DeltaChip delta={kpiDelta("positive", kpis.delta_points.positive)} />
            </dd>
          </MetricCard>
          <MetricCard
            label="Cần duyệt"
            icon="flag"
            value={formatPercent(kpis.flagged_share ?? 0, 1)}
            caption={`${formatNumber(kpis.n_flagged ?? 0)} câu bị gắn cờ`}
          />
        </>
      ) : (
        <>
          <MetricCard label={<SentimentTitle sentiment="neutral" />} value={formatPercent(kpis.shares.neutral, 1)} />
          <MetricCard label={<SentimentTitle sentiment="positive" />} value={formatPercent(kpis.shares.positive, 1)} />
        </>
      )}
    </div>
  );
}
