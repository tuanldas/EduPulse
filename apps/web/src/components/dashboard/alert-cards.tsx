import { WrappingIllustrativeBadge } from "@/components/batch/wrapping-illustrative-badge";
import { Callout } from "@/components/shared/callout";
import { Icon } from "@/components/shared/icon";
import { LabelBadge } from "@/components/shared/label-badge";
import { Caption } from "@/components/shared/section-heading";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { InsightsResponse } from "@/lib/batch/types";
import { TONE_CLASS } from "@/lib/colors";
import {
  ALERT_MARGIN_POINTS,
  ALERT_MIN_N,
  calmTopics,
} from "@/lib/dashboard/insight-view";
import { emphasizeSummary } from "@/lib/dashboard/text-highlight";
import { formatDecimal, formatNumber, formatPercent } from "@/lib/format";
import { labelMeta } from "@/lib/labels";
import { cn } from "@/lib/utils";

type Alert = InsightsResponse["alerts"][number];
type Topics = InsightsResponse["topics"];

/** Khoảng tin cậy dưới 10% ghi một chữ số thập phân ("ít nhất 7,4%"), còn lại làm tròn ("ít nhất 89%"). */
function lowerBound(value: number): string {
  return formatPercent(value, value < 0.1 ? 1 : 0);
}

function AlertCard({
  alert,
  illustrativeGroups,
  onSeeQuotes,
}: {
  alert: Alert;
  illustrativeGroups: boolean;
  onSeeQuotes: (alert: Alert) => void;
}) {
  const topic = labelMeta("topic", alert.topic);
  const share = alert.kind === "topic_share";
  const accent = share ? TONE_CLASS[topic.tone] : TONE_CLASS.neg;
  const keywords = alert.keywords.slice(0, 3);
  return (
    <Card className="relative h-full gap-2.5 overflow-hidden pl-5">
      <span aria-hidden="true" className={cn("absolute inset-y-0 left-0 w-1", accent.solid)} />
      <h3 className="flex items-start gap-2 text-base leading-[1.4] font-semibold">
        <Icon name={share ? "campaign" : "trending_up"} className={cn("mt-px", accent.text)} />
        {alert.title}
      </h3>
      <p className="text-[15px] leading-[1.55]">
        {emphasizeSummary(alert.summary).map((segment, index) =>
          segment.mark ? (
            <b key={index} className="font-semibold">
              {segment.text}
            </b>
          ) : (
            <span key={index}>{segment.text}</span>
          ),
        )}
      </p>
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[13px] text-muted-foreground">
        <LabelBadge task="topic" label={alert.topic} />
        <span>Khoảng tin cậy 95%: ít nhất {lowerBound(alert.evidence.ci_low)}</span>
        <span className="tabular-nums">n = {formatNumber(alert.evidence.n)}</span>
        {alert.group !== null ? <span>Nhóm: {alert.group}</span> : null}
        {alert.group !== null && illustrativeGroups ? <WrappingIllustrativeBadge /> : null}
        {keywords.length > 0 ? <span>Nổi bật: {keywords.join(", ")}</span> : null}
        {alert.note ? <span>{alert.note}</span> : null}
      </div>
      <Button variant="ghost" size="sm" className="-ml-3 self-start" onClick={() => onSeeQuotes(alert)}>
        Xem câu liên quan
        <Icon name="arrow_forward" size="sm" />
      </Button>
    </Card>
  );
}

interface AlertCardsProps {
  alerts: InsightsResponse["alerts"];
  topics: Topics;
  /** Nhóm là dữ liệu minh hoạ (file mẫu): cảnh báo theo nhóm phải ghi rõ điều đó. */
  illustrativeGroups: boolean;
  onSeeQuotes: (alert: Alert) => void;
}

/** Dòng "Giảng viên 34,5% (nền 35,6) · Chương trình 72,7% (78,0) · …" cho thông báo không có cảnh báo. */
function BaselineList({ topics }: { topics: Topics }) {
  return (
    <>
      {topics.map((stat, index) => (
        <span key={stat.topic}>
          {index > 0 ? " · " : ""}
          {labelMeta("topic", stat.topic).vi} {formatPercent(stat.sentiment_shares.negative, 1)} (
          {index === 0 ? "nền " : ""}
          {formatDecimal(stat.baseline_negative_rate * 100, 1)})
        </span>
      ))}
    </>
  );
}

/** Tối đa ba thẻ cảnh báo so với mức nền, hoặc thông báo "không chủ đề nào vượt mức nền đáng kể". */
export function AlertCards({ alerts, topics, illustrativeGroups, onSeeQuotes }: AlertCardsProps) {
  const calm = calmTopics(topics, alerts);
  return (
    <div className="flex flex-col gap-3">
      {alerts.length === 0 ? (
        <Callout tone="ok" title="Không chủ đề nào vượt mức nền đáng kể." role="status">
          Tiêu cực: <BaselineList topics={topics} />.
        </Callout>
      ) : (
        <div className="grid grid-cols-3 gap-4 @max-[800px]:grid-cols-2 @max-[620px]:grid-cols-1">
          {alerts.map((alert, index) => (
            <AlertCard
              key={`${alert.kind}-${alert.topic}-${alert.group ?? ""}-${index}`}
              alert={alert}
              illustrativeGroups={illustrativeGroups}
              onSeeQuotes={onSeeQuotes}
            />
          ))}
        </div>
      )}
      <Caption>
        Mức nền = tỉ lệ tiêu cực trong dữ liệu huấn luyện UIT-VSFC:{" "}
        {topics
          .map((stat) => `${labelMeta("topic", stat.topic).vi} ${formatPercent(stat.baseline_negative_rate, 1)}`)
          .join(" · ")}
        . Chỉ cảnh báo khi có ít nhất {formatNumber(ALERT_MIN_N)} câu và vẫn cao hơn mức nền trên{" "}
        {formatNumber(ALERT_MARGIN_POINTS)} điểm ở khoảng tin cậy 95%.
        {alerts.length > 0
          ? calm.map((item) => (
              <span key={item.topic}>
                {" "}
                <b className="font-semibold text-foreground">
                  {labelMeta("topic", item.topic).vi}
                  {alerts.some((alert) => alert.topic === item.topic && alert.group !== null) ? " (toàn tập)" : ""}:
                </b>{" "}
                {item.reason === "enough"
                  ? `trong mức bình thường (${formatPercent(item.negativeRate)}).`
                  : `chưa đủ câu để kết luận (n = ${formatNumber(item.n)}).`}
              </span>
            ))
          : null}
      </Caption>
    </div>
  );
}
