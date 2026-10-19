import { Icon } from "@/components/shared/icon";
import { Caption } from "@/components/shared/section-heading";
import { Card } from "@/components/ui/card";
import type { InsightsResponse, SentimentLabel } from "@/lib/batch/types";
import { TONE_CLASS } from "@/lib/colors";
import { topicBars, type TopicBar } from "@/lib/dashboard/insight-view";
import { formatNumber, formatPercent } from "@/lib/format";
import { labelMeta, SENTIMENT_META } from "@/lib/labels";
import { cn } from "@/lib/utils";

// Chữ trong đoạn: trắng trên Tiêu cực, tối trên Trung tính/Tích cực để đủ tương phản ở cả hai theme
// (màu nền của đoạn không đổi theo theme nên màu chữ cũng không đổi).
const TEXT_ON_FILL: Record<SentimentLabel, string> = {
  negative: "text-white",
  neutral: "text-black",
  positive: "text-black",
};

/** Câu mô tả đầy đủ của một hàng cho trình đọc màn hình, vì thanh chỉ là hình. */
function describe(bar: TopicBar): string {
  const name = labelMeta("topic", bar.topic).vi;
  if (bar.empty) return `${name}: chưa có câu nào.`;
  const shares = bar.segments
    .map((segment) => `${formatPercent(segment.share)} ${SENTIMENT_META[segment.sentiment].vi.toLowerCase()}`)
    .join(", ");
  return `${name}: ${shares}. Mức nền tiêu cực ${formatPercent(bar.baseline, 1)}. ${formatNumber(bar.n)} câu.`;
}

function Row({ bar }: { bar: TopicBar }) {
  const meta = labelMeta("topic", bar.topic);
  return (
    <li className="contents">
      <span className="flex items-center gap-1.5 leading-tight whitespace-nowrap max-[640px]:whitespace-normal">
        <Icon name={meta.icon} size="xs" className={TONE_CLASS[meta.tone].text} />
        {meta.vi}
      </span>
      <div className="relative">
        <span className="sr-only">{describe(bar)}</span>
        {bar.empty ? (
          <div aria-hidden="true" className="flex h-7 items-center rounded-[4px] bg-grid px-2 text-xs text-muted-foreground">
            Chưa có câu nào
          </div>
        ) : (
          <div aria-hidden="true" className="flex h-7">
            {bar.segments.map((segment) =>
              segment.share > 0 ? (
                <i
                  key={segment.sentiment}
                  className={cn(
                    "@container flex items-center justify-center overflow-hidden text-xs font-semibold whitespace-nowrap not-italic first:rounded-l-[4px] last:rounded-r-[4px]",
                    TONE_CLASS[SENTIMENT_META[segment.sentiment].tone].solid,
                    TEXT_ON_FILL[segment.sentiment],
                  )}
                  style={{ width: `${segment.share * 100}%` }}
                >
                  {/* Số chỉ hiện khi đoạn đủ rộng (đo bằng container query), tránh bị cắt nửa chừng. */}
                  <span className="hidden @min-[34px]:inline">{formatPercent(segment.share)}</span>
                </i>
              ) : null,
            )}
          </div>
        )}
        <span
          aria-hidden="true"
          className="absolute -top-1 -bottom-1 w-[3px] -translate-x-1/2 rounded-[2px] bg-foreground shadow-[0_0_0_1.5px_var(--background)]"
          style={{ left: `${bar.baseline * 100}%` }}
        />
      </div>
      <span className="text-right text-muted-foreground tabular-nums">{formatNumber(bar.n)}</span>
    </li>
  );
}

interface SentimentByTopicProps {
  topics: InsightsResponse["topics"];
  className?: string;
}

/**
 * Cột chồng 100% "Cảm xúc theo chủ đề" kèm vạch mức nền tiêu cực của TỪNG chủ đề. Dựng bằng CSS flex
 * thay vì Recharts vì vạch nền khác nhau ở mỗi hàng; số nằm trong ô và có mô tả bằng chữ.
 */
export function SentimentByTopic({ topics, className }: SentimentByTopicProps) {
  const bars = topicBars(topics);
  return (
    <Card className={className}>
      <div>
        <h3 className="text-[15px] font-semibold">Cảm xúc theo chủ đề</h3>
        <Caption>% số câu trong từng chủ đề · vạch đen = mức nền tiêu cực</Caption>
      </div>
      <ul className="grid grid-cols-[132px_minmax(0,1fr)_52px] items-center gap-x-3 gap-y-3 text-sm max-[640px]:grid-cols-[116px_minmax(0,1fr)_36px]">
        {bars.map((bar) => (
          <Row key={bar.topic} bar={bar} />
        ))}
      </ul>
      <div
        aria-hidden="true"
        className="mr-[64px] ml-[144px] flex justify-between text-xs text-muted-foreground tabular-nums max-[640px]:mr-[48px] max-[640px]:ml-[128px]"
      >
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[13px] text-muted-foreground">
        {(["negative", "neutral", "positive"] as const).map((sentiment) => (
          <span key={sentiment} className="inline-flex items-center gap-1.5">
            <i
              aria-hidden="true"
              className={cn("size-3 rounded-[3px]", TONE_CLASS[SENTIMENT_META[sentiment].tone].solid)}
            />
            {SENTIMENT_META[sentiment].vi}
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5">
          <i aria-hidden="true" className="h-3.5 w-[3px] rounded-[1px] bg-foreground" />
          Mức nền tiêu cực
        </span>
        <span>· cột phải = số câu</span>
      </div>
    </Card>
  );
}
