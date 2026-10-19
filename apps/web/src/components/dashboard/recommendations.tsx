import { Callout } from "@/components/shared/callout";
import { Icon, type IconName } from "@/components/shared/icon";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import type { InsightsResponse, TopicLabel } from "@/lib/batch/types";
import { TOPIC_SHORT } from "@/lib/dashboard/insight-view";
import { labelMeta } from "@/lib/labels";

type Recommendation = InsightsResponse["recommendations"][number];

// Icon riêng cho từng luật (khoá `id` ổn định của máy chủ); luật mới chưa có ở đây dùng icon của chủ đề.
const ICON_BY_RULE: Readonly<Record<string, IconName>> = {
  facility_projector: "videocam",
  program_theory_practice: "balance",
  lecturer_pace: "record_voice_over",
  lecturer_absence: "event_busy",
  facility_ventilation: "air",
};

function iconFor(recommendation: Recommendation): IconName {
  return ICON_BY_RULE[recommendation.id] ?? labelMeta("topic", recommendation.topic).icon;
}

function TopicTag({ topic }: { topic: TopicLabel }) {
  const meta = labelMeta("topic", topic);
  return (
    <Badge variant={meta.tone} title={meta.vi}>
      {TOPIC_SHORT[topic]}
    </Badge>
  );
}

interface RecommendationsProps {
  recommendations: InsightsResponse["recommendations"];
}

/** Khuyến nghị sinh từ luật từ khoá: hành động + căn cứ + chủ đề, kèm cảnh báo "chỉ để gợi ý". */
export function Recommendations({ recommendations }: RecommendationsProps) {
  return (
    <div className="flex flex-col gap-3">
      {recommendations.length === 0 ? (
        <Callout tone="info" icon="lightbulb" role="status">
          Chưa có khuyến nghị nào: không có từ khoá nào đủ nổi bật để gợi ý hành động.
        </Callout>
      ) : (
        <Card className="gap-0 px-4 py-1">
          <ul className="divide-y divide-divider">
            {recommendations.map((recommendation) => (
              <li
                key={recommendation.id}
                className="grid grid-cols-[auto_minmax(0,1fr)_auto] gap-x-3 gap-y-1 py-3.5 max-[640px]:grid-cols-[auto_minmax(0,1fr)]"
              >
                <Icon name={iconFor(recommendation)} className="mt-0.5 text-primary" />
                <div className="flex min-w-0 flex-col gap-1">
                  <p className="text-[15px] leading-[1.45] font-semibold">{recommendation.action}</p>
                  <p className="text-sm leading-[1.5] text-muted-foreground">{recommendation.evidence}</p>
                  {recommendation.note ? (
                    <p className="text-sm leading-[1.5] text-muted-foreground">{recommendation.note}</p>
                  ) : null}
                </div>
                <div className="flex items-start justify-end max-[640px]:col-start-2 max-[640px]:justify-start">
                  <TopicTag topic={recommendation.topic} />
                </div>
              </li>
            ))}
          </ul>
        </Card>
      )}
      <Callout tone="warn" icon="gavel" role="note">
        Khuyến nghị sinh tự động từ luật từ khoá, <b className="font-semibold">chỉ để gợi ý</b>. Cán bộ cần đọc câu gốc
        trước khi quyết định. Không dùng để đánh giá cá nhân giảng viên.
      </Callout>
    </div>
  );
}
