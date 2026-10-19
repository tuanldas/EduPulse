import { LabelBadge } from "@/components/shared/label-badge";
import { Caption } from "@/components/shared/section-heading";
import { Card } from "@/components/ui/card";
import type { InsightsResponse, TopicLabel } from "@/lib/batch/types";
import { KEYWORD_TOPIC_ORDER } from "@/lib/dashboard/insight-view";
import { formatNumber } from "@/lib/format";
import { labelMeta } from "@/lib/labels";
import { cn } from "@/lib/utils";

export interface SelectedKeyword {
  topic: TopicLabel;
  term: string;
}

interface KeywordChipsProps {
  keywords: InsightsResponse["keywords"];
  selected: SelectedKeyword | null;
  /** Chọn lại từ đang chọn thì bỏ chọn (null). */
  onSelect: (keyword: SelectedKeyword | null) => void;
}

/**
 * Ba khối từ khoá đặc trưng trong câu tiêu cực (CSVC, Chương trình, Giảng viên; "Khác" quá phân tán nên
 * không có). Bấm một từ để lọc "Câu tiêu biểu"; từ đang chọn có `aria-pressed`.
 */
export function KeywordChips({ keywords, selected, onSelect }: KeywordChipsProps) {
  const byTopic = new Map(keywords.map((block) => [block.topic, block]));
  const blocks = KEYWORD_TOPIC_ORDER.map((topic) => byTopic.get(topic)).filter(
    (block): block is NonNullable<typeof block> => block !== undefined,
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-3 gap-4 max-[900px]:grid-cols-1">
        {blocks.map((block) => (
          <Card key={block.topic} aria-label={`Từ khoá chủ đề ${labelMeta("topic", block.topic).vi}`} role="group">
            <div className="flex items-center justify-between gap-2">
              <LabelBadge task="topic" label={block.topic} size="lg" />
              <span className="text-sm text-muted-foreground tabular-nums">{formatNumber(block.n_negative)} câu</span>
            </div>
            {block.keywords.length === 0 ? (
              <Caption>Chưa có từ khoá nổi bật.</Caption>
            ) : (
              <ul className="flex flex-wrap gap-2">
                {block.keywords.map((keyword) => {
                  const active = selected?.topic === block.topic && selected.term === keyword.term;
                  return (
                    <li key={keyword.term}>
                      <button
                        type="button"
                        aria-pressed={active}
                        onClick={() => onSelect(active ? null : { topic: block.topic, term: keyword.term })}
                        className={cn(
                          "inline-flex min-h-[34px] items-center gap-1.5 rounded-full border px-3 py-[5px] text-sm leading-[1.35] outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
                          active
                            ? "border-primary bg-primary-soft font-medium text-primary-text"
                            : "border-border bg-background hover:bg-accent",
                        )}
                      >
                        {keyword.term}
                        <span className={cn("text-xs tabular-nums", active ? "text-inherit" : "text-muted-foreground")}>
                          <span aria-hidden="true">{formatNumber(keyword.n)}</span>
                          {/* Tên của nút phải chứa chữ nhìn thấy (WCAG 2.5.3) và nói rõ con số là số câu. */}
                          <span className="sr-only"> {formatNumber(keyword.n)} câu</span>
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </Card>
        ))}
      </div>
      <Caption>
        Từ khoá chọn bằng log-odds có trọng số (so với phần còn lại), không cần danh sách từ dừng. Số = số câu tiêu cực
        chứa từ. Chủ đề Khác không hiện từ khoá vì quá phân tán. Bấm một từ để lọc câu tiêu biểu bên dưới.
      </Caption>
    </div>
  );
}
