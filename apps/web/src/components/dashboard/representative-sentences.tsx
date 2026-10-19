"use client";

import { useState } from "react";

import { Icon } from "@/components/shared/icon";
import { LabelBadge } from "@/components/shared/label-badge";
import { PiiText } from "@/components/shared/pii-chip";
import { Caption } from "@/components/shared/section-heading";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { InsightsResponse, TopicLabel } from "@/lib/batch/types";
import { TONE_CLASS } from "@/lib/colors";
import { filterQuotes, type QuoteFilter } from "@/lib/dashboard/insight-view";
import { splitByTerms } from "@/lib/dashboard/text-highlight";
import { formatNumber } from "@/lib/format";
import { labelMeta } from "@/lib/labels";
import { cn } from "@/lib/utils";

type Quote = InsightsResponse["quotes"][number]["quotes"][number];

/** Thứ tự tab: ba chủ đề có từ khoá trước, "Khác" sau cùng (khớp khối từ khoá bên trên). */
const TAB_ORDER = ["facility", "training_program", "lecturer", "others"] as const satisfies readonly TopicLabel[];

const PAGE_SIZE = 5;

function Highlighted({ text, terms }: { text: string; terms: readonly string[] }) {
  return (
    <>
      {splitByTerms(text, terms).map((segment, index) =>
        segment.mark ? (
          <mark key={index} className="rounded-[3px] bg-xai-1 px-0.5 font-semibold text-inherit">
            {segment.text}
          </mark>
        ) : (
          <PiiText key={index} text={segment.text} />
        ),
      )}
    </>
  );
}

function QuoteItem({ quote }: { quote: Quote }) {
  return (
    <li className="flex gap-2.5 border-b border-divider py-2.5 text-[15px] leading-[1.55] last:border-b-0">
      <Icon name="format_quote" className="mt-0.5 text-muted-foreground" />
      <div className="flex min-w-0 flex-col gap-1">
        <span>
          <Highlighted text={quote.text} terms={quote.matched} />
        </span>
        <span className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-sm text-muted-foreground">
          <LabelBadge task="sentiment" label={quote.sentiment} confidence={quote.confidence} />
          {quote.group !== null ? <span>· {quote.group}</span> : null}
        </span>
      </div>
    </li>
  );
}

function FilterChip({ label, onClear, clearLabel }: { label: string; onClear: () => void; clearLabel: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md bg-primary-soft py-px pr-1 pl-2 text-sm">
      {label}
      <button
        type="button"
        aria-label={clearLabel}
        onClick={onClear}
        className="rounded-sm text-muted-foreground outline-none hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50"
      >
        <Icon name="close" size="xs" />
      </button>
    </span>
  );
}

interface TopicQuotesProps {
  topic: TopicLabel;
  quotes: readonly Quote[];
  filter: QuoteFilter;
  onFilterChange: (filter: QuoteFilter) => void;
}

function TopicQuotes({ topic, quotes, filter, onFilterChange }: TopicQuotesProps) {
  const [limit, setLimit] = useState(PAGE_SIZE);
  const matching = filterQuotes(quotes, filter);
  const visible = matching.slice(0, limit);
  const filtering = filter.term !== null || filter.group !== null;

  return (
    <div className="flex flex-col gap-3">
      {filtering ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-muted-foreground">Đang lọc:</span>
          {filter.term !== null ? (
            <FilterChip
              label={filter.term}
              clearLabel={`Bỏ lọc từ khoá ${filter.term}`}
              onClear={() => onFilterChange({ ...filter, term: null })}
            />
          ) : null}
          {filter.group !== null ? (
            <FilterChip
              label={`Nhóm: ${filter.group}`}
              clearLabel={`Bỏ lọc nhóm ${filter.group}`}
              onClear={() => onFilterChange({ ...filter, group: null })}
            />
          ) : null}
        </div>
      ) : null}

      {matching.length === 0 ? (
        <p className="text-sm text-muted-foreground" role="status">
          {filtering
            ? `Không có câu nào về ${labelMeta("topic", topic).vi} khớp bộ lọc này.`
            : `Chưa có câu tiêu biểu nào về ${labelMeta("topic", topic).vi}.`}
        </p>
      ) : (
        <ul>
          {visible.map((quote, index) => (
            <QuoteItem key={`${index}-${quote.text}`} quote={quote} />
          ))}
        </ul>
      )}

      {matching.length > visible.length ? (
        <Button variant="outline" size="sm" className="self-start" onClick={() => setLimit((value) => value + PAGE_SIZE)}>
          Xem thêm ({formatNumber(matching.length - visible.length)} câu)
        </Button>
      ) : null}
    </div>
  );
}

interface RepresentativeSentencesProps {
  quotes: InsightsResponse["quotes"];
  tab: TopicLabel;
  onTabChange: (tab: TopicLabel) => void;
  filter: QuoteFilter;
  onFilterChange: (filter: QuoteFilter) => void;
}

/**
 * "Câu tiêu biểu" theo chủ đề: từ khoá in đậm, nhãn cảm xúc kèm độ tin cậy và nhóm. Câu đã ẩn thông tin cá
 * nhân từ máy chủ; lọc theo từ khoá/nhóm chạy ngay trên trình duyệt (`matched`, `group` của từng câu).
 */
export function RepresentativeSentences({ quotes, tab, onTabChange, filter, onFilterChange }: RepresentativeSentencesProps) {
  const byTopic = new Map(quotes.map((block) => [block.topic, block.quotes]));
  return (
    <div className="flex flex-col gap-3">
      <Tabs value={tab} onValueChange={(value) => onTabChange(value as TopicLabel)}>
        <TabsList aria-label="Chọn chủ đề của câu tiêu biểu">
          {TAB_ORDER.map((topic) => {
            const meta = labelMeta("topic", topic);
            return (
              <TabsTrigger key={topic} value={topic}>
                <Icon name={meta.icon} size="xs" className={cn(TONE_CLASS[meta.tone].text)} />
                {meta.vi}
              </TabsTrigger>
            );
          })}
        </TabsList>
        {TAB_ORDER.map((topic) => (
          <TabsContent key={topic} value={topic}>
            <TopicQuotes
              key={`${filter.term ?? ""}|${filter.group ?? ""}`}
              topic={topic}
              quotes={byTopic.get(topic) ?? []}
              filter={filter}
              onFilterChange={onFilterChange}
            />
          </TabsContent>
        ))}
      </Tabs>
      <Caption>
        Ưu tiên câu gọn (6–30 từ), chứa nhiều từ khoá, độ tin cậy ≥ 70%, bỏ câu gần trùng. Tên riêng đã ẩn ở mức cố gắng
        tối đa.
      </Caption>
    </div>
  );
}

