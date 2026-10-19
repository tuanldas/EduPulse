"use client";

import { useState } from "react";

import { Caption, SectionHeading } from "@/components/shared/section-heading";
import type { InsightsResponse, TopicLabel } from "@/lib/batch/types";
import type { QuoteFilter } from "@/lib/dashboard/insight-view";

import { AlertCards } from "./alert-cards";
import { DeltaHeatmap } from "./delta-heatmap";
import { KeywordChips, type SelectedKeyword } from "./keyword-chips";
import { KpiRow } from "./kpi-row";
import { Recommendations } from "./recommendations";
import { RepresentativeSentences } from "./representative-sentences";
import { SentimentByTopic } from "./sentiment-by-topic";

type Alert = InsightsResponse["alerts"][number];

const QUOTES_ANCHOR = "cau-tieu-bieu";

function scrollToQuotes() {
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
  document
    .getElementById(QUOTES_ANCHOR)
    ?.scrollIntoView?.({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
}

interface InsightsViewProps {
  data: InsightsResponse;
  /** Tên cột nhóm (vd. "Môn học") để ghi vào tiêu đề heatmap. */
  groupLabel: string;
}

/** Thân dashboard: KPI → cảnh báo → biểu đồ → từ khoá → câu tiêu biểu + khuyến nghị. Dựng lại khi đổi nguồn. */
export function InsightsView({ data, groupLabel }: InsightsViewProps) {
  const [tab, setTab] = useState<TopicLabel>("facility");
  const [keyword, setKeyword] = useState<SelectedKeyword | null>(null);
  const [group, setGroup] = useState<string | null>(null);

  const filter: QuoteFilter = { term: keyword?.term ?? null, group };

  function selectKeyword(next: SelectedKeyword | null) {
    setKeyword(next);
    if (next !== null) {
      setTab(next.topic);
      scrollToQuotes();
    }
  }

  function changeTab(next: TopicLabel) {
    setTab(next);
    // Từ khoá thuộc chủ đề khác sẽ không khớp câu nào ở tab mới, nên bỏ lọc từ khoá.
    if (keyword !== null && keyword.topic !== next) setKeyword(null);
  }

  function seeQuotes(alert: Alert) {
    setTab(alert.topic);
    setKeyword(null);
    setGroup(alert.group);
    scrollToQuotes();
  }

  function changeFilter(next: QuoteFilter) {
    if (next.term === null) setKeyword(null);
    setGroup(next.group);
  }

  return (
    <div className="flex flex-col gap-4">
      <KpiRow kpis={data.kpis} />

      <SectionHeading icon="notifications">Cảnh báo so với mức nền</SectionHeading>
      <AlertCards
        alerts={data.alerts}
        topics={data.topics}
        illustrativeGroups={data.illustrative_groups}
        onSeeQuotes={seeQuotes}
      />

      <div className="grid grid-cols-2 items-start gap-4 @max-[800px]:grid-cols-1">
        <SentimentByTopic topics={data.topics} className={data.heatmap === null ? "col-span-2 @max-[800px]:col-span-1" : undefined} />
        {data.heatmap !== null ? (
          <DeltaHeatmap heatmap={data.heatmap} groupLabel={groupLabel} illustrative={data.illustrative_groups} />
        ) : null}
      </div>

      {data.heatmap === null ? (
        <Caption>Heatmap nhóm × chủ đề chỉ hiện khi file có cột nhóm; thẻ cảnh báo ở trên so cả tập với mức nền.</Caption>
      ) : null}

      <SectionHeading icon="sell">Từ khoá đặc trưng trong câu tiêu cực</SectionHeading>
      <KeywordChips keywords={data.keywords} selected={keyword} onSelect={selectKeyword} />

      <div className="grid grid-cols-[minmax(0,3fr)_minmax(0,2fr)] items-start gap-x-6 gap-y-4 @max-[900px]:grid-cols-1">
        <section aria-labelledby={QUOTES_ANCHOR} className="flex min-w-0 flex-col gap-3">
          <SectionHeading id={QUOTES_ANCHOR} icon="format_quote" className="scroll-mt-4">
            Câu tiêu biểu
          </SectionHeading>
          <RepresentativeSentences
            quotes={data.quotes}
            tab={tab}
            onTabChange={changeTab}
            filter={filter}
            onFilterChange={changeFilter}
          />
        </section>
        <section aria-labelledby="khuyen-nghi" className="flex min-w-0 flex-col gap-3">
          <SectionHeading id="khuyen-nghi" icon="lightbulb">
            Khuyến nghị
          </SectionHeading>
          <Recommendations recommendations={data.recommendations} />
        </section>
      </div>
    </div>
  );
}

