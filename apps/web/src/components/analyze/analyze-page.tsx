"use client";

import { useState } from "react";

import { FailureNotice } from "@/components/shared/failure-notice";
import { Icon } from "@/components/shared/icon";
import { Caption } from "@/components/shared/section-heading";
import { PageHeader } from "@/components/shell/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import { failureMessage } from "@/lib/api/errors";
import { useAppConfig } from "@/lib/app-config";
import { formatPercent } from "@/lib/format";
import { labelMeta } from "@/lib/labels";

import { AnalyzeForm } from "./analyze-form";
import { ComparePanel } from "./compare-panel";
import { ExplainPanel } from "./explain-panel";
import { ResultCards, ResultHeader } from "./result-cards";
import { EmptyState, NoContentNotice, NoDiacriticsNotice, PiiNotice } from "./state-notices";
import { useAnalysis, type AnalysisState } from "./use-analysis";

/** Câu ngắn cho trình đọc màn hình khi trạng thái đổi; nội dung chi tiết vẫn đọc được theo cấu trúc trang. */
function announcement(state: AnalysisState): string {
  switch (state.phase) {
    case "idle":
      return "";
    case "loading":
      return "Đang phân tích câu.";
    case "error":
      return failureMessage(state.failure);
    case "done": {
      const { sentiment, topic, flag } = state.analysis;
      const summary = `Kết quả: cảm xúc ${labelMeta("sentiment", sentiment.label).vi} ${formatPercent(sentiment.confidence)}, chủ đề ${labelMeta("topic", topic.label).vi} ${formatPercent(topic.confidence)}.`;
      const review = flag.needs_review ? " Câu này cần xem xét." : "";
      const explained = state.explain.status === "ready" ? " Đã có phần giải thích từng từ." : "";
      return `${summary}${review}${explained}`;
    }
  }
}

function ResultSkeleton() {
  return (
    <div className="flex flex-col gap-4" aria-busy="true" aria-label="Đang phân tích">
      <Skeleton className="h-8 w-64" />
      <div className="grid grid-cols-2 gap-4 max-[640px]:grid-cols-1">
        <Skeleton className="h-64" />
        <Skeleton className="h-64" />
      </div>
    </div>
  );
}

/** Trang 1: nhập một câu, xem nhãn ngay, giải thích tải sau, tuỳ chọn so sánh với mô hình cơ sở. */
export function AnalyzePage() {
  const appConfig = useAppConfig();
  const config = appConfig.status === "ready" ? appConfig.config : undefined;
  const [text, setText] = useState("");
  const [compare, setCompare] = useState(false);
  const { state, baseline, run, retryExplain, reset } = useAnalysis(compare);

  const activeText = state.phase === "idle" ? null : state.text;

  function pickSample(sample: string) {
    setText(sample);
    run(sample);
  }

  function clear() {
    setText("");
    reset();
  }

  return (
    <>
      <PageHeader
        icon="rate_review"
        title="Phân tích câu"
        description="Nhập một câu góp ý của sinh viên. EduPulse đoán cảm xúc và chủ đề, cho biết độ tin cậy và tô các từ ảnh hưởng nhiều nhất đến kết quả."
      />

      <AnalyzeForm
        text={text}
        onTextChange={setText}
        onSubmit={() => run(text)}
        onClear={clear}
        loading={state.phase === "loading"}
        maxChars={config?.limits.max_chars}
        samples={config?.ui_samples ?? []}
        demoAvailable={config?.demo_available}
        demoReason={config?.demo_reason}
        activeText={activeText}
        onPickSample={pickSample}
        compare={compare}
        onCompareChange={setCompare}
      />

      <hr className="my-2 border-divider" />

      <p role="status" className="sr-only">
        {announcement(state)}
      </p>

      <div className="flex flex-col gap-4">
        {state.phase === "idle" ? <EmptyState /> : null}
        {state.phase === "loading" ? <ResultSkeleton /> : null}
        {state.phase === "error" && state.failure.code === "no_content" ? <NoContentNotice /> : null}
        {state.phase === "error" && state.failure.code !== "no_content" ? (
          <FailureNotice failure={state.failure} onRetry={() => run(state.text)} />
        ) : null}
        {state.phase === "done" ? (
          <>
            <ResultHeader analysis={state.analysis} defaultBudget={config?.default_budget} />
            {state.analysis.no_diacritics ? <NoDiacriticsNotice /> : null}
            {state.analysis.pii_types.length > 0 ? <PiiNotice displayText={state.analysis.display_text} /> : null}
            <ResultCards analysis={state.analysis} />
            <ExplainPanel
              analysis={state.analysis}
              explain={state.explain}
              maxWords={config?.limits.xai_max_words}
              onRetry={retryExplain}
            />
            {compare ? <ComparePanel analysis={state.analysis} explain={state.explain} baseline={baseline} /> : null}
          </>
        ) : null}
      </div>

      <Caption>
        <Icon name="lock" size="xs" /> Câu bạn nhập không được lưu. Kết quả do mô hình dự đoán, có thể sai.
      </Caption>
    </>
  );
}
