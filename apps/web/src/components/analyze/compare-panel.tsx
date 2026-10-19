import * as React from "react";

import { Callout } from "@/components/shared/callout";
import { FailureNotice } from "@/components/shared/failure-notice";
import { LabelBadge } from "@/components/shared/label-badge";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { AnalyzeResponse, BaselineComparison, ExplainResponse } from "@/lib/api/types";
import { formatNumber } from "@/lib/format";

import type { BaselineState, ExplainState } from "./use-analysis";

/** Danh sách từ in đậm cách nhau bằng dấu phẩy; rỗng thì ghi gạch ngang. */
function Terms({ terms }: { terms: readonly string[] }) {
  if (terms.length === 0) return <>—</>;
  return (
    <>
      {terms.map((term, index) => (
        <React.Fragment key={`${term}-${index}`}>
          {index > 0 ? ", " : ""}
          <b className="font-semibold text-foreground">{term}</b>
        </React.Fragment>
      ))}
    </>
  );
}

function supportWords(explain: ExplainResponse, task: "sentiment" | "topic"): string[] {
  const head = explain[task];
  return head.support.map((index) => head.tokens[index]?.text).filter((text): text is string => Boolean(text));
}

function elapsedText(ms: number): string {
  return ms < 1000 ? `${formatNumber(ms, ms < 10 ? 1 : 0)} ms` : `${formatNumber(ms / 1000, 1)} giây`;
}

/** Câu kết luận khi so hai mô hình: cùng nhãn hay bất đồng ở tác vụ nào. */
function agreementMessage(baseline: BaselineComparison): { tone: "ok" | "info"; title: string; rest: string } {
  const { agree_sentiment: sentiment, agree_topic: topic } = baseline;
  if (sentiment && topic) {
    return { tone: "ok", title: "Hai mô hình cùng nhãn", rest: "cho cả cảm xúc và chủ đề." };
  }
  const what = !sentiment && !topic ? "cả cảm xúc và chủ đề" : sentiment ? "chủ đề" : "cảm xúc";
  return {
    tone: "info",
    title: `Hai mô hình khác nhau về ${what}.`,
    rest: "Khi hai mô hình bất đồng, nên để người xem xét câu này.",
  };
}

interface ComparePanelProps {
  analysis: AnalyzeResponse;
  explain: ExplainState;
  baseline: BaselineState;
}

/** So sánh PhoBERT (đang dùng) với mô hình cơ sở TF-IDF + LinearSVC trên cùng câu. */
export function ComparePanel({ analysis, explain, baseline }: ComparePanelProps) {
  const explained = explain.status === "ready" ? explain.data : null;
  const agreement = baseline.status === "ready" ? agreementMessage(baseline.data) : null;
  return (
    <section aria-label="So sánh mô hình" className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4 max-[640px]:grid-cols-1">
        <Card>
          <div className="flex items-center justify-between gap-2">
            <b className="font-semibold">PhoBERT đa nhiệm</b>
            <Badge>Đang dùng</Badge>
          </div>
          <div className="flex flex-wrap gap-2">
            <LabelBadge task="sentiment" label={analysis.sentiment.label} confidence={analysis.sentiment.confidence} />
            <LabelBadge task="topic" label={analysis.topic.label} confidence={analysis.topic.confidence} />
          </div>
          <p className="text-sm leading-[1.55] text-muted-foreground">
            {explained ? (
              <>
                Từ quyết định: <Terms terms={supportWords(explained, "sentiment")} /> · chủ đề:{" "}
                <Terms terms={supportWords(explained, "topic")} />.
              </>
            ) : explain.status === "skipped" ? (
              "Câu dài nên không tính từ quyết định."
            ) : (
              "Từ quyết định hiện khi phần giải thích tải xong."
            )}
          </p>
        </Card>

        {baseline.status === "ready" ? (
          <Card>
            <div className="flex items-center justify-between gap-2">
              <b className="font-semibold">{baseline.data.name}</b>
              <Badge variant="gray">Mô hình cơ sở</Badge>
            </div>
            <div className="flex flex-wrap gap-2">
              <LabelBadge
                task="sentiment"
                label={baseline.data.sentiment.label}
                confidence={baseline.data.sentiment.confidence}
              />
              <LabelBadge task="topic" label={baseline.data.topic.label} confidence={baseline.data.topic.confidence} />
            </div>
            <p className="text-sm leading-[1.55] text-muted-foreground">
              Từ quyết định: <Terms terms={baseline.data.sentiment.top_terms} /> · chủ đề:{" "}
              <Terms terms={baseline.data.topic.top_terms} />. Mất {elapsedText(baseline.data.elapsed_ms)}.
            </p>
          </Card>
        ) : baseline.status === "loading" ? (
          <Card aria-busy="true" aria-label="Đang tính mô hình cơ sở">
            <Skeleton className="h-5 w-48" />
            <Skeleton className="h-6 w-64" />
            <Skeleton className="h-4 w-full" />
          </Card>
        ) : baseline.status === "error" ? (
          <div className="flex items-start">
            <FailureNotice failure={baseline.failure} className="w-full" />
          </div>
        ) : null}
      </div>

      {agreement ? (
        <Callout tone={agreement.tone} icon="compare_arrows" role="status">
          <b className="font-semibold">{agreement.title}</b> {agreement.rest}
        </Callout>
      ) : null}
    </section>
  );
}
