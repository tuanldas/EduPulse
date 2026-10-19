"use client";

import { useMemo, useState } from "react";

import { Icon } from "@/components/shared/icon";
import { MetricCard } from "@/components/shared/metric-card";
import { PiiText } from "@/components/shared/pii-chip";
import { Caption } from "@/components/shared/section-heading";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { LabelChange, Overrides } from "@/lib/batch/overrides";
import { pageOfPosition, rowsForTab, type ReviewTab } from "@/lib/batch/review-rows";
import type { ResultsResponse, ResultsSummary } from "@/lib/batch/types";
import { formatNumber, formatPercent } from "@/lib/format";
import { PII_ORDER, PII_TAG } from "@/lib/labels";

import { ExplainSheet } from "./explain-sheet";
import { ReviewTable } from "./review-table";

function piiCaption(summary: ResultsSummary): string {
  if (summary.pii_rows === 0) return "Không phát hiện";
  const rows = new Map(summary.pii_counts.map((item) => [item.type, item.rows]));
  return PII_ORDER.filter((type) => rows.has(type))
    .map((type) => `${PII_TAG[type]} ${formatNumber(rows.get(type) ?? 0)}`)
    .join(" · ");
}

function Metrics({ summary, correctedCount }: { summary: ResultsSummary; correctedCount: number }) {
  return (
    <div className="grid grid-cols-4 gap-4 max-[900px]:grid-cols-2 max-[640px]:grid-cols-1">
      <MetricCard
        label="Số câu"
        value={formatNumber(summary.n_analyzed)}
        caption={
          <>
            {summary.n_groups !== null ? <span className="block">{formatNumber(summary.n_groups)} nhóm</span> : null}
            {summary.n_skipped > 0 ? (
              <span className="block">Bỏ qua {formatNumber(summary.n_skipped)} dòng không có nội dung</span>
            ) : null}
            {summary.n_groups === null && summary.n_skipped === 0 ? <span className="block">Đã phân tích xong</span> : null}
          </>
        }
      />
      <MetricCard
        label="Cần duyệt"
        icon="flag"
        value={formatNumber(summary.n_flagged)}
        caption={`${formatPercent(summary.flagged_share, 1)} số câu (ngân sách ${formatPercent(summary.budget / 100)})`}
      />
      <MetricCard
        label="Đã ẩn thông tin cá nhân"
        icon="shield_person"
        value={
          <>
            {formatNumber(summary.pii_rows)} <span className="text-base font-medium">câu</span>
          </>
        }
        caption={piiCaption(summary)}
      />
      <MetricCard
        label="Đã sửa nhãn"
        icon="edit"
        value={
          <>
            {formatNumber(correctedCount)}{" "}
            <span className="text-base font-medium">/ {formatNumber(summary.n_review)}</span>
          </>
        }
        caption="câu trong danh sách duyệt"
      />
    </div>
  );
}

interface ResultsSectionProps {
  batchId: string;
  results: ResultsResponse;
  /** Tên cột nhóm đã chọn; null nếu không dùng nhóm. */
  groupLabel: string | null;
  xaiMaxWords: number;
  overrides: Overrides;
  correctedCount: number;
  onChange: (index: number, change: LabelChange) => void;
}

/** Bước 3: bốn số liệu, hai tab (Cần duyệt / Tất cả), bảng duyệt, ngăn giải thích và các dòng bị bỏ qua. */
export function ResultsSection({
  batchId,
  results,
  groupLabel,
  xaiMaxWords,
  overrides,
  correctedCount,
  onChange,
}: ResultsSectionProps) {
  const { summary } = results;
  const [tab, setTab] = useState<ReviewTab>("review");
  const [pages, setPages] = useState<Record<ReviewTab, number>>({ review: 1, all: 1 });
  const [explain, setExplain] = useState<{ tab: ReviewTab; position: number } | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);

  const rowsByTab = useMemo(
    () => ({ review: rowsForTab(results, "review"), all: rowsForTab(results, "all") }),
    [results],
  );

  function setPage(target: ReviewTab, page: number) {
    setPages((previous) => ({ ...previous, [target]: page }));
  }

  function openExplain(target: ReviewTab, position: number) {
    setExplain({ tab: target, position });
    setSheetOpen(true);
  }

  function moveExplain(position: number) {
    if (explain === null) return;
    setExplain({ tab: explain.tab, position });
    setPage(explain.tab, pageOfPosition(position));
  }

  const table = (target: ReviewTab) => (
    <ReviewTable
      rows={rowsByTab[target]}
      overrides={overrides}
      editable={target === "review"}
      groupLabel={groupLabel}
      xaiMaxWords={xaiMaxWords}
      page={pages[target]}
      onPageChange={(page) => setPage(target, page)}
      onChange={onChange}
      onExplain={(position) => openExplain(target, position)}
    />
  );

  return (
    <div className="flex flex-col gap-4">
      <Metrics summary={summary} correctedCount={correctedCount} />

      <Tabs value={tab} onValueChange={(value) => setTab(value as ReviewTab)}>
        <TabsList aria-label="Chọn danh sách câu">
          <TabsTrigger value="review">
            <Icon name="flag" size="xs" />
            Cần duyệt ({formatNumber(summary.n_review)})
          </TabsTrigger>
          <TabsTrigger value="all">Tất cả kết quả ({formatNumber(summary.n_analyzed)})</TabsTrigger>
        </TabsList>
        <TabsContent value="review" className="flex flex-col gap-3">
          <Caption>
            Gồm <b className="font-semibold text-foreground">{formatNumber(summary.n_flagged)} câu bị gắn cờ</b> và{" "}
            <b className="font-semibold text-foreground">{formatNumber(summary.n_audit)} câu chọn ngẫu nhiên</b> (không
            bị cờ, để kiểm tra chéo độ chính xác), sắp theo độ tin cậy tăng dần. Sửa nhãn ở hai cột “Sửa…”.{" "}
            <b className="font-semibold text-foreground">
              {formatNumber(summary.n_explained)} câu cờ đầu đã có giải thích sẵn
            </b>
            ; câu khác bấm “Giải thích” để tính (khoảng 1 giây/câu).
          </Caption>
          {table("review")}
        </TabsContent>
        <TabsContent value="all" className="flex flex-col gap-3">
          <Caption>
            Toàn bộ {formatNumber(summary.n_analyzed)} câu theo thứ tự trong file. Nhãn đã sửa hiện ở cột “Đã sửa”; muốn
            sửa nhãn, dùng tab “Cần duyệt”.
          </Caption>
          {table("all")}
        </TabsContent>
      </Tabs>

      {results.skipped.length > 0 ? (
        <Collapsible className="rounded-lg border border-border">
          <CollapsibleTrigger className="group flex min-h-11 w-full items-center gap-2.5 rounded-lg px-3.5 text-left text-[15px] outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
            <Icon name="block" className="text-muted-foreground" />
            <span>
              {formatNumber(results.skipped.length)} dòng không có nội dung để phân tích (chỉ có thông tin cá nhân hoặc
              ký hiệu)
            </span>
            <Icon
              name="keyboard_arrow_down"
              className="ml-auto text-muted-foreground transition-transform group-data-[panel-open]:rotate-180"
            />
          </CollapsibleTrigger>
          <CollapsibleContent className="flex flex-col gap-2 border-t border-divider px-4 pt-2 pb-4">
            <ul className="flex flex-col divide-y divide-divider">
              {results.skipped.map((row) => (
                <li key={row.index} className="flex gap-3 py-1.5 text-sm">
                  <span className="w-12 shrink-0 text-right text-muted-foreground tabular-nums">{row.source_row}</span>
                  <span className="min-w-0 break-words">
                    <PiiText text={row.text} />
                  </span>
                </li>
              ))}
            </ul>
            <Caption>Các dòng này không có nhãn và không nằm trong danh sách duyệt hay file xuất nhãn.</Caption>
          </CollapsibleContent>
        </Collapsible>
      ) : null}

      <ExplainSheet
        open={sheetOpen}
        onOpenChange={setSheetOpen}
        batchId={batchId}
        rows={explain === null ? [] : rowsByTab[explain.tab]}
        position={explain?.position ?? null}
        onPositionChange={moveExplain}
        xaiMaxWords={xaiMaxWords}
      />
    </div>
  );
}
