import { IllustrativeBadge } from "@/components/shared/illustrative-badge";
import { Caption } from "@/components/shared/section-heading";
import { Card } from "@/components/ui/card";
import type { InsightsResponse } from "@/lib/batch/types";
import { DELTA_SCALE, deltaColor } from "@/lib/colors";
import { heatmapRows, TOPIC_SHORT, type CellView } from "@/lib/dashboard/insight-view";
import { formatNumber, formatPercent, formatPoints } from "@/lib/format";
import { labelMeta } from "@/lib/labels";
import { cn } from "@/lib/utils";

type Heatmap = NonNullable<InsightsResponse["heatmap"]>;

const STEP_LABELS = ["≤ 0", "0–5", "5–10", "10–15", "15–20", "≥ 20 điểm"] as const;

const CELL_BASE =
  "flex min-h-[46px] flex-col items-center justify-center rounded-[4px] px-0.5 text-center leading-[1.2] tabular-nums";

function describeCell(group: string, topic: string, cell: CellView): string {
  const where = `${group}, ${topic}`;
  if (cell.kind === "empty") return `${where}: chưa có câu nào.`;
  const rate = formatPercent(cell.rate ?? 0);
  if (cell.kind === "sparse") {
    return `${where}: ${rate} tiêu cực, chỉ ${formatNumber(cell.n)} câu nên chưa đủ để so với mức nền.`;
  }
  const delta = cell.delta ?? 0;
  return `${where}: ${rate} tiêu cực, ${formatPoints(delta)} điểm so với mức nền, ${formatNumber(cell.n)} câu.`;
}

function Cell({ cell, label }: { cell: CellView; label: string }) {
  if (cell.kind === "shaded") {
    const { background, color } = deltaColor(cell.delta ?? 0);
    return (
      <div role="cell" aria-label={label} className={CELL_BASE} style={{ background, color }}>
        <b className="text-sm font-semibold">{formatPercent(cell.rate ?? 0)}</b>
        <small className="text-xs opacity-90">
          {formatPoints(cell.delta ?? 0)} · n {formatNumber(cell.n)}
        </small>
      </div>
    );
  }
  return (
    <div
      role="cell"
      aria-label={label}
      className={cn(CELL_BASE, "border-[1.5px] border-dashed border-border bg-transparent text-muted-foreground")}
    >
      <b className="text-sm font-semibold">{cell.kind === "empty" ? "–" : formatPercent(cell.rate ?? 0)}</b>
      <small className="text-xs">n {formatNumber(cell.n)}</small>
    </div>
  );
}

interface DeltaHeatmapProps {
  heatmap: Heatmap;
  /** Tên cột nhóm (vd. "Môn học"), để ghi vào tiêu đề. */
  groupLabel: string;
  illustrative: boolean;
  className?: string;
}

/**
 * Heatmap nhóm × chủ đề tô theo CHÊNH so với mức nền của chủ đề (không tô % thô vì nền từng chủ đề khác
 * nhau). Ô ít câu hơn `min_n` không tô, viền nét đứt. Dựng bằng CSS grid với vai trò bảng để đọc được
 * bằng trình đọc màn hình; mọi số đều nằm trong ô.
 */
export function DeltaHeatmap({ heatmap, groupLabel, illustrative, className }: DeltaHeatmapProps) {
  const rows = heatmapRows(heatmap);
  return (
    <Card className={className}>
      <div className="flex flex-col gap-1.5">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <h3 className="text-[15px] font-semibold">Tiêu cực theo {groupLabel} × Chủ đề</h3>
          {illustrative ? <IllustrativeBadge /> : null}
        </div>
        <Caption>Ô ghi % tiêu cực · màu = cao hơn mức nền của chủ đề bao nhiêu điểm</Caption>
      </div>
      <div className="overflow-x-auto">
        <div
          role="table"
          aria-label={`Tỉ lệ tiêu cực theo ${groupLabel} và chủ đề`}
          className="grid min-w-[420px] grid-cols-[minmax(96px,1.3fr)_repeat(4,minmax(0,1fr))] gap-[3px]"
        >
          <div role="row" className="contents">
            <span role="columnheader" className="sr-only">
              {groupLabel}
            </span>
            {heatmap.topics.map((topic) => (
              <span
                key={topic}
                role="columnheader"
                title={labelMeta("topic", topic).vi}
                className="self-end pb-0.5 text-center text-xs leading-[1.3] text-muted-foreground"
              >
                {TOPIC_SHORT[topic]}
              </span>
            ))}
          </div>
          {rows.map((row) => (
            <div key={row.group} role="row" className="contents">
              <span
                role="rowheader"
                className="flex items-center justify-end pr-1.5 text-right text-[13px] leading-[1.3]"
              >
                {row.group}
              </span>
              {row.cells.map((cell, column) => (
                <Cell
                  key={heatmap.topics[column]}
                  cell={cell}
                  label={describeCell(row.group, labelMeta("topic", heatmap.topics[column]).vi, cell)}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[13px] text-muted-foreground">
        {DELTA_SCALE.map((background, step) => (
          <span key={background} className="inline-flex items-center gap-1.5">
            <i
              aria-hidden="true"
              className={cn("size-3 rounded-[3px]", step === 0 && "border border-border")}
              style={{ background }}
            />
            {STEP_LABELS[step]}
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5">
          <i aria-hidden="true" className="size-3 rounded-[3px] border-[1.5px] border-dashed border-border" />
          &lt; {formatNumber(heatmap.min_n)} câu (không tô)
        </span>
      </div>
    </Card>
  );
}
