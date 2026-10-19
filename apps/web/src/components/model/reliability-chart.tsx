"use client";

import { useMemo } from "react";

import { Callout } from "@/components/shared/callout";
import { Icon } from "@/components/shared/icon";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { ModelCard } from "@/lib/api/types";
import { formatDecimal, formatNumber, formatPercent } from "@/lib/format";
import { calibrationExample, regroupReliability, trimLeadingEmpty, type DisplayBin } from "@/lib/reliability";
import { useElementWidth } from "@/lib/use-element-width";

type TaskMetrics = ModelCard["metrics"]["sentiment"];

const HEIGHT = 262;
const MARGIN = { left: 50, right: 20, top: 20, bottom: 42 };
const Y_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1];

interface ChartProps {
  bins: DisplayBin[];
  width: number;
  ece: number;
}

function ReliabilitySvg({ bins, width, ece }: ChartProps) {
  const x0 = bins[0].lower;
  const plotWidth = width - MARGIN.left - MARGIN.right;
  const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
  const x = (value: number) => MARGIN.left + ((value - x0) / (1 - x0)) * plotWidth;
  const y = (value: number) => MARGIN.top + (1 - value) * plotHeight;
  const xTicks = [...bins.map((bin) => bin.lower), 1];
  return (
    <svg
      viewBox={`0 0 ${width} ${HEIGHT}`}
      width={width}
      height={HEIGHT}
      role="img"
      aria-label={`Biểu đồ độ tin cậy: cột là tỉ lệ đúng thực tế trong từng khoảng độ tin cậy, ECE ${formatDecimal(ece, 3)}. Bảng số nằm bên dưới.`}
      className="max-w-full font-sans text-xs"
    >
      <g stroke="var(--grid)" strokeWidth="1">
        {Y_TICKS.map((tick) => (
          <line key={tick} x1={MARGIN.left} y1={y(tick)} x2={width - MARGIN.right} y2={y(tick)} />
        ))}
      </g>
      <g fill="var(--primary)" fillOpacity="0.78">
        {bins.map((bin) =>
          bin.n > 0 && bin.accuracy !== null ? (
            <rect
              key={bin.lower}
              x={x(bin.lower) + 1}
              y={y(bin.accuracy)}
              width={Math.max(1, x(bin.upper) - x(bin.lower) - 2)}
              height={Math.max(0, plotHeight * bin.accuracy)}
            />
          ) : null,
        )}
      </g>
      <line
        x1={x(x0)}
        y1={y(x0)}
        x2={x(1)}
        y2={y(1)}
        stroke="var(--foreground)"
        strokeWidth="1.5"
        strokeDasharray="5 4"
      />
      <g textAnchor="middle" fill="var(--muted-foreground)" stroke="var(--background)" strokeWidth="3" paintOrder="stroke">
        {bins.map((bin) =>
          bin.n > 0 && bin.accuracy !== null ? (
            <text key={bin.lower} x={(x(bin.lower) + x(bin.upper)) / 2} y={Math.max(12, y(bin.accuracy) - 5)}>
              {formatNumber(bin.n)}
            </text>
          ) : null,
        )}
      </g>
      <g textAnchor="middle" fill="var(--muted-foreground)">
        {xTicks.map((tick) => (
          <text key={tick} x={x(tick)} y={HEIGHT - 24}>
            {formatDecimal(tick, 1)}
          </text>
        ))}
        <text x={MARGIN.left + plotWidth / 2} y={HEIGHT - 6} fill="var(--foreground)">
          Độ tin cậy mô hình báo
        </text>
      </g>
      <g textAnchor="end" fill="var(--muted-foreground)">
        {Y_TICKS.map((tick) => (
          <text key={tick} x={MARGIN.left - 6} y={y(tick) + 4}>
            {formatDecimal(tick, 1)}
          </text>
        ))}
      </g>
      <text
        x={12}
        y={MARGIN.top + plotHeight / 2}
        fill="var(--foreground)"
        textAnchor="middle"
        transform={`rotate(-90 12 ${MARGIN.top + plotHeight / 2})`}
      >
        Tỉ lệ đúng thực tế
      </text>
    </svg>
  );
}

/** Bảng số tương đương của biểu đồ (đủ thông tin cho người không xem được hình). */
function ReliabilityTable({ bins }: { bins: DisplayBin[] }) {
  return (
    <Table label="Độ tin cậy báo so với tỉ lệ đúng thực tế theo từng khoảng">
      <caption className="sr-only">Độ tin cậy báo so với tỉ lệ đúng thực tế theo từng khoảng</caption>
      <TableHeader>
        <TableRow>
          <TableHead>Khoảng độ tin cậy</TableHead>
          <TableHead className="text-right">Số câu</TableHead>
          <TableHead className="text-right">Độ tin cậy trung bình</TableHead>
          <TableHead className="text-right">Tỉ lệ đúng thực tế</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {bins
          .filter((bin) => bin.n > 0)
          .map((bin) => (
            <TableRow key={bin.lower}>
              <TableCell className="tabular-nums">
                {formatDecimal(bin.lower, 1)} – {formatDecimal(bin.upper, 1)}
              </TableCell>
              <TableCell className="text-right tabular-nums">{formatNumber(bin.n)}</TableCell>
              <TableCell className="text-right tabular-nums">
                {bin.confidence === null ? "—" : formatPercent(bin.confidence, 1)}
              </TableCell>
              <TableCell className="text-right tabular-nums">
                {bin.accuracy === null ? "—" : formatPercent(bin.accuracy, 1)}
              </TableCell>
            </TableRow>
          ))}
      </TableBody>
    </Table>
  );
}

/**
 * Biểu đồ "Độ tin cậy có khớp thực tế không?" (reliability diagram) vẽ bằng SVG theo bề rộng thật của
 * khung: cột là tỉ lệ đúng trong từng khoảng độ tin cậy, đường nét đứt là khớp hoàn hảo.
 */
export function ReliabilityChart({ metrics }: { metrics: TaskMetrics }) {
  const [ref, measured] = useElementWidth<HTMLDivElement>();
  const bins = useMemo(() => trimLeadingEmpty(regroupReliability(metrics.reliability)), [metrics.reliability]);
  const width = Math.min(640, Math.max(280, measured ?? 420));
  const example = calibrationExample(bins);

  if (bins.length === 0) {
    return (
      <Callout tone="info" icon="info" role="status">
        Model card chưa có dữ liệu độ tin cậy cho tác vụ này.
      </Callout>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <div ref={ref} className="w-full">
        <ReliabilitySvg bins={bins} width={width} ece={metrics.ece} />
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <i className="inline-block size-3 rounded-[3px] bg-primary opacity-80" aria-hidden="true" />
          tỉ lệ đúng trong từng khoảng (số = số câu)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <i className="inline-block h-0 w-4 border-t-2 border-dashed border-foreground" aria-hidden="true" />
          khớp hoàn hảo
        </span>
      </div>
      <p className="text-sm leading-[1.55] text-muted-foreground">
        Cột càng sát đường chéo càng đáng tin
        {example && example.accuracy !== null
          ? `: trong các câu mô hình báo ${formatDecimal(example.lower * 100, 0)}–${formatDecimal(example.upper * 100, 0)}%, thực tế đúng ${formatPercent(example.accuracy)} (${formatNumber(example.n)} câu)`
          : ""}
        . Hiệu chỉnh bằng temperature scaling trên tập validation.
      </p>
      <Collapsible>
        <CollapsibleTrigger className="inline-flex items-center gap-1 rounded-md text-sm text-link outline-none focus-visible:ring-3 focus-visible:ring-ring/70 [&[data-panel-open]_svg]:rotate-180">
          <Icon name="keyboard_arrow_down" size="sm" className="transition-transform" />
          Xem bảng số
        </CollapsibleTrigger>
        <CollapsibleContent className="pt-2">
          <ReliabilityTable bins={bins} />
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
