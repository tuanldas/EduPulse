import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { ModelCard } from "@/lib/api/types";
import { formatDecimal, formatScore } from "@/lib/format";
import { cn } from "@/lib/utils";

type Baseline = ModelCard["baselines"][number];
type Architecture = ModelCard["architectures"][number];

const DASH = "—";

// Tên mô hình do bundle đặt và có thể dài ("PhoBERT đa nhiệm · int8 (bản triển khai)", "Đơn nhiệm (STL), sơ bộ:
// validation, 1 seed"), nên cột tên được xuống dòng thay vì đẩy các cột số ra ngoài khung bảng.
const NAME_CELL = "min-w-48";

function BaselineRowView({ row }: { row: Baseline }) {
  const strong = row.deployed ? "font-semibold" : undefined;
  return (
    <TableRow data-highlight={row.deployed}>
      <TableCell className={cn(NAME_CELL, strong)}>
        {row.name} {row.deployed ? <Badge>đang dùng</Badge> : null}
      </TableCell>
      <TableCell className={cn("text-right tabular-nums", strong)}>{formatScore(row.sentiment_accuracy)}</TableCell>
      <TableCell className={cn("text-right tabular-nums", strong)}>{formatScore(row.sentiment_macro_f1)}</TableCell>
      <TableCell className={cn("text-right tabular-nums", strong)}>{formatScore(row.topic_accuracy)}</TableCell>
      <TableCell className={cn("text-right tabular-nums", strong)}>{formatScore(row.topic_macro_f1)}</TableCell>
      <TableCell className={cn("text-right tabular-nums", strong)}>
        {row.sentiment_ece == null ? DASH : formatDecimal(row.sentiment_ece, 3)}
      </TableCell>
    </TableRow>
  );
}

function ArchitectureRowView({ row }: { row: Architecture }) {
  const stat = (value: { mean: number; std: number; n: number }) =>
    `${formatScore(value.mean)} ± ${formatScore(value.std)}`;
  // Số seed chỉ ghi thêm khi tên chưa nói (tránh "1 seed (1 seed)").
  const seedNote = /seed/i.test(row.name) ? null : (
    <span className="text-xs text-muted-foreground">({row.seeds.length} seed)</span>
  );
  return (
    <TableRow>
      <TableCell className={NAME_CELL}>
        {row.name} {seedNote}
      </TableCell>
      <TableCell className="text-right text-muted-foreground">{DASH}</TableCell>
      <TableCell className="text-right whitespace-nowrap tabular-nums">{stat(row.sentiment_macro_f1)}</TableCell>
      <TableCell className="text-right text-muted-foreground">{DASH}</TableCell>
      <TableCell className="text-right whitespace-nowrap tabular-nums">{stat(row.topic_macro_f1)}</TableCell>
      <TableCell className="text-right text-muted-foreground">{DASH}</TableCell>
    </TableRow>
  );
}

/**
 * So sánh các mô hình cơ sở với mô hình đang dùng (dòng đang dùng in đậm). Mô hình huấn luyện nhiều
 * seed (đơn nhiệm / đa nhiệm) chỉ có Macro-F1 dạng trung bình ± độ lệch chuẩn nên ô khác ghi "—".
 */
export function ComparisonTable({ baselines, architectures }: { baselines: Baseline[]; architectures: Architecture[] }) {
  if (baselines.length === 0 && architectures.length === 0) return null;
  return (
    <Table label="So sánh với mô hình cơ sở">
      <caption className="sr-only">So sánh với mô hình cơ sở</caption>
      <TableHeader>
        <TableRow>
          <TableHead>Mô hình</TableHead>
          <TableHead className="text-right">Cảm xúc · Accuracy</TableHead>
          <TableHead className="text-right">Cảm xúc · Macro-F1</TableHead>
          <TableHead className="text-right">Chủ đề · Accuracy</TableHead>
          <TableHead className="text-right">Chủ đề · Macro-F1</TableHead>
          <TableHead className="text-right">ECE cảm xúc</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {baselines.map((row) => (
          <BaselineRowView key={row.name} row={row} />
        ))}
        {architectures.map((row) => (
          <ArchitectureRowView key={row.name} row={row} />
        ))}
      </TableBody>
    </Table>
  );
}
