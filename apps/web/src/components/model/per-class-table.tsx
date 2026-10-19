import { Icon } from "@/components/shared/icon";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { ModelCard } from "@/lib/api/types";
import { TONE_CLASS } from "@/lib/colors";
import { formatDecimal, formatNumber } from "@/lib/format";
import { labelMeta, type Task } from "@/lib/labels";
import { cn } from "@/lib/utils";

type TaskMetrics = ModelCard["metrics"]["sentiment"];

interface PerClassTableProps {
  task: Task;
  metrics: TaskMetrics;
  /** "test" hoặc "validation", để ghi vào tiêu đề cột số câu. */
  splitName: string;
}

/** Bảng chỉ số từng lớp: Precision, Recall, F1 (kèm thanh) và số câu của lớp trong tập đánh giá. */
export function PerClassTable({ task, metrics, splitName }: PerClassTableProps) {
  return (
    <Table>
      <caption className="sr-only">Chỉ số từng lớp trên tập {splitName}</caption>
      <TableHeader>
        <TableRow>
          <TableHead>Lớp</TableHead>
          <TableHead className="text-right">Precision</TableHead>
          <TableHead className="text-right">Recall</TableHead>
          <TableHead>F1</TableHead>
          <TableHead className="text-right">Số câu</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {metrics.per_class.map((item) => {
          const meta = labelMeta(task, item.label);
          return (
            <TableRow key={item.label}>
              <TableCell>
                <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
                  <Icon name={meta.icon} size="xs" className={cn(TONE_CLASS[meta.tone].text)} />
                  {meta.vi}
                </span>
              </TableCell>
              <TableCell className="text-right tabular-nums">{formatDecimal(item.precision)}</TableCell>
              <TableCell className="text-right tabular-nums">{formatDecimal(item.recall)}</TableCell>
              <TableCell>
                <div className="flex min-w-24 items-center gap-2 tabular-nums">
                  <span className="h-1.5 min-w-12 flex-1 overflow-hidden rounded-[3px] bg-grid" aria-hidden="true">
                    <i className="block h-full bg-primary" style={{ width: `${item.f1 * 100}%` }} />
                  </span>
                  {formatDecimal(item.f1)}
                </div>
              </TableCell>
              <TableCell className="text-right tabular-nums">{formatNumber(item.support)}</TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
