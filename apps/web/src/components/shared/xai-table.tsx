import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { HeadExplanation } from "@/lib/api/types";
import { formatPercent, formatPoints } from "@/lib/format";
import { explainRows } from "@/lib/xai";
import { cn } from "@/lib/utils";

interface XaiTableProps {
  head: HeadExplanation;
  /** Tên nhãn tiếng Việt của lớp đang giải thích, vd. "Tiêu cực". */
  labelVi: string;
  className?: string;
}

/** Bảng "Bỏ từ này thì “Nhãn” …: 93% → 46%" cho các từ ủng hộ mạnh nhất và từ phản đối. */
export function XaiTable({ head, labelVi, className }: XaiTableProps) {
  const rows = explainRows(head);
  const maxChange = Math.max(...rows.map((row) => Math.abs(row.changePoints)), 1);
  return (
    <Table className={className} containerClassName="min-w-0">
      <caption className="sr-only">Từ ảnh hưởng nhất đến nhãn “{labelVi}” và xác suất khi bỏ từng từ</caption>
      <TableHeader>
        <TableRow>
          <TableHead className="w-px text-right">#</TableHead>
          <TableHead>Từ</TableHead>
          <TableHead>Bỏ từ này thì “{labelVi}”</TableHead>
          <TableHead>Mức thay đổi</TableHead>
          <TableHead>Vai trò</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((row) => (
          <TableRow key={`${row.role}-${row.index}`}>
            <TableCell className="text-right text-muted-foreground tabular-nums">
              {row.rank ?? "−"}
            </TableCell>
            <TableCell className="font-semibold">{row.text}</TableCell>
            <TableCell className="whitespace-nowrap tabular-nums">
              {formatPercent(row.pFull)} → {formatPercent(row.pWithout)}
            </TableCell>
            <TableCell>
              <div className="flex min-w-28 items-center gap-2 tabular-nums">
                <span className="h-1.5 min-w-12 flex-1 overflow-hidden rounded-[3px] bg-grid" aria-hidden="true">
                  <i
                    className={cn("block h-full", row.role === "support" ? "bg-primary" : "bg-neu")}
                    style={{ width: `${(Math.abs(row.changePoints) / maxChange) * 100}%` }}
                  />
                </span>
                {formatPoints(row.changePoints)}
              </div>
            </TableCell>
            <TableCell className={cn(row.role === "oppose" && "text-muted-foreground")}>
              {row.role === "support" ? "Ủng hộ" : "Phản đối"}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
