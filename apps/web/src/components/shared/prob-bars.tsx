import { Icon } from "@/components/shared/icon";
import { TONE_CLASS } from "@/lib/colors";
import { formatPercent } from "@/lib/format";
import { labelMeta, type Task } from "@/lib/labels";
import { cn } from "@/lib/utils";

interface ProbBarsProps {
  task: Task;
  /** Nhãn -> xác suất (0..1), như `probs` của API. */
  probs: Readonly<Record<string, number>>;
  className?: string;
}

/**
 * Xác suất từng lớp, xếp giảm dần (lớp cao nhất in đậm). Số % luôn có bằng chữ; thanh chỉ là phụ nên
 * ẩn khỏi trình đọc màn hình.
 */
export function ProbBars({ task, probs, className }: ProbBarsProps) {
  const rows = Object.entries(probs).sort(([, a], [, b]) => b - a);
  return (
    <ul
      aria-label="Xác suất từng lớp"
      className={cn(
        "grid grid-cols-[minmax(92px,auto)_minmax(0,1fr)_44px] items-center gap-x-2.5 gap-y-1.5 text-sm max-[640px]:grid-cols-[minmax(84px,auto)_minmax(0,1fr)_40px]",
        className,
      )}
    >
      {rows.map(([label, probability], position) => {
        const meta = labelMeta(task, label);
        const tone = TONE_CLASS[meta.tone];
        return (
          <li key={label} className="contents">
            <span className={cn("flex items-center gap-1.5 whitespace-nowrap", position === 0 && "font-semibold")}>
              <Icon name={meta.icon} size="xs" className={tone.text} />
              {meta.vi}
            </span>
            <span className="h-3.5 overflow-hidden rounded-[3px] bg-grid" aria-hidden="true">
              <i className={cn("block h-full", tone.solid)} style={{ width: `${Math.min(1, Math.max(0, probability)) * 100}%` }} />
            </span>
            <span className="text-right text-muted-foreground tabular-nums">{formatPercent(probability)}</span>
          </li>
        );
      })}
    </ul>
  );
}
