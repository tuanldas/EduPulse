import { Callout } from "@/components/shared/callout";
import type { ModelCard } from "@/lib/api/types";
import { sequentialColor } from "@/lib/colors";
import { formatNumber } from "@/lib/format";
import { labelMeta, type Task } from "@/lib/labels";

type TaskMetrics = ModelCard["metrics"]["sentiment"];

/**
 * Ma trận nhầm lẫn theo phần trăm của hàng nhãn thật. Dựng bằng `<table>` để trình đọc màn hình
 * đọc được từng ô; màu ô theo thang tuần tự tường minh, số luôn có trong ô (không chỉ dựa vào màu).
 */
export function ConfusionMatrix({ task, metrics }: { task: Task; metrics: TaskMetrics }) {
  const labels = metrics.per_class.map((item) => item.label);
  const square = metrics.confusion.length === labels.length && metrics.confusion.every((row) => row.length === labels.length);
  if (labels.length === 0 || !square) {
    return (
      <Callout tone="info" icon="info" role="status">
        Model card chưa có ma trận nhầm lẫn cho tác vụ này.
      </Callout>
    );
  }
  return (
    <table className="w-full border-separate border-spacing-[3px] text-[13px]">
      <caption className="sr-only">Ma trận nhầm lẫn, phần trăm theo hàng nhãn thật</caption>
      <thead>
        <tr>
          <td />
          {labels.map((label, column) => (
            <th
              key={label}
              scope="col"
              className="px-1 pb-0.5 text-center align-bottom text-xs leading-tight font-normal text-muted-foreground"
            >
              {column === 0 ? "Đoán: " : ""}
              {labelMeta(task, label).vi}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {metrics.confusion.map((row, index) => (
          <tr key={labels[index]}>
            <th scope="row" className="pr-1.5 text-right text-[13px] leading-tight font-normal">
              {index === 0 ? "Thật: " : ""}
              {labelMeta(task, labels[index]).vi}
            </th>
            {row.map((share, column) => {
              const { background, color } = sequentialColor(share);
              return (
                <td
                  key={labels[column]}
                  className="h-[46px] rounded text-center tabular-nums"
                  style={{ background, color }}
                >
                  <b className="text-sm font-semibold">{formatNumber(share * 100, 1)}</b>
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
