import { SegmentedControl, type SegmentedOption } from "@/components/shared/segmented-control";
import { formatNumber } from "@/lib/format";

export type InsightSource = "demo" | "batch";

interface SourceSwitchProps {
  value: InsightSource;
  onValueChange: (source: InsightSource) => void;
  /** `demo_available` của `/api/config`. */
  demoAvailable: boolean;
  /** Số câu của dữ liệu demo, khi đã biết (sau lần tải đầu tiên). */
  demoRows: number | null;
  /** Có file vừa tải (còn trong tab này) để xem. */
  fileAvailable: boolean;
  fileRows: number | null;
}

/** "Nguồn dữ liệu": dữ liệu demo UIT-VSFC hoặc file vừa tải ở trang Phân tích hàng loạt. */
export function SourceSwitch({
  value,
  onValueChange,
  demoAvailable,
  demoRows,
  fileAvailable,
  fileRows,
}: SourceSwitchProps) {
  const options: SegmentedOption<InsightSource>[] = [
    {
      value: "demo",
      icon: "dataset",
      disabled: !demoAvailable,
      label: `Dữ liệu demo · UIT-VSFC${demoRows === null ? "" : ` (${formatNumber(demoRows)} câu)`}`,
    },
    {
      value: "batch",
      icon: "upload_file",
      disabled: !fileAvailable,
      label: `File vừa tải${fileRows === null ? "" : ` · ${formatNumber(fileRows)} câu`}`,
    },
  ];
  return (
    <div>
      <div className="mb-1.5 text-sm">Nguồn dữ liệu</div>
      <SegmentedControl value={value} onValueChange={onValueChange} options={options} ariaLabel="Nguồn dữ liệu" />
    </div>
  );
}
