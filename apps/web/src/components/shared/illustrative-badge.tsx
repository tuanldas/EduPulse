import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

import { Icon } from "./icon";

/**
 * Nhãn cho dữ liệu nhóm gán ngẫu nhiên (`illustrative_groups` = true): kết quả theo nhóm không có ý nghĩa thật.
 * Nhãn dài (khoảng 310px) nên co theo khung chứa và tự xuống dòng, icon đứng ở dòng đầu, thay vì bị cắt hoặc
 * làm cả trang cuộn ngang ở chỗ hẹp (thẻ cảnh báo nhiều cột, màn hình nhỏ).
 */
export function IllustrativeBadge({ className }: { className?: string }) {
  return (
    <Badge
      variant="gray"
      className={cn(
        "max-w-full shrink items-start justify-start text-left whitespace-normal [&>svg]:mt-[3px] [&>svg]:shrink-0",
        className,
      )}
    >
      <Icon name="info" size="xs" />
      Dữ liệu minh hoạ — nhóm gán ngẫu nhiên
    </Badge>
  );
}
