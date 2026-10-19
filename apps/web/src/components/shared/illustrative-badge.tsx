import { Badge } from "@/components/ui/badge";

import { Icon } from "./icon";

/** Nhãn cho dữ liệu nhóm gán ngẫu nhiên (`illustrative_groups` = true): kết quả theo nhóm không có ý nghĩa thật. */
export function IllustrativeBadge({ className }: { className?: string }) {
  return (
    <Badge variant="gray" className={className}>
      <Icon name="info" size="xs" />
      Dữ liệu minh hoạ — nhóm gán ngẫu nhiên
    </Badge>
  );
}
