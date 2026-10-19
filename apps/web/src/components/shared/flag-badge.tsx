import { Badge } from "@/components/ui/badge";

import { Icon } from "./icon";

/** Cờ "Cần xem xét": icon cờ đặc + chữ, không dựa vào màu vàng để truyền nghĩa. */
export function FlagBadge({ size = "default", className }: { size?: "default" | "lg"; className?: string }) {
  return (
    <Badge variant="flag" size={size} className={className}>
      <Icon name="flag" filled size="xs" />
      Cần xem xét
    </Badge>
  );
}
