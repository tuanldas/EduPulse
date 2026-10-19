import type { ReactNode } from "react";

import { Icon } from "@/components/shared/icon";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface HelpTipProps {
  /** Tên của nút cho trình đọc màn hình, vd. "Giải thích ngân sách duyệt". */
  label: string;
  children: ReactNode;
}

/** Dấu "?" nhỏ cạnh nhãn: rê chuột hoặc focus bàn phím để đọc giải thích ngắn (st.help). */
export function HelpTip({ label, children }: HelpTipProps) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <button
            type="button"
            aria-label={label}
            className="rounded-full text-muted-foreground outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
          />
        }
      >
        <Icon name="help" size="xs" />
      </TooltipTrigger>
      <TooltipContent>{children}</TooltipContent>
    </Tooltip>
  );
}
