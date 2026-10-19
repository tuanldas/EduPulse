import * as React from "react";

import { cn } from "@/lib/utils";

import { ICON_PATHS, type IconName } from "./icon-paths";

export type { IconName } from "./icon-paths";

// Material Symbols Rounded đúng thiết kế (docs/design-guidelines.md mục 9), nhúng dạng path SVG nên
// không phải tải font icon. `lucide-react` chỉ còn nằm trong các component shadcn sinh ra.
const SIZE_CLASS = {
  xs: "size-4",
  sm: "size-[18px]",
  md: "size-5",
  lg: "size-6",
} as const;

export interface IconProps extends Omit<React.SVGProps<SVGSVGElement>, "children"> {
  name: IconName;
  /** Dùng bản đặc nếu icon có (vd. `flag`). */
  filled?: boolean;
  size?: keyof typeof SIZE_CLASS;
  /** Có nhãn thì icon tự mang nghĩa (role=img); mặc định chỉ để trang trí, ẩn khỏi trình đọc màn hình. */
  label?: string;
}

export function Icon({ name, filled = false, size = "md", label, className, ...props }: IconProps) {
  const filledName = `${name}-fill`;
  const key: IconName = filled && filledName in ICON_PATHS ? (filledName as IconName) : name;
  return (
    <svg
      viewBox="0 -960 960 960"
      fill="currentColor"
      focusable="false"
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      className={cn("inline-block shrink-0 align-[-0.2em]", SIZE_CLASS[size], className)}
      {...props}
    >
      <path d={ICON_PATHS[key]} />
    </svg>
  );
}
