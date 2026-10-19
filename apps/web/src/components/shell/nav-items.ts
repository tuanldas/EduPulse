import type { IconName } from "@/components/shared/icon-paths";

export interface NavItem {
  href: string;
  label: string;
  icon: IconName;
}

// Bốn trang theo wireframe. Có dấu "/" cuối vì bản export tĩnh dùng `trailingSlash`.
export const NAV_ITEMS: readonly NavItem[] = [
  { href: "/", label: "Phân tích câu", icon: "rate_review" },
  { href: "/batch/", label: "Phân tích hàng loạt", icon: "upload_file" },
  { href: "/dashboard/", label: "Dashboard & insight", icon: "monitoring" },
  { href: "/model/", label: "Mô hình & đạo đức", icon: "balance" },
];

function trimSlash(path: string): string {
  return path.length > 1 ? path.replace(/\/+$/, "") : path;
}

/** Mục điều hướng đang mở: so khớp đường dẫn bỏ dấu "/" cuối, trang gốc chỉ khớp chính nó. */
export function isActivePath(pathname: string, href: string): boolean {
  const current = trimSlash(pathname);
  const target = trimSlash(href);
  return target === "/" ? current === "/" : current === target || current.startsWith(`${target}/`);
}
