"use client";

import { useTheme } from "next-themes";
import { useSyncExternalStore } from "react";

import { Icon, type IconName } from "@/components/shared/icon";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

const OPTIONS: readonly { value: string; label: string; icon: IconName }[] = [
  { value: "light", label: "Sáng", icon: "light_mode" },
  { value: "dark", label: "Tối", icon: "dark_mode" },
  { value: "system", label: "Theo hệ thống", icon: "computer" },
];

const subscribeNothing = () => () => {};

/** Công tắc Sáng / Tối / Theo hệ thống. Chọn một giá trị được lưu bởi next-themes (localStorage). */
export function ThemeToggle({ className }: { className?: string }) {
  const { theme, setTheme } = useTheme();
  // Trước khi hydrate chưa biết theme đã lưu: không đánh dấu mục nào để HTML khớp giữa máy chủ và trình duyệt.
  const mounted = useSyncExternalStore(
    subscribeNothing,
    () => true,
    () => false,
  );
  const current = mounted ? (theme ?? "system") : null;
  return (
    <ToggleGroup
      aria-label="Giao diện"
      size="sm"
      value={current ? [current] : []}
      onValueChange={(next) => {
        const chosen = next[0];
        if (chosen) setTheme(chosen);
      }}
      className={className}
    >
      {OPTIONS.map((option) => (
        <ToggleGroupItem key={option.value} value={option.value} aria-label={option.label} title={option.label}>
          <Icon name={option.icon} size="sm" />
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
