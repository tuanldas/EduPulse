"use client";

import * as React from "react";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

import { Icon, type IconName } from "./icon";

export interface SegmentedOption<T extends string> {
  value: T;
  label: React.ReactNode;
  icon?: IconName;
  disabled?: boolean;
}

interface SegmentedControlProps<T extends string> {
  value: T;
  onValueChange: (value: T) => void;
  options: readonly SegmentedOption<T>[];
  /** Tên của nhóm cho trình đọc màn hình, vd. "Chọn tác vụ để giải thích". */
  ariaLabel: string;
  className?: string;
}

/**
 * Điều khiển phân đoạn (st.segmented_control): luôn có đúng một mục được chọn. Base UI cho phép bỏ
 * chọn khi bấm lại mục đang chọn; ở đây chặn việc đó để giá trị không bao giờ rỗng.
 */
export function SegmentedControl<T extends string>({
  value,
  onValueChange,
  options,
  ariaLabel,
  className,
}: SegmentedControlProps<T>) {
  return (
    <ToggleGroup
      aria-label={ariaLabel}
      value={[value]}
      onValueChange={(next) => {
        const chosen = next[0];
        if (chosen !== undefined) onValueChange(chosen as T);
      }}
      className={className}
    >
      {options.map((option) => (
        <ToggleGroupItem key={option.value} value={option.value} disabled={option.disabled}>
          {option.icon ? <Icon name={option.icon} size="xs" /> : null}
          {option.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
