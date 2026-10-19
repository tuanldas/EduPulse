import { useId, type ComponentProps } from "react";

import { Icon } from "@/components/shared/icon";
import { cn } from "@/lib/utils";

export interface NativeSelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

interface NativeSelectProps
  extends Omit<ComponentProps<"select">, "value" | "onChange" | "size" | "children"> {
  value: string;
  onValueChange: (value: string) => void;
  options: readonly NativeSelectOption[];
  /** Lựa chọn rỗng đứng đầu (giá trị ""), vd. "—" cho "giữ nhãn của mô hình". */
  placeholder?: string;
  size?: "default" | "sm";
}

/**
 * Ô chọn dùng thẻ `<select>` gốc, dáng giống `Select`/`Input` của nền tảng. Bảng duyệt có tới 100 ô chọn
 * mỗi trang nên dùng bản gốc: nhẹ, đọc được bằng trình đọc màn hình, mở được bằng bàn phím và dùng bộ
 * chọn của hệ điều hành trên điện thoại.
 */
export function NativeSelect({
  value,
  onValueChange,
  options,
  placeholder,
  size = "default",
  className,
  id,
  ...props
}: NativeSelectProps) {
  const generatedId = useId();
  return (
    <span className="relative inline-flex w-full items-center">
      <select
        id={id ?? generatedId}
        value={value}
        onChange={(event) => onValueChange(event.target.value)}
        className={cn(
          "w-full appearance-none rounded-lg border border-input bg-muted pr-8 pl-3 text-foreground transition-colors outline-none",
          "focus-visible:border-primary focus-visible:ring-2 focus-visible:ring-ring/45",
          "disabled:cursor-not-allowed disabled:opacity-50",
          size === "sm" ? "h-8 text-sm" : "h-10 text-[15px] pl-3.5",
          value === "" && placeholder !== undefined && "text-muted-foreground",
          className,
        )}
        {...props}
      >
        {placeholder !== undefined && <option value="">{placeholder}</option>}
        {options.map((option) => (
          <option key={option.value} value={option.value} disabled={option.disabled}>
            {option.label}
          </option>
        ))}
      </select>
      <Icon
        name="keyboard_arrow_down"
        size="xs"
        className="pointer-events-none absolute right-2.5 text-muted-foreground"
      />
    </span>
  );
}
