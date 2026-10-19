"use client";

import { useId } from "react";

import { WrappingIllustrativeBadge } from "@/components/batch/wrapping-illustrative-badge";
import { Icon } from "@/components/shared/icon";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import type { InsightsResponse } from "@/lib/batch/types";
import { groupSummary } from "@/lib/dashboard/insight-view";
import { formatNumber } from "@/lib/format";

interface GroupFilterProps {
  /** Tên cột nhóm đã dùng khi chạy (vd. "Môn học"). */
  columnLabel: string;
  groups: NonNullable<InsightsResponse["groups"]>;
  /** Nhóm đang chọn; rỗng nghĩa là tất cả. */
  selected: readonly string[];
  onChange: (selected: string[]) => void;
  illustrative: boolean;
}

/** "Lọc theo Môn học": nhiều lựa chọn, chọn hết hoặc không chọn gì đều là xem tất cả. */
export function GroupFilter({ columnLabel, groups, selected, onChange, illustrative }: GroupFilterProps) {
  const chosen = new Set(selected);
  const labelId = useId();
  const valueId = useId();

  function toggle(name: string, checked: boolean) {
    const next = new Set(chosen);
    if (checked) next.add(name);
    else next.delete(name);
    // Giữ thứ tự của máy chủ và coi "chọn đủ" là "tất cả" (không gửi bộ lọc thừa).
    const ordered = groups.map((group) => group.name).filter((groupName) => next.has(groupName));
    onChange(ordered.length === groups.length ? [] : ordered);
  }

  return (
    <div>
      <div className="mb-1.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-sm">
        <span id={labelId}>Lọc theo {columnLabel}</span>
        {illustrative ? <WrappingIllustrativeBadge /> : null}
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <Button
              variant="outline"
              aria-labelledby={`${labelId} ${valueId}`}
              className="w-full justify-between bg-muted px-3.5 font-normal"
            />
          }
        >
          <span id={valueId} className="truncate">
            {groupSummary(selected, groups.length)}
          </span>
          <Icon name="keyboard_arrow_down" size="xs" className="text-muted-foreground" />
        </DropdownMenuTrigger>
        <DropdownMenuContent className="max-h-80 min-w-64">
          <DropdownMenuItem disabled={selected.length === 0} onClick={() => onChange([])}>
            <Icon name="restart_alt" size="xs" />
            Xem tất cả {formatNumber(groups.length)} nhóm
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          {groups.map((group) => (
            <DropdownMenuCheckboxItem
              key={group.name}
              checked={chosen.has(group.name)}
              onCheckedChange={(checked) => toggle(group.name, checked)}
            >
              <span className="min-w-0 flex-1 truncate">{group.name}</span>
              <span className="text-xs text-muted-foreground tabular-nums">{formatNumber(group.n)}</span>
            </DropdownMenuCheckboxItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
