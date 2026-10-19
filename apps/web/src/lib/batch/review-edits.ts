import { useCallback, useSyncExternalStore } from "react";

import {
  applyChange,
  countOverrides,
  NO_OVERRIDES,
  parseOverrides,
  type LabelChange,
  type Overrides,
} from "./overrides";
import { createSessionStore, type SessionStore } from "./session-store";

/**
 * Nhãn người duyệt đã sửa của một batch. `rev` tăng mỗi lần sửa; `exportedRev` là `rev` tại lần
 * xuất gần nhất, nên "còn sửa chưa xuất" ⇔ `rev !== exportedRev`.
 */
export interface ReviewEdits {
  overrides: Overrides;
  rev: number;
  exportedRev: number;
}

export const NO_EDITS: ReviewEdits = Object.freeze({
  overrides: NO_OVERRIDES,
  rev: 0,
  exportedRev: 0,
});

/** Có nhãn đã sửa mà chưa nằm trong file nào đã tải: đóng tab lúc này sẽ mất công duyệt. */
export function hasUnexportedEdits(edits: ReviewEdits): boolean {
  return edits.rev !== edits.exportedRev && countOverrides(edits.overrides) > 0;
}

export function changeLabel(edits: ReviewEdits, index: number, change: LabelChange): ReviewEdits {
  return { ...edits, overrides: applyChange(edits.overrides, index, change), rev: edits.rev + 1 };
}

export function markExported(edits: ReviewEdits): ReviewEdits {
  return { ...edits, exportedRev: edits.rev };
}

function nonNegativeInt(value: unknown): number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : 0;
}

export function parseReviewEdits(raw: unknown): ReviewEdits {
  if (typeof raw !== "object" || raw === null) return NO_EDITS;
  const value = raw as Record<string, unknown>;
  const rev = nonNegativeInt(value.rev);
  return {
    overrides: parseOverrides(value.overrides),
    rev,
    exportedRev: Math.min(nonNegativeInt(value.exportedRev), rev),
  };
}

const KEY_PREFIX = "edupulse.edits.";
const stores = new Map<string, SessionStore<ReviewEdits>>();

function storeFor(batchId: string): SessionStore<ReviewEdits> {
  let store = stores.get(batchId);
  if (store === undefined) {
    store = createSessionStore<ReviewEdits>(KEY_PREFIX + batchId, NO_EDITS, parseReviewEdits);
    stores.set(batchId, store);
  }
  return store;
}

/** Kho rỗng cho trường hợp chưa có batch: đọc ra `NO_EDITS`, ghi bị bỏ qua. */
const NO_STORE: SessionStore<ReviewEdits> = {
  subscribe: () => () => {},
  getSnapshot: () => NO_EDITS,
  getServerSnapshot: () => NO_EDITS,
  set: () => {},
  remove: () => {},
};

/** Xoá nhãn đã sửa của batch (khi batch bị đóng hoặc người dùng bắt đầu lại). */
export function discardReviewEdits(batchId: string): void {
  storeFor(batchId).remove();
}

export interface ReviewEditsApi {
  edits: ReviewEdits;
  /** Số câu có nhãn đã sửa. */
  count: number;
  /** Còn nhãn sửa chưa xuất: bật cảnh báo trước khi rời trang. */
  unexported: boolean;
  change: (index: number, change: LabelChange) => void;
  markExported: () => void;
}

export function useReviewEdits(batchId: string | null): ReviewEditsApi {
  const store = batchId === null ? NO_STORE : storeFor(batchId);
  const edits = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getServerSnapshot);

  const change = useCallback(
    (index: number, patch: LabelChange) => {
      store.set(changeLabel(store.getSnapshot(), index, patch));
    },
    [store],
  );
  const exported = useCallback(() => {
    store.set(markExported(store.getSnapshot()));
  }, [store]);

  return {
    edits,
    count: countOverrides(edits.overrides),
    unexported: hasUnexportedEdits(edits),
    change,
    markExported: exported,
  };
}
