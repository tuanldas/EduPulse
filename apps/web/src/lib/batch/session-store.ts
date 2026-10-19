/**
 * Kho nhỏ trên `sessionStorage` cho `useSyncExternalStore`: dữ liệu sống theo tab và mất khi đóng tab,
 * khớp lời hứa "không lưu dữ liệu bạn nhập". Storage bị chặn (chế độ riêng tư, chính sách trình
 * duyệt) thì rơi về bộ nhớ trong của trang thay vì báo lỗi.
 */

export interface SessionStore<T> {
  subscribe: (listener: () => void) => () => void;
  getSnapshot: () => T;
  /** Giá trị khi dựng tĩnh/hydrate: luôn là `empty`, để HTML dựng sẵn khớp lần vẽ đầu của trình duyệt. */
  getServerSnapshot: () => T;
  set: (value: T) => void;
  remove: () => void;
}

/**
 * @param key khoá trong sessionStorage
 * @param empty giá trị khi chưa có hoặc dữ liệu hỏng; PHẢI là một hằng số dùng lại (snapshot ổn định)
 * @param parse kiểm tra và dựng giá trị từ JSON đã đọc; ném lỗi hoặc trả `empty` khi sai hình dạng
 */
export function createSessionStore<T>(
  key: string,
  empty: T,
  parse: (raw: unknown) => T,
): SessionStore<T> {
  const listeners = new Set<() => void>();
  let storageUsable = true;
  let memoryRaw: string | null = null;
  let cachedRaw: string | null | undefined;
  let cachedValue: T = empty;

  function readRaw(): string | null {
    if (!storageUsable) return memoryRaw;
    try {
      return window.sessionStorage.getItem(key);
    } catch {
      storageUsable = false;
      return memoryRaw;
    }
  }

  function writeRaw(raw: string | null): void {
    memoryRaw = raw;
    if (!storageUsable) return;
    try {
      if (raw === null) window.sessionStorage.removeItem(key);
      else window.sessionStorage.setItem(key, raw);
    } catch {
      storageUsable = false;
    }
  }

  function notify(): void {
    for (const listener of listeners) listener();
  }

  function getSnapshot(): T {
    const raw = readRaw();
    if (raw === cachedRaw) return cachedValue;
    cachedRaw = raw;
    if (raw === null) {
      cachedValue = empty;
    } else {
      try {
        cachedValue = parse(JSON.parse(raw));
      } catch {
        cachedValue = empty;
      }
    }
    return cachedValue;
  }

  return {
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getSnapshot,
    getServerSnapshot: () => empty,
    set(value) {
      writeRaw(JSON.stringify(value));
      notify();
    },
    remove() {
      writeRaw(null);
      notify();
    },
  };
}
