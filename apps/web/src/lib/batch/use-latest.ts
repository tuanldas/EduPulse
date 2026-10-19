import { useEffect, useRef, type RefObject } from "react";

/**
 * Giữ giá trị mới nhất trong ref để timer và trình nghe sự kiện đọc được mà không phải gỡ/gắn lại
 * mỗi lần giá trị đổi. Chỉ đọc `ref.current` bên trong callback, đừng đọc lúc render.
 */
export function useLatest<T>(value: T): RefObject<T> {
  const ref = useRef(value);
  useEffect(() => {
    ref.current = value;
  });
  return ref;
}
