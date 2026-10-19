"use client";

import { useEffect, useRef, useState, type RefObject } from "react";

/**
 * Chiều rộng hiện tại của một phần tử (px), để vẽ biểu đồ SVG đúng cỡ chữ thật thay vì co giãn cả
 * khung (chữ trục sẽ nhỏ hơn 12px trên điện thoại). `null` cho tới khi đo được lần đầu.
 */
export function useElementWidth<T extends HTMLElement>(): [RefObject<T | null>, number | null] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState<number | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[entries.length - 1];
      if (entry) setWidth(Math.round(entry.contentRect.width));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, width];
}
