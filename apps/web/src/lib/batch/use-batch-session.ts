import { useEffect } from "react";

import { isNotFound } from "./failure";
import { useLatest } from "./use-latest";

export interface UseBatchSessionOptions {
  /** null: không có phiên cần giữ (chưa có file, hoặc đang xem dữ liệu demo). */
  batchId: string | null;
  /** Chu kỳ heartbeat, lấy từ `/api/config` (`heartbeat_s`). */
  heartbeatS: number;
  sendHeartbeat: (batchId: string, signal: AbortSignal) => Promise<void>;
  /** Báo máy chủ xoá batch khi đóng tab; phải dùng được lúc trang đang đóng (`sendBeacon`). */
  sendClose: (batchId: string) => void;
  /** Heartbeat nhận 404: máy chủ không còn batch. */
  onGone: () => void;
  /** Còn nhãn sửa chưa xuất: hỏi lại trước khi đóng hoặc tải lại trang. */
  confirmLeave?: boolean;
}

/**
 * Giữ batch sống khi tab còn mở và dọn khi tab đóng:
 * - heartbeat mỗi `heartbeatS` giây (và ngay khi tab hiện lại sau lúc bị ẩn);
 * - `pagehide` → `sendClose`, nên batch bị xoá ngay thay vì chờ TTL;
 * - `beforeunload` hỏi lại khi còn nhãn sửa chưa xuất.
 * Rời trang bằng điều hướng trong ứng dụng (`<Link>`) chỉ dừng heartbeat, KHÔNG đóng batch, để
 * chuyển giữa Phân tích hàng loạt và Dashboard không mất phiên.
 */
export function useBatchSession(options: UseBatchSessionOptions): void {
  const { batchId, heartbeatS, confirmLeave = false } = options;
  const sendHeartbeat = useLatest(options.sendHeartbeat);
  const sendClose = useLatest(options.sendClose);
  const onGone = useLatest(options.onGone);

  useEffect(() => {
    if (batchId === null || !Number.isFinite(heartbeatS) || heartbeatS <= 0) return;
    let ended = false;
    let inFlight: AbortController | null = null;
    let timer: number | undefined;

    const stop = () => {
      ended = true;
      window.clearInterval(timer);
      inFlight?.abort();
    };

    const beat = () => {
      if (ended) return;
      inFlight?.abort();
      const controller = new AbortController();
      inFlight = controller;
      sendHeartbeat.current(batchId, controller.signal).catch((failure: unknown) => {
        if (controller.signal.aborted || ended) return;
        if (isNotFound(failure)) {
          stop();
          onGone.current();
        }
        // Lỗi mạng tạm thời: bỏ qua, lượt sau thử lại; TTL 30 phút là lưới an toàn.
      });
    };

    const start = () => {
      ended = false;
      window.clearInterval(timer);
      timer = window.setInterval(beat, heartbeatS * 1000);
    };
    start();

    const onVisibility = () => {
      if (document.visibilityState === "visible") beat();
    };
    // Trang được khôi phục từ bộ nhớ đệm đi/về (bfcache) sau khi `pagehide` đã gửi `close`: thăm dò
    // ngay, 404 nghĩa là phiên đã kết thúc (`onGone`), còn nếu máy chủ vẫn giữ batch thì heartbeat
    // chạy lại bình thường.
    const onPageShow = (event: PageTransitionEvent) => {
      if (!event.persisted) return;
      start();
      beat();
    };
    const onPageHide = () => {
      if (ended) return;
      stop();
      sendClose.current(batchId);
    };

    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("pageshow", onPageShow);
    window.addEventListener("pagehide", onPageHide);
    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pageshow", onPageShow);
      window.removeEventListener("pagehide", onPageHide);
    };
  }, [batchId, heartbeatS, sendHeartbeat, sendClose, onGone]);

  useEffect(() => {
    if (!confirmLeave) return;
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      // Trình duyệt cũ chỉ hiện hộp thoại khi `returnValue` được gán.
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [confirmLeave]);
}
