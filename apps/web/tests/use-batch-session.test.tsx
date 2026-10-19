import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useBatchSession, type UseBatchSessionOptions } from "@/lib/batch/use-batch-session";

class HttpError extends Error {
  constructor(readonly status: number) {
    super(`http ${status}`);
  }
}

function setup(overrides: Partial<UseBatchSessionOptions> = {}) {
  const sendHeartbeat = vi.fn<UseBatchSessionOptions["sendHeartbeat"]>().mockResolvedValue();
  const sendClose = vi.fn<UseBatchSessionOptions["sendClose"]>();
  const onGone = vi.fn<UseBatchSessionOptions["onGone"]>();
  const options: UseBatchSessionOptions = {
    batchId: "b1",
    heartbeatS: 60,
    sendHeartbeat,
    sendClose,
    onGone,
    ...overrides,
  };
  const hook = renderHook((props: UseBatchSessionOptions) => useBatchSession(props), {
    initialProps: options,
  });
  return { ...hook, sendHeartbeat, sendClose, onGone, options };
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe("useBatchSession", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("gửi heartbeat theo chu kỳ cấu hình, không gửi sớm hơn", async () => {
    const { sendHeartbeat } = setup({ heartbeatS: 60 });
    await advance(59_000);
    expect(sendHeartbeat).not.toHaveBeenCalled();
    await advance(1000);
    expect(sendHeartbeat).toHaveBeenCalledTimes(1);
    expect(sendHeartbeat.mock.calls[0][0]).toBe("b1");
    await advance(120_000);
    expect(sendHeartbeat).toHaveBeenCalledTimes(3);
  });

  it("không làm gì khi không có batch hoặc chu kỳ không hợp lệ", async () => {
    const none = setup({ batchId: null });
    await advance(300_000);
    expect(none.sendHeartbeat).not.toHaveBeenCalled();
    window.dispatchEvent(new Event("pagehide"));
    expect(none.sendClose).not.toHaveBeenCalled();

    const invalid = setup({ heartbeatS: 0 });
    await advance(300_000);
    expect(invalid.sendHeartbeat).not.toHaveBeenCalled();
  });

  it("gửi close một lần khi pagehide và ngừng heartbeat", async () => {
    const { sendHeartbeat, sendClose } = setup();
    window.dispatchEvent(new Event("pagehide"));
    window.dispatchEvent(new Event("pagehide"));
    expect(sendClose).toHaveBeenCalledTimes(1);
    expect(sendClose).toHaveBeenCalledWith("b1");
    await advance(300_000);
    expect(sendHeartbeat).not.toHaveBeenCalled();
  });

  it("rời trang bằng điều hướng trong ứng dụng không đóng batch", async () => {
    const { unmount, sendClose, sendHeartbeat } = setup();
    unmount();
    expect(sendClose).not.toHaveBeenCalled();
    window.dispatchEvent(new Event("pagehide"));
    expect(sendClose).not.toHaveBeenCalled();
    await advance(300_000);
    expect(sendHeartbeat).not.toHaveBeenCalled();
  });

  it("nhận 404 thì báo onGone một lần và dừng heartbeat", async () => {
    const { sendHeartbeat, onGone } = setup();
    sendHeartbeat.mockRejectedValue(new HttpError(404));
    await advance(60_000);
    expect(onGone).toHaveBeenCalledTimes(1);
    await advance(300_000);
    expect(sendHeartbeat).toHaveBeenCalledTimes(1);
    expect(onGone).toHaveBeenCalledTimes(1);
  });

  it("lỗi mạng tạm thời không kết thúc phiên, lượt sau vẫn thử lại", async () => {
    const { sendHeartbeat, onGone } = setup();
    sendHeartbeat.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await advance(60_000);
    await advance(60_000);
    expect(sendHeartbeat).toHaveBeenCalledTimes(2);
    expect(onGone).not.toHaveBeenCalled();
  });

  it("gửi heartbeat ngay khi tab hiện lại", async () => {
    const { sendHeartbeat } = setup();
    Object.defineProperty(document, "visibilityState", { value: "visible", configurable: true });
    document.dispatchEvent(new Event("visibilitychange"));
    expect(sendHeartbeat).toHaveBeenCalledTimes(1);
    Object.defineProperty(document, "visibilityState", { value: "hidden", configurable: true });
    document.dispatchEvent(new Event("visibilitychange"));
    expect(sendHeartbeat).toHaveBeenCalledTimes(1);
  });

  it("thăm dò ngay và chạy lại heartbeat khi trang được khôi phục từ bfcache", async () => {
    const { sendHeartbeat, sendClose } = setup();
    window.dispatchEvent(new Event("pagehide"));
    expect(sendClose).toHaveBeenCalledTimes(1);
    window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true }));
    expect(sendHeartbeat).toHaveBeenCalledTimes(1);
    await advance(60_000);
    expect(sendHeartbeat).toHaveBeenCalledTimes(2);

    window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: false }));
    expect(sendHeartbeat).toHaveBeenCalledTimes(2);
  });

  it("chuyển sang batch khác thì heartbeat theo batch mới", async () => {
    const { rerender, sendHeartbeat, options } = setup();
    rerender({ ...options, batchId: "b2" });
    await advance(60_000);
    expect(sendHeartbeat).toHaveBeenCalledTimes(1);
    expect(sendHeartbeat.mock.calls[0][0]).toBe("b2");
  });

  describe("cảnh báo trước khi rời trang", () => {
    function fireBeforeUnload() {
      const event = new Event("beforeunload", { cancelable: true });
      window.dispatchEvent(event);
      return event.defaultPrevented;
    }

    it("chặn khi còn nhãn sửa chưa xuất", () => {
      setup({ confirmLeave: true });
      expect(fireBeforeUnload()).toBe(true);
    });

    it("không chặn khi không có gì để mất", () => {
      setup({ confirmLeave: false });
      expect(fireBeforeUnload()).toBe(false);
    });

    it("gỡ cảnh báo khi người dùng đã xuất", () => {
      const { rerender, options } = setup({ confirmLeave: true });
      rerender({ ...options, confirmLeave: false });
      expect(fireBeforeUnload()).toBe(false);
    });
  });
});
