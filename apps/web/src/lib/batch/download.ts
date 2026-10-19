/** Tải một `Blob` về máy bằng thẻ `<a download>`; không cần máy chủ đặt tên file. */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.rel = "noopener";
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
  // Thu hồi muộn: trình duyệt cần thời gian bắt đầu tải trước khi địa chỉ tạm bị huỷ.
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** Phần tên file (không đuôi), chỉ giữ chữ, số, `.`, `_`, `-` để an toàn trên mọi hệ điều hành. */
export function fileStem(filename: string, fallback = "ket-qua"): string {
  const withoutExtension = filename.replace(/\.[^./\\]+$/, "");
  const cleaned = withoutExtension
    .normalize("NFC")
    .replace(/[^\p{L}\p{N}._-]+/gu, "-")
    .replace(/^[-.]+|[-.]+$/g, "")
    .slice(0, 60);
  return cleaned === "" ? fallback : cleaned;
}
