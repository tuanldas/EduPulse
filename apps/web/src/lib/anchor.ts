/**
 * Id của phần tử mà đoạn neo (`location.hash`) trỏ tới. Neo do người dùng gõ hoặc dán nên có thể hỏng
 * (vd. `#%E0%A4%A`): `decodeURIComponent` ném `URIError` với chuỗi phần trăm không hợp lệ, nên khi giải mã
 * lỗi thì dùng nguyên văn (không có phần tử nào khớp thì không cuộn đi đâu, trang vẫn hiện bình thường).
 */
export function anchorId(hash: string): string {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}
