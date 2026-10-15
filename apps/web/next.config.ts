import type { NextConfig } from "next";

// Bản export tĩnh do FastAPI phục vụ cùng origin với /api, nên không có server Next lúc chạy.
// `trailingSlash` bắt buộc: thiếu nó thì /batch trả 404 vì StaticFiles chỉ tìm batch/index.html.
// Không dùng `rewrites`: với trailingSlash, Next chuyển /api/x thành /api/x/ và làm rơi thân POST.
// Lúc dev, trình duyệt gọi thẳng NEXT_PUBLIC_API_BASE (xem EDUPULSE_CORS_ORIGINS ở phía API).
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  // `next dev` mặc định tự ghi khối hướng dẫn vào AGENTS.md; tắt để không sinh file ngoài ý muốn.
  agentRules: false,
};

export default nextConfig;
