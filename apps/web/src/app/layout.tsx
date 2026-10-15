import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EduPulse",
  description:
    "Phân loại cảm xúc và chủ đề góp ý của sinh viên bằng tiếng Việt, kèm giải thích và gợi ý duyệt.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
