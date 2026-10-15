import type { Metadata } from "next";
import "./globals.css";
import { Geist } from "next/font/google";
import { cn } from "@/lib/utils";

const geist = Geist({subsets:['latin'],variable:'--font-sans'});

export const metadata: Metadata = {
  title: "EduPulse",
  description:
    "Phân loại cảm xúc và chủ đề góp ý của sinh viên bằng tiếng Việt, kèm giải thích và gợi ý duyệt.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi" className={cn("font-sans", geist.variable)}>
      <body>{children}</body>
    </html>
  );
}
