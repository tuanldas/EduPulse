import path from "node:path";

import { defineConfig } from "vitest/config";

// Test đơn vị và component chạy trong jsdom; Playwright (thư mục e2e/) có cấu hình riêng.
// JSX theo `jsx: "react-jsx"` trong tsconfig.json (runtime tự động, không cần import React).
export default defineConfig({
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "src") },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    css: false,
  },
});
