import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Gỡ cây React sau mỗi test để các test không dính DOM của nhau.
afterEach(() => {
  cleanup();
});
