// Tính tương phản WCAG 2.1 giữa hai màu (dùng chung cho các test tương phản).
export type Rgba = [number, number, number, number];

export function parseColor(value: string): Rgba {
  const text = value.trim();
  if (text.startsWith("#")) {
    return [
      parseInt(text.slice(1, 3), 16),
      parseInt(text.slice(3, 5), 16),
      parseInt(text.slice(5, 7), 16),
      1,
    ];
  }
  const match = /^rgba\(([^)]+)\)$/.exec(text);
  if (!match) throw new Error(`không đọc được màu: ${value}`);
  const [r, g, b, a] = match[1].split(",").map(Number);
  return [r, g, b, a];
}

/** Màu `fg` có độ trong suốt đặt lên nền đặc `bg`. */
export function over(fg: Rgba, bg: Rgba): Rgba {
  const [r, g, b, a] = fg;
  return [a * r + (1 - a) * bg[0], a * g + (1 - a) * bg[1], a * b + (1 - a) * bg[2], 1];
}

function luminance([r, g, b]: Rgba): number {
  const channel = (x: number) => {
    const c = x / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

export function contrast(a: Rgba, b: Rgba): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}
