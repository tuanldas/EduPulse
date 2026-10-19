import { IllustrativeBadge } from "@/components/shared/illustrative-badge";
import { cn } from "@/lib/utils";

/**
 * `IllustrativeBadge` cho những chỗ hẹp (thẻ cảnh báo ba cột, màn hình nhỏ): nhãn dài khoảng 310px và không
 * xuống dòng, nên đặt thẳng vào khung hẹp thì bị cắt hoặc làm cả trang cuộn ngang. Ở đây nhãn được co lại
 * theo khung và tự xuống dòng.
 */
export function WrappingIllustrativeBadge({ className }: { className?: string }) {
  return (
    <IllustrativeBadge
      className={cn("max-w-full shrink justify-start text-left whitespace-normal [&>svg]:shrink-0", className)}
    />
  );
}
