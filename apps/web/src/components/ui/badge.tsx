import { mergeProps } from "@base-ui/react/merge-props"
import { useRender } from "@base-ui/react/use-render"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "cn"

import { TONE_CLASS } from "@/lib/colors"

// Badge theo wireframe (.bd): bo 6px, chữ 14px/500, cặp chữ-nền theo khoá màu ngữ nghĩa. Màu luôn đi
// kèm icon + chữ ở chỗ dùng (docs/design-guidelines.md mục 1.3).
const badgeVariants = cva(
  "group/badge inline-flex w-fit shrink-0 items-center justify-center gap-1 overflow-hidden rounded-md border border-transparent px-2 py-px text-sm leading-[1.6] font-medium whitespace-nowrap transition-colors focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/70 [&>svg]:pointer-events-none [&>svg]:size-4",
  {
    variants: {
      variant: {
        default: "bg-primary-soft text-primary-text",
        gray: TONE_CLASS.neu.badge,
        secondary: "bg-secondary text-secondary-foreground",
        outline: "border-border text-foreground",
        neg: TONE_CLASS.neg.badge,
        neu: TONE_CLASS.neu.badge,
        pos: TONE_CLASS.pos.badge,
        gv: TONE_CLASS.gv.badge,
        ct: TONE_CLASS.ct.badge,
        csvc: TONE_CLASS.csvc.badge,
        khac: TONE_CLASS.khac.badge,
        flag: TONE_CLASS.flag.badge,
      },
      size: {
        default: "",
        lg: "px-2.5 py-[3px] text-[15px]",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Badge({
  className,
  variant = "default",
  size = "default",
  render,
  ...props
}: useRender.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return useRender({
    defaultTagName: "span",
    props: mergeProps<"span">(
      {
        className: cn(badgeVariants({ variant, size }), className),
      },
      props
    ),
    render,
    state: {
      slot: "badge",
      variant,
    },
  })
}

export { Badge, badgeVariants }
