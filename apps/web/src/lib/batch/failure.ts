import { ApiFailure } from "@/lib/api/errors";

export { isAbortError } from "@/lib/api/errors";

/** 404: batch không còn (hết hạn, đã đóng hoặc id sai). */
export function isNotFound(error: unknown): boolean {
  return error instanceof ApiFailure && error.status === 404;
}
