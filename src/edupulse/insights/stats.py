"""Thống kê cho insight: khoảng tin cậy Wilson và từ khoá bằng log-odds có trọng số.

Chỉ dùng thư viện chuẩn (math): runtime không có scipy.
"""

import math
from collections import Counter
from collections.abc import Iterable, Mapping

Z_95 = 1.96  # phân vị chuẩn hai phía 95%
# Prior tối thiểu của một n-gram (như Jeffreys): n-gram chưa có hoặc rất hiếm ở train sẽ có prior
# rất nhỏ, khi đó phương sai 1/(y + α) của vế "không thấy" lớn tới mức từ mới (vd. "zoom") không
# bao giờ nổi bật dù lặp lại nhiều lần.
ALPHA_FLOOR = 0.5


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Khoảng tin cậy Wilson cho tỉ lệ `successes / n`; n = 0 thì chưa biết gì: (0, 1).

    Tốt hơn khoảng chuẩn khi n nhỏ hoặc tỉ lệ sát 0/1 (Brown, Cai, DasGupta 2001). Ví dụ 4/5 có cận
    dưới 0,38; 28/40 là 0,55; 70/100 là 0,60.
    """
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= successes <= n:
        raise ValueError(f"successes phải trong [0, n], nhận {successes}/{n}")
    p = successes / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - margin), min(1.0, centre + margin)


def wilson_lower(successes: int, n: int, z: float = Z_95) -> float:
    return wilson_interval(successes, n, z)[0]


def weighted_log_odds(
    group: Mapping[str, int],
    rest: Mapping[str, int],
    prior: Mapping[str, int],
    *,
    prior_total: int,
    alpha0: float,
    alpha_floor: float = ALPHA_FLOOR,
) -> dict[str, float]:
    """z-score của log-odds có trọng số với prior Dirichlet theo thông tin (Monroe et al. 2008).

    So tần suất từng n-gram trong `group` (vd. câu tiêu cực về cơ sở vật chất) với `rest` (mọi câu
    còn lại), có kéo về tần suất nền `prior` (đếm trên tập train, tổng `prior_total`) với độ mạnh
    `alpha0`. Từ chung ở mọi loại câu có z gần 0 nên không cần danh sách từ dừng; z dương lớn là
    đặc trưng của `group`. Prior của mỗi n-gram không thấp hơn `alpha_floor` (xem `ALPHA_FLOOR`).
    """
    n_group = sum(group.values())
    n_rest = sum(rest.values())
    if n_group == 0 or n_rest == 0 or prior_total <= 0 or alpha0 <= 0:
        return {}
    scores: dict[str, float] = {}
    for term in group:
        alpha = max(alpha0 * prior.get(term, 0) / prior_total, alpha_floor)
        in_group = group[term] + alpha
        in_rest = rest.get(term, 0) + alpha
        odds_group = math.log(in_group / (n_group + alpha0 - in_group))
        odds_rest = math.log(in_rest / (n_rest + alpha0 - in_rest))
        variance = 1.0 / in_group + 1.0 / in_rest
        scores[term] = (odds_group - odds_rest) / math.sqrt(variance)
    return scores


def document_frequency(documents: Iterable[Iterable[str]]) -> Counter[str]:
    """Số tài liệu (câu) chứa mỗi n-gram, mỗi câu tính một lần."""
    counts: Counter[str] = Counter()
    for grams in documents:
        counts.update(set(grams))
    return counts
