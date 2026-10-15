"""Quy ước đường dẫn: endpoint JSON nằm dưới `/api`, phần còn lại là bản web export tĩnh."""

API_PREFIX = "/api"


def is_api_path(path: str) -> bool:
    return path == API_PREFIX or path.startswith(API_PREFIX + "/")
