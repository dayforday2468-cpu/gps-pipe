from datetime import datetime
from zoneinfo import ZoneInfo


UTC = ZoneInfo("UTC")
KST = ZoneInfo("Asia/Seoul")


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")

    return dt.astimezone(UTC)


def to_kst(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")

    return dt.astimezone(KST)