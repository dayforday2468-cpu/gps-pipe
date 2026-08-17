from datetime import datetime
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
KST = ZoneInfo("Asia/Seoul")


def kst_to_utc(dt: datetime) -> datetime:
    dt_kst = dt.replace(tzinfo=KST)
    return dt_kst.astimezone(UTC)


def utc_to_kst(dt: datetime) -> datetime:
    dt_utc = dt.replace(tzinfo=UTC)
    return dt_utc.astimezone(KST)


def utc_to_kst_range(
    start: datetime,
    end: datetime,
) -> tuple[datetime, datetime]:
    if start >= end:
        raise ValueError("start must be earlier than end")

    return utc_to_kst(start), utc_to_kst(end)


def kst_to_utc_range(
    start: datetime,
    end: datetime,
) -> tuple[datetime, datetime]:
    if start >= end:
        raise ValueError("start must be earlier than end")

    return kst_to_utc(start), kst_to_utc(end)