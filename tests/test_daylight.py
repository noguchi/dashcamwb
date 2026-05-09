from datetime import datetime, timezone, timedelta
from dcwb.daylight import is_daytime, TOKYO_LAT, TOKYO_LON

JST = timezone(timedelta(hours=9))

def test_noon_in_tokyo_is_daytime():
    t = datetime(2026, 5, 5, 12, 0, tzinfo=JST)
    assert is_daytime(t) is True

def test_midnight_in_tokyo_is_not_daytime():
    t = datetime(2026, 5, 5, 0, 0, tzinfo=JST)
    assert is_daytime(t) is False

def test_just_before_sunrise_is_not_daytime():
    # 5月5日東京の日の出は約 04:42。30分マージンで 05:12 までは not daytime
    t = datetime(2026, 5, 5, 5, 0, tzinfo=JST)
    assert is_daytime(t) is False

def test_well_after_sunrise_is_daytime():
    t = datetime(2026, 5, 5, 6, 0, tzinfo=JST)
    assert is_daytime(t) is True

def test_uses_provided_lat_lon():
    # Sydney (-33.86, 151.21) で 12:00 JST は現地で 13:00 → 昼
    t = datetime(2026, 5, 5, 12, 0, tzinfo=JST)
    assert is_daytime(t, lat=-33.86, lon=151.21) is True

def test_default_constants_are_tokyo():
    assert abs(TOKYO_LAT - 35.6762) < 0.001
    assert abs(TOKYO_LON - 139.6503) < 0.001
