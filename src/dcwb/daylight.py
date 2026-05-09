from datetime import datetime, timedelta
from astral import LocationInfo
from astral.sun import sun

TOKYO_LAT = 35.6762
TOKYO_LON = 139.6503
MARGIN = timedelta(minutes=30)

def is_daytime(
    when: datetime,
    lat: float = TOKYO_LAT,
    lon: float = TOKYO_LON,
) -> bool:
    """Return True if `when` is between sunrise+30min and sunset-30min."""
    if when.tzinfo is None:
        raise ValueError("`when` must be timezone-aware")
    loc = LocationInfo("custom", "custom", str(when.tzinfo), lat, lon)
    s = sun(loc.observer, date=when.date(), tzinfo=when.tzinfo)
    return s["sunrise"] + MARGIN <= when <= s["sunset"] - MARGIN
