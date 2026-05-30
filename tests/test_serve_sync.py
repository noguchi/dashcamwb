import json
from pathlib import Path
from dcwb.serve.app import create_app

def _app(tmp_path):
    usb = tmp_path / "usb"; usb.mkdir()
    prof = tmp_path / "profiles"; prof.mkdir()
    out = tmp_path / "corrected"
    sync_dir = out / "sync" / "2026-05-27"; sync_dir.mkdir(parents=True)
    (sync_dir / "sync.json").write_text(json.dumps({
        "date": "2026-05-27", "delta_s": 53.0, "confidence": 0.10,
        "signal": "gps_yaw|gyro",
        "paths": {"insta_display": "x/insta-flat.mp4",
                  "tesla_concat": "x/tesla-concat.mp4",
                  "combined": "x/combined-2026-05-27.mp4"},
        "telemetry": [[0.0, 0.0, 0.0, "PARK"]],
    }))
    app = create_app(usb_root=usb, profiles_dir=prof, out_root=out,
                     pipeline_cfg={}, cache_root=tmp_path / "cache")
    return app, sync_dir

def test_sync_data_route_returns_manifest(tmp_path):
    app, _ = _app(tmp_path)
    r = app.test_client().get("/sync-data/2026-05-27")
    assert r.status_code == 200
    assert r.get_json()["delta_s"] == 53.0

def test_sync_nudge_updates_delta(tmp_path):
    app, sync_dir = _app(tmp_path)
    r = app.test_client().post("/sync-nudge/2026-05-27", json={"delta_s": 55.5})
    assert r.status_code == 200
    saved = json.loads((sync_dir / "sync.json").read_text())
    assert saved["delta_s"] == 55.5

def test_sync_nudge_missing_date_404(tmp_path):
    app, _ = _app(tmp_path)
    r = app.test_client().post("/sync-nudge/2099-01-01", json={"delta_s": 1.0})
    assert r.status_code == 404

def test_sync_player_page_renders(tmp_path):
    app, _ = _app(tmp_path)
    r = app.test_client().get("/sync/2026-05-27")
    assert r.status_code == 200
    assert b"<video" in r.data
