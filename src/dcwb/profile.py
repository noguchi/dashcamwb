from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from dcwb.matrix import Matrix3x3, from_diag

@dataclass
class CalibrationMeta:
    samples_used: int
    events_sampled: int
    method: str
    calibrated_at: datetime
    samples_per_event_max: int

@dataclass
class Profile:
    camera: str
    gain_r: float
    gain_g: float
    gain_b: float
    matrix_3x3: Matrix3x3
    calibration: CalibrationMeta

    @classmethod
    def from_white_point(
        cls, camera: str, rgb_white: np.ndarray, meta: CalibrationMeta
    ) -> "Profile":
        rw, gw, bw = float(rgb_white[0]), float(rgb_white[1]), float(rgb_white[2])
        gain_r = gw / rw
        gain_g = 1.0
        gain_b = gw / bw
        return cls(
            camera=camera,
            gain_r=gain_r,
            gain_g=gain_g,
            gain_b=gain_b,
            matrix_3x3=from_diag(gain_r, gain_g, gain_b),
            calibration=meta,
        )

    def to_json(self, path: Path) -> None:
        d = {
            "camera": self.camera,
            "gain_r": self.gain_r,
            "gain_g": self.gain_g,
            "gain_b": self.gain_b,
            "matrix_3x3": self.matrix_3x3.tolist(),
            "calibration": {
                "samples_used": self.calibration.samples_used,
                "events_sampled": self.calibration.events_sampled,
                "method": self.calibration.method,
                "calibrated_at": self.calibration.calibrated_at.isoformat(),
                "samples_per_event_max": self.calibration.samples_per_event_max,
            },
        }
        path.write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "Profile":
        d = json.loads(path.read_text())
        cal = d["calibration"]
        return cls(
            camera=d["camera"],
            gain_r=float(d["gain_r"]),
            gain_g=float(d["gain_g"]),
            gain_b=float(d["gain_b"]),
            matrix_3x3=np.array(d["matrix_3x3"], dtype=np.float64),
            calibration=CalibrationMeta(
                samples_used=int(cal["samples_used"]),
                events_sampled=int(cal["events_sampled"]),
                method=cal["method"],
                calibrated_at=datetime.fromisoformat(cal["calibrated_at"]),
                samples_per_event_max=int(cal["samples_per_event_max"]),
            ),
        )
