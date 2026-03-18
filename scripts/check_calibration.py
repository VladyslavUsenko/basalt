#!/usr/bin/env python3
#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CAM_TIME_OFFSET_LIMIT_NS = 1_000_000
MOCAP_TIME_OFFSET_LIMIT_NS = 1_000_000

INTRINSIC_PIXEL_LIMIT = 1.0
INTRINSIC_SHAPE_LIMIT = 0.02
EXTRINSIC_TRANSLATION_LIMIT_M = 0.01
EXTRINSIC_ROTATION_LIMIT_DEG = 1.0

ACCEL_BIAS_LIMIT = 0.01
GYRO_BIAS_LIMIT = 0.001
EUROC_ACCEL_BIAS_LIMIT = 0.02
EUROC_GYRO_BIAS_LIMIT = 0.005
ACCEL_SCALE_LIMIT = 0.01
GYRO_SCALE_LIMIT = 0.01

IDENTITY_TRANSLATION_LIMIT_M = 0.01
IDENTITY_ROTATION_LIMIT_DEG = 1.0


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def vec_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


def quat_angle_deg(a: Dict[str, float], b: Dict[str, float]) -> float:
    dot = abs(
        a["qx"] * b["qx"]
        + a["qy"] * b["qy"]
        + a["qz"] * b["qz"]
        + a["qw"] * b["qw"]
    )
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.degrees(math.acos(dot))


def trans_delta(a: Dict[str, float], b: Dict[str, float]) -> float:
    return vec_norm(
        (
            a["px"] - b["px"],
            a["py"] - b["py"],
            a["pz"] - b["pz"],
        )
    )


def compare_calibration(
    generated_path: Path, reference_path: Path, label: str
) -> Tuple[List[str], List[str]]:
    generated = load_json(generated_path)["value0"]
    reference = load_json(reference_path)["value0"]

    summary: List[str] = [f"[{label}]"]
    failures: List[str] = []

    for cam_idx, (gen_intr, ref_intr) in enumerate(
        zip(generated["intrinsics"], reference["intrinsics"])
    ):
        gen_params = gen_intr["intrinsics"]
        ref_params = ref_intr["intrinsics"]
        for key, gen_value in gen_params.items():
            delta = abs(float(gen_value) - float(ref_params[key]))
            limit = (
                INTRINSIC_PIXEL_LIMIT
                if key in {"fx", "fy", "cx", "cy"}
                else INTRINSIC_SHAPE_LIMIT
            )
            summary.append(
                f"  cam{cam_idx} {key}: delta={delta:.6g} limit={limit:.6g}"
            )
            if delta > limit:
                failures.append(
                    f"{label}: cam{cam_idx} intrinsics {key} delta {delta:.6g} exceeds {limit:.6g}"
                )

    for cam_idx, (gen_pose, ref_pose) in enumerate(
        zip(generated["T_imu_cam"], reference["T_imu_cam"])
    ):
        trans = trans_delta(gen_pose, ref_pose)
        rot = quat_angle_deg(gen_pose, ref_pose)
        summary.append(
            f"  cam{cam_idx} T_imu_cam: translation={trans:.6g}m rotation={rot:.6g}deg"
        )
        if trans > EXTRINSIC_TRANSLATION_LIMIT_M:
            failures.append(
                f"{label}: cam{cam_idx} T_imu_cam translation {trans:.6g} exceeds {EXTRINSIC_TRANSLATION_LIMIT_M:.6g}"
            )
        if rot > EXTRINSIC_ROTATION_LIMIT_DEG:
            failures.append(
                f"{label}: cam{cam_idx} T_imu_cam rotation {rot:.6g} exceeds {EXTRINSIC_ROTATION_LIMIT_DEG:.6g}"
            )

    accel_bias = vec_norm(
        g - r for g, r in zip(generated["calib_accel_bias"][:3], reference["calib_accel_bias"][:3])
    )
    gyro_bias = vec_norm(
        g - r for g, r in zip(generated["calib_gyro_bias"][:3], reference["calib_gyro_bias"][:3])
    )
    accel_scale = vec_norm(generated["calib_accel_bias"][3:9])
    gyro_scale = vec_norm(generated["calib_gyro_bias"][3:12])
    cam_time_offset = abs(int(generated["cam_time_offset_ns"]))
    accel_bias_limit = (
        EUROC_ACCEL_BIAS_LIMIT if label.startswith("euroc_") else ACCEL_BIAS_LIMIT
    )
    gyro_bias_limit = (
        EUROC_GYRO_BIAS_LIMIT if label.startswith("euroc_") else GYRO_BIAS_LIMIT
    )

    summary.extend(
        [
            f"  accel_bias_delta={accel_bias:.6g}",
            f"  gyro_bias_delta={gyro_bias:.6g}",
            f"  accel_scale_delta={accel_scale:.6g}",
            f"  gyro_scale_delta={gyro_scale:.6g}",
            f"  cam_time_offset_ns={cam_time_offset}",
            f"  accel_bias_limit={accel_bias_limit:.6g}",
            f"  gyro_bias_limit={gyro_bias_limit:.6g}",
        ]
    )

    if accel_bias > accel_bias_limit:
        failures.append(
            f"{label}: accel bias delta {accel_bias:.6g} exceeds {accel_bias_limit:.6g}"
        )
    if gyro_bias > gyro_bias_limit:
        failures.append(
            f"{label}: gyro bias delta {gyro_bias:.6g} exceeds {gyro_bias_limit:.6g}"
        )
    if accel_scale > ACCEL_SCALE_LIMIT:
        failures.append(
            f"{label}: accel scale delta {accel_scale:.6g} exceeds {ACCEL_SCALE_LIMIT:.6g}"
        )
    if gyro_scale > GYRO_SCALE_LIMIT:
        failures.append(
            f"{label}: gyro scale delta {gyro_scale:.6g} exceeds {GYRO_SCALE_LIMIT:.6g}"
        )
    if cam_time_offset >= CAM_TIME_OFFSET_LIMIT_NS:
        failures.append(
            f"{label}: cam_time_offset_ns {cam_time_offset} exceeds {CAM_TIME_OFFSET_LIMIT_NS}"
        )

    return summary, failures


def compare_mocap_identity(mocap_path: Path, label: str) -> Tuple[List[str], List[str]]:
    mocap = load_json(mocap_path)["value0"]
    identity = {
        "px": 0.0,
        "py": 0.0,
        "pz": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }

    trans = trans_delta(mocap["T_imu_marker"], identity)
    rot = quat_angle_deg(mocap["T_imu_marker"], identity)
    time_offset = abs(int(mocap["mocap_time_offset_ns"]))

    summary = [
        f"[{label} mocap]",
        f"  T_imu_marker translation={trans:.6g}m",
        f"  T_imu_marker rotation={rot:.6g}deg",
        f"  mocap_time_offset_ns={time_offset}",
    ]
    failures: List[str] = []

    if trans > IDENTITY_TRANSLATION_LIMIT_M:
        failures.append(
            f"{label}: T_imu_marker translation {trans:.6g} exceeds {IDENTITY_TRANSLATION_LIMIT_M:.6g}"
        )
    if rot > IDENTITY_ROTATION_LIMIT_DEG:
        failures.append(
            f"{label}: T_imu_marker rotation {rot:.6g} exceeds {IDENTITY_ROTATION_LIMIT_DEG:.6g}"
        )
    if time_offset >= MOCAP_TIME_OFFSET_LIMIT_NS:
        failures.append(
            f"{label}: mocap_time_offset_ns {time_offset} exceeds {MOCAP_TIME_OFFSET_LIMIT_NS}"
        )

    return summary, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated calibration against reference data.")
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mocap", type=Path)
    parser.add_argument("--check-mocap-identity", action="store_true")
    args = parser.parse_args()

    summary, failures = compare_calibration(args.generated, args.reference, args.label)

    if args.check_mocap_identity:
        if not args.mocap:
            raise SystemExit("--mocap is required when --check-mocap-identity is set")
        mocap_summary, mocap_failures = compare_mocap_identity(args.mocap, args.label)
        summary.extend(mocap_summary)
        failures.extend(mocap_failures)

    print("\n".join(summary))
    if failures:
      print("\nFAILURES:")
      for failure in failures:
          print(f"  - {failure}")
      return 1

    print(f"\n{args.label}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
