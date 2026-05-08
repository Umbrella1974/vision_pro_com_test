"""
Standalone Apple Vision Pro multi-receiver test for avp_stream.

This script intentionally depends only on the Python standard library,
avp_stream, and numpy. It does not import any code from this repository.

Install dependencies:
    pip install avp_stream numpy

Dual receiver test:
    python test_avp_multi_receiver.py --ip 192.168.1.29 --duration 60

Single receiver test:
    python test_avp_multi_receiver.py --ip 192.168.1.29 --mode single --duration 60

Test with record enabled:
    python test_avp_multi_receiver.py --ip 192.168.1.29 --duration 60 --record

Notes:
- When --record is enabled, VisionProStreamer(record=True) is allowed to
  create recording files in the current working directory.
- poll_fps is the local polling loop frequency for reading vps.latest.
- valid_fps is the frequency of receiving non-empty dict payloads.
- changed_fps is the frequency at which at least one of right_wrist,
  left_wrist, right_fingers, or left_fingers changed numerically.
- If source frame metadata such as timestamp/frame_id/seq/counter exists,
  source_new_frame_fps is also reported.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import socket
import struct
import sys
import time
from typing import Any


KEY_FIELDS = (
    "right_wrist",
    "left_wrist",
    "right_fingers",
    "left_fingers",
)

SOURCE_FRAME_KEYS = (
    "timestamp",
    "time",
    "frame_id",
    "frame",
    "seq",
    "counter",
)

MISSING = object()
POLL_SLEEP_SECONDS = 0.002

VIEWER_MAGIC = b"VTLP"
VIEWER_HEADER_FMT = "<4s B B I d H"
VIEWER_MSG_HAND_RIGHT = 11
VIEWER_N_JOINTS = 21
VIEWER_N_DIMS = 3
VIEWER_N_FLOATS = VIEWER_N_JOINTS * VIEWER_N_DIMS
VIEWER_PAYLOAD_FMT = f"<{VIEWER_N_FLOATS}f"

# Vision Pro 25-point hand layout -> MediaPipe-like 21-point layout.
VP_TO_VIEWER_21 = [
    0,
    1, 2, 3, 4,
    6, 7, 8, 9,
    11, 12, 13, 14,
    16, 17, 18, 19,
    21, 22, 23, 24,
]

# ARKit -> viewer coordinates so fingers point up in the default xy projection.
ARKIT_TO_VIEWER = (
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


class DependencyError(RuntimeError):
    """Raised when required third-party dependencies are unavailable."""


def load_dependencies():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise DependencyError(
            "Missing required dependency 'numpy'.\n"
            "Install dependencies with:\n"
            "    pip install avp_stream numpy\n"
            f"Original error: {exc}"
        ) from exc

    try:
        from avp_stream import VisionProStreamer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise DependencyError(
            "Missing required dependency 'avp_stream'.\n"
            "Install dependencies with:\n"
            "    pip install avp_stream numpy\n"
            f"Original error: {exc}"
        ) from exc

    return np, VisionProStreamer


def format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def pack_viewer_right_hand_packet(joints: Any, frame_id: int, timestamp: float) -> bytes:
    flat = []
    for row in joints:
        flat.extend(float(v) for v in row)
    header = struct.pack(
        VIEWER_HEADER_FMT,
        VIEWER_MAGIC,
        1,
        VIEWER_MSG_HAND_RIGHT,
        frame_id & 0xFFFFFFFF,
        float(timestamp),
        VIEWER_N_FLOATS,
    )
    payload = struct.pack(VIEWER_PAYLOAD_FMT, *flat)
    return header + payload


def extract_finger_positions(np_module: Any, fingers_25x4x4: Any) -> Any:
    array = np_module.asarray(fingers_25x4x4, dtype=np_module.float64)
    if array.shape != (25, 4, 4):
        raise ValueError(f"right_fingers shape must be (25, 4, 4), got {array.shape}")
    return array[:, :3, 3].astype(np_module.float32)


def convert_positions_to_viewer(np_module: Any, positions_25x3: Any, wrist_4x4: Any = None) -> Any:
    positions = np_module.asarray(positions_25x3, dtype=np_module.float32)
    if positions.shape != (25, 3):
        raise ValueError(f"positions shape must be (25, 3), got {positions.shape}")

    joints = positions[VP_TO_VIEWER_21].copy()

    # avp_stream right_fingers appears to already be expressed in a wrist/palm-local
    # hand frame where point 0 is the hand root. Subtracting the absolute wrist world
    # position here would incorrectly deform the hand when the user translates the arm.
    # We therefore keep the hand in its local frame for viewer rendering.
    joints -= joints[0]

    transform = np_module.asarray(ARKIT_TO_VIEWER, dtype=np_module.float32)
    return (joints @ transform.T).astype(np_module.float32)


def safe_shape(np_module: Any, value: Any) -> Any:
    if value is MISSING:
        return None
    if value is None:
        return ()
    try:
        return tuple(np_module.asarray(value).shape)
    except Exception as exc:
        return f"shape_error:{type(exc).__name__}"


def make_field_state(np_module: Any, value: Any) -> dict[str, Any]:
    if value is MISSING:
        return {"kind": "missing", "value": None}
    if value is None:
        return {"kind": "none", "value": None}

    try:
        array = np_module.asarray(value)
    except Exception as exc:
        return {"kind": "repr", "value": f"<convert_error:{type(exc).__name__}:{exc}>"}

    return {
        "kind": "array",
        "shape": tuple(array.shape),
        "array": array.copy(),
    }


def field_states_equal(np_module: Any, previous: dict[str, Any], current: dict[str, Any]) -> bool:
    if previous["kind"] != current["kind"]:
        return False

    kind = previous["kind"]
    if kind == "array":
        if previous["shape"] != current["shape"]:
            return False
        return bool(
            np_module.allclose(previous["array"], current["array"], equal_nan=True)
        )

    return previous["value"] == current["value"]


def extract_source_frame_info(np_module: Any, data: dict[str, Any]) -> tuple[Any, Any]:
    for key in SOURCE_FRAME_KEYS:
        if key not in data:
            continue

        value = data[key]
        try:
            array = np_module.asarray(value)
            if array.shape == ():
                normalized = array.item()
            else:
                normalized = repr(array.tolist())
        except Exception:
            normalized = repr(value)
        return key, normalized

    return None, None


def put_status(status_queue: Any, status: dict[str, Any]) -> None:
    try:
        status_queue.put_nowait(status)
    except queue.Full:
        pass


def build_status(
    receiver_id: str,
    start_monotonic: float,
    poll_count: int,
    valid_count: int,
    changed_count: int,
    unchanged_count: int,
    last_keys: list[str],
    right_wrist_exists: bool,
    left_wrist_exists: bool,
    right_fingers_shape: Any,
    left_fingers_shape: Any,
    local_receive_time: Any,
    last_error: Any,
    invalid_reason: Any,
    source_frame_key: Any,
    source_frame_value: Any,
    source_new_frame_count: int,
    source_unchanged_count: int,
) -> dict[str, Any]:
    elapsed = max(time.monotonic() - start_monotonic, 0.0)
    return {
        "receiver_id": receiver_id,
        "elapsed_time": elapsed,
        "poll_count": poll_count,
        "poll_fps": (poll_count / elapsed) if elapsed > 0.0 else 0.0,
        "valid_count": valid_count,
        "valid_fps": (valid_count / elapsed) if elapsed > 0.0 else 0.0,
        "changed_count": changed_count,
        "changed_fps": (changed_count / elapsed) if elapsed > 0.0 else 0.0,
        "data_keys": last_keys,
        "right_wrist_exists": right_wrist_exists,
        "left_wrist_exists": left_wrist_exists,
        "right_fingers_shape": right_fingers_shape,
        "left_fingers_shape": left_fingers_shape,
        "last_error": last_error,
        "invalid_reason": invalid_reason,
        "unchanged_count": unchanged_count,
        "local_receive_time": local_receive_time,
        "source_frame_key": source_frame_key,
        "source_frame_value": source_frame_value,
        "source_new_frame_count": source_new_frame_count,
        "source_new_frame_fps": (source_new_frame_count / elapsed) if elapsed > 0.0 else 0.0,
        "source_unchanged_count": source_unchanged_count,
        "status_sent_time": time.time(),
    }


def receiver_worker(
    receiver_id: str,
    ip: str,
    record: bool,
    hz: float,
    viewer_config: dict[str, Any],
    stop_event: Any,
    status_queue: Any,
) -> None:
    try:
        np_module, VisionProStreamer = load_dependencies()
    except DependencyError as exc:
        start_monotonic = time.monotonic()
        put_status(
            status_queue,
            build_status(
                receiver_id=receiver_id,
                start_monotonic=start_monotonic,
                poll_count=0,
                valid_count=0,
                changed_count=0,
                unchanged_count=0,
                last_keys=[],
                right_wrist_exists=False,
                left_wrist_exists=False,
                right_fingers_shape=None,
                left_fingers_shape=None,
                local_receive_time=None,
                last_error=str(exc),
                invalid_reason="dependency_load_failed",
                source_frame_key=None,
                source_frame_value=None,
                source_new_frame_count=0,
                source_unchanged_count=0,
            ),
        )
        return

    report_interval = max(0.2, 1.0 / max(hz, 1e-6))
    start_monotonic = time.monotonic()
    next_report_time = start_monotonic

    poll_count = 0
    valid_count = 0
    changed_count = 0
    unchanged_count = 0
    source_new_frame_count = 0
    source_unchanged_count = 0

    last_keys: list[str] = []
    right_wrist_exists = False
    left_wrist_exists = False
    right_fingers_shape = None
    left_fingers_shape = None
    local_receive_time = None
    last_error = None
    invalid_reason = None

    previous_states: dict[str, dict[str, Any]] | None = None
    previous_source_key = None
    previous_source_value = None
    vps = None
    viewer_sock = None
    viewer_frame_id = 0
    viewer_last_send_monotonic = 0.0

    viewer_enabled = bool(viewer_config.get("enabled")) and (
        viewer_config.get("receiver_id") == receiver_id
    )
    viewer_ip = viewer_config.get("ip", "127.0.0.1")
    viewer_port = int(viewer_config.get("port", 5005))
    viewer_rate = float(viewer_config.get("rate", 30.0))
    viewer_period = 1.0 / max(viewer_rate, 1e-6)

    if viewer_enabled:
        viewer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        while not stop_event.is_set():
            if vps is None:
                try:
                    vps = VisionProStreamer(ip=ip, record=record)
                    last_error = None
                    invalid_reason = None
                except Exception as exc:
                    last_error = format_exception(exc)
                    invalid_reason = "streamer_init_failed"
                    now = time.monotonic()
                    if now >= next_report_time:
                        put_status(
                            status_queue,
                            build_status(
                                receiver_id=receiver_id,
                                start_monotonic=start_monotonic,
                                poll_count=poll_count,
                                valid_count=valid_count,
                                changed_count=changed_count,
                                unchanged_count=unchanged_count,
                                last_keys=last_keys,
                                right_wrist_exists=right_wrist_exists,
                                left_wrist_exists=left_wrist_exists,
                                right_fingers_shape=right_fingers_shape,
                                left_fingers_shape=left_fingers_shape,
                                local_receive_time=local_receive_time,
                                last_error=last_error,
                                invalid_reason=invalid_reason,
                                source_frame_key=previous_source_key,
                                source_frame_value=previous_source_value,
                                source_new_frame_count=source_new_frame_count,
                                source_unchanged_count=source_unchanged_count,
                            ),
                        )
                        next_report_time = now + report_interval
                    if stop_event.wait(1.0):
                        break
                    continue

            poll_count += 1
            try:
                data = vps.latest
                last_error = None
            except Exception as exc:
                last_error = format_exception(exc)
                invalid_reason = "read_latest_failed"
                data = None

            local_receive_time = time.time()
            now = time.monotonic()

            last_keys = []
            right_wrist_exists = False
            left_wrist_exists = False
            right_fingers_shape = None
            left_fingers_shape = None

            if data is None:
                invalid_reason = invalid_reason or "data_is_none"
            elif not isinstance(data, dict):
                invalid_reason = f"data_is_{type(data).__name__}_not_dict"
            elif not data:
                invalid_reason = "data_is_empty_dict"
            else:
                invalid_reason = None
                valid_count += 1
                last_keys = sorted(str(key) for key in data.keys())
                right_wrist_exists = "right_wrist" in data
                left_wrist_exists = "left_wrist" in data
                right_fingers_shape = safe_shape(np_module, data.get("right_fingers", MISSING))
                left_fingers_shape = safe_shape(np_module, data.get("left_fingers", MISSING))

                current_states = {
                    field: make_field_state(np_module, data.get(field, MISSING))
                    for field in KEY_FIELDS
                }

                any_changed = previous_states is None or any(
                    not field_states_equal(np_module, previous_states[field], current_states[field])
                    for field in KEY_FIELDS
                )

                if any_changed:
                    changed_count += 1
                    unchanged_count = 0
                else:
                    unchanged_count += 1

                previous_states = current_states

                source_frame_key, source_frame_value = extract_source_frame_info(np_module, data)
                if source_frame_key is not None:
                    if (
                        previous_source_key != source_frame_key
                        or previous_source_value != source_frame_value
                    ):
                        source_new_frame_count += 1
                        source_unchanged_count = 0
                    else:
                        source_unchanged_count += 1
                    previous_source_key = source_frame_key
                    previous_source_value = source_frame_value
                else:
                    previous_source_key = None
                    previous_source_value = None
                    source_unchanged_count = 0

                if viewer_enabled and right_wrist_exists and "right_fingers" in data:
                    if now - viewer_last_send_monotonic >= viewer_period:
                        try:
                            positions = extract_finger_positions(np_module, data["right_fingers"])
                            wrist = data.get("right_wrist")
                            joints = convert_positions_to_viewer(np_module, positions, wrist)
                            packet = pack_viewer_right_hand_packet(
                                joints=joints,
                                frame_id=viewer_frame_id,
                                timestamp=local_receive_time,
                            )
                            viewer_sock.sendto(packet, (viewer_ip, viewer_port))
                            viewer_frame_id = (viewer_frame_id + 1) & 0xFFFFFFFF
                            viewer_last_send_monotonic = now
                        except Exception as exc:
                            last_error = f"viewer_forward_failed: {format_exception(exc)}"

            if now >= next_report_time:
                put_status(
                    status_queue,
                    build_status(
                        receiver_id=receiver_id,
                        start_monotonic=start_monotonic,
                        poll_count=poll_count,
                        valid_count=valid_count,
                        changed_count=changed_count,
                        unchanged_count=unchanged_count,
                        last_keys=last_keys,
                        right_wrist_exists=right_wrist_exists,
                        left_wrist_exists=left_wrist_exists,
                        right_fingers_shape=right_fingers_shape,
                        left_fingers_shape=left_fingers_shape,
                        local_receive_time=local_receive_time,
                        last_error=last_error,
                        invalid_reason=invalid_reason,
                        source_frame_key=previous_source_key,
                        source_frame_value=previous_source_value,
                        source_new_frame_count=source_new_frame_count,
                        source_unchanged_count=source_unchanged_count,
                    ),
                )
                next_report_time = now + report_interval

            if stop_event.wait(POLL_SLEEP_SECONDS):
                break
    finally:
        if viewer_sock is not None:
            viewer_sock.close()

        put_status(
            status_queue,
            build_status(
                receiver_id=receiver_id,
                start_monotonic=start_monotonic,
                poll_count=poll_count,
                valid_count=valid_count,
                changed_count=changed_count,
                unchanged_count=unchanged_count,
                last_keys=last_keys,
                right_wrist_exists=right_wrist_exists,
                left_wrist_exists=left_wrist_exists,
                right_fingers_shape=right_fingers_shape,
                left_fingers_shape=left_fingers_shape,
                local_receive_time=local_receive_time,
                last_error=last_error,
                invalid_reason=invalid_reason,
                source_frame_key=previous_source_key,
                source_frame_value=previous_source_value,
                source_new_frame_count=source_new_frame_count,
                source_unchanged_count=source_unchanged_count,
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether avp_stream supports one or two concurrent receivers."
    )
    parser.add_argument("--ip", required=True, help="Vision Pro IP address, e.g. 192.168.1.29")
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Test duration in seconds. Default: 60",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=5.0,
        help="Main-process print frequency in Hz. Default: 5",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Pass record=True to VisionProStreamer.",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "dual"),
        default="dual",
        help="single starts one receiver, dual starts two. Default: dual",
    )
    parser.add_argument(
        "--forward-viewer",
        action="store_true",
        help="Forward one receiver's right-hand data to the hand-ex-render UDP viewer.",
    )
    parser.add_argument(
        "--viewer-ip",
        default="127.0.0.1",
        help="UDP target IP for viewer forwarding. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=5005,
        help="UDP target port for viewer forwarding. Default: 5005",
    )
    parser.add_argument(
        "--viewer-rate",
        type=float,
        default=30.0,
        help="Viewer forwarding packet rate in Hz. Default: 30",
    )
    parser.add_argument(
        "--viewer-receiver",
        choices=("A", "B"),
        default="A",
        help="Which receiver forwards right-hand data to the viewer. Default: A",
    )
    return parser.parse_args()


def drain_status_queue(status_queue: Any, latest_status: dict[str, dict[str, Any]]) -> None:
    while True:
        try:
            status = status_queue.get_nowait()
        except queue.Empty:
            break
        status["_parent_received_monotonic"] = time.monotonic()
        latest_status[status["receiver_id"]] = status


def format_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_shape(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)


def print_status_block(
    receiver_ids: list[str],
    latest_status: dict[str, dict[str, Any]],
    overall_elapsed: float,
    hz: float,
) -> None:
    report_timeout = max(2.0, 3.0 / max(hz, 1e-6))
    wall_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"\n=== status {wall_time} | overall_elapsed={overall_elapsed:.2f}s ===")

    for receiver_id in receiver_ids:
        status = latest_status.get(receiver_id)
        if status is None:
            print(f"{receiver_id} | waiting_for_first_status=yes")
            print(f"    WARNING: receiver {receiver_id} has not reported recently")
            continue

        age = time.monotonic() - status["_parent_received_monotonic"]
        recently_reported = age <= report_timeout
        line_1 = (
            f"{receiver_id} | reported_recently={recently_reported}"
            f" | elapsed={format_float(status['elapsed_time'])}s"
            f" | poll_count={status['poll_count']}"
            f" | poll_fps={format_float(status['poll_fps'])}"
            f" | valid_count={status['valid_count']}"
            f" | valid_fps={format_float(status['valid_fps'])}"
            f" | changed_count={status['changed_count']}"
            f" | changed_fps={format_float(status['changed_fps'])}"
            f" | unchanged_count={status['unchanged_count']}"
        )
        print(line_1)

        source_bits = "source_frame=None"
        if status["source_frame_key"] is not None:
            source_bits = (
                f"source_frame={status['source_frame_key']}={status['source_frame_value']}"
                f" | source_new_frame_count={status['source_new_frame_count']}"
                f" | source_new_frame_fps={format_float(status['source_new_frame_fps'])}"
                f" | source_unchanged_count={status['source_unchanged_count']}"
            )

        line_2 = (
            f"    keys={status['data_keys']}"
            f" | right_wrist_exists={status['right_wrist_exists']}"
            f" | left_wrist_exists={status['left_wrist_exists']}"
            f" | right_fingers_shape={format_shape(status['right_fingers_shape'])}"
            f" | left_fingers_shape={format_shape(status['left_fingers_shape'])}"
        )
        print(line_2)
        print(
            f"    local_receive_time={format_float(status['local_receive_time'], digits=6)}"
            f" | invalid_reason={status['invalid_reason']}"
            f" | last_error={status['last_error']}"
            f" | {source_bits}"
        )

        if not recently_reported:
            print(f"    WARNING: receiver {receiver_id} has not reported recently")

        if status["source_frame_key"] is not None and status["source_unchanged_count"] > 0:
            print("    WARNING: source frame not advancing")
        elif status["unchanged_count"] > 0:
            print("    WARNING: data unchanged across reads")


def stop_processes(processes: list[mp.Process], stop_event: Any) -> None:
    stop_event.set()
    for process in processes:
        process.join(timeout=5.0)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)


def main() -> int:
    args = parse_args()

    if args.duration <= 0:
        print("--duration must be > 0", file=sys.stderr)
        return 2
    if args.hz <= 0:
        print("--hz must be > 0", file=sys.stderr)
        return 2
    if args.viewer_rate <= 0:
        print("--viewer-rate must be > 0", file=sys.stderr)
        return 2
    if not (0 <= args.viewer_port <= 65535):
        print("--viewer-port must be between 0 and 65535", file=sys.stderr)
        return 2

    try:
        load_dependencies()
    except DependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    status_queue = ctx.Queue()
    receiver_ids = ["A"] if args.mode == "single" else ["A", "B"]
    viewer_config = {
        "enabled": args.forward_viewer,
        "ip": args.viewer_ip,
        "port": args.viewer_port,
        "rate": args.viewer_rate,
        "receiver_id": args.viewer_receiver,
    }

    if args.forward_viewer and args.viewer_receiver not in receiver_ids:
        print(
            f"--viewer-receiver {args.viewer_receiver} is not active in --mode {args.mode}",
            file=sys.stderr,
        )
        return 2

    processes: list[mp.Process] = []
    for receiver_id in receiver_ids:
        process = ctx.Process(
            target=receiver_worker,
            args=(
                receiver_id,
                args.ip,
                args.record,
                args.hz,
                viewer_config,
                stop_event,
                status_queue,
            ),
            name=f"avp_receiver_{receiver_id}",
        )
        process.start()
        processes.append(process)

    latest_status: dict[str, dict[str, Any]] = {}
    start_monotonic = time.monotonic()
    deadline = start_monotonic + args.duration
    next_print_time = start_monotonic
    interrupted = False

    try:
        while time.monotonic() < deadline:
            drain_status_queue(status_queue, latest_status)
            now = time.monotonic()
            if now >= next_print_time:
                print_status_block(
                    receiver_ids=receiver_ids,
                    latest_status=latest_status,
                    overall_elapsed=now - start_monotonic,
                    hz=args.hz,
                )
                next_print_time = now + (1.0 / args.hz)
            time.sleep(0.05)
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received. Stopping receivers...")
    finally:
        stop_processes(processes, stop_event)
        drain_status_queue(status_queue, latest_status)
        print_status_block(
            receiver_ids=receiver_ids,
            latest_status=latest_status,
            overall_elapsed=time.monotonic() - start_monotonic,
            hz=args.hz,
        )

    return 130 if interrupted else 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
