# Vision Pro Multi-Receiver Test

Standalone Python tool for testing whether `avp_stream` can serve one or two
independent Apple Vision Pro receivers at the same time.

This script can also forward the right-hand skeleton to a UDP viewer in
real time, which is useful when you want diagnostics and visualization in the
same run.

## Features

- No dependency on the original `VisionPro_Teleop-main` project code
- Works from any directory as a single Python file
- Supports `single` and `dual` receiver modes
- Reports polling rate, valid data rate, changed-data rate, and basic health
- Optional real-time UDP forwarding to a hand skeleton viewer
- Compatible with Windows, macOS, and Linux
- Uses `multiprocessing` with Windows `spawn` safety

## Requirements

Install the only required Python packages:

```bash
pip install avp_stream numpy
```

## Files

- [test_avp_multi_receiver.py](./test_avp_multi_receiver.py): main script

## Quick Start

### 1. Single receiver test

```bash
python test_avp_multi_receiver.py --ip 192.168.1.29 --mode single --duration 60
```

### 2. Dual receiver test

```bash
python test_avp_multi_receiver.py --ip 192.168.1.29 --duration 60
```

`--mode` defaults to `dual`.

### 3. Enable `record=True` in `VisionProStreamer`

```bash
python test_avp_multi_receiver.py --ip 192.168.1.29 --duration 60 --record
```

## Real-Time Viewer Forwarding

This script can forward one receiver's right-hand data to a UDP viewer.

### Important

- `--viewer-rate` changes the **UDP forwarding rate to the viewer**
- It does **not** change the Vision Pro receive loop rate
- It does **not** change `poll_fps` directly

For example, this sends viewer packets at `20 Hz`:

```bash
python test_avp_multi_receiver.py --ip 192.168.1.29 --duration 600 --forward-viewer --viewer-rate 20
```

### Forward to a viewer on the same machine

```bash
python test_avp_multi_receiver.py --ip 192.168.1.29 --duration 600 --forward-viewer --viewer-ip 127.0.0.1 --viewer-port 5005 --viewer-rate 30
```

### Choose which receiver forwards data

```bash
python test_avp_multi_receiver.py --ip 192.168.1.29 --mode dual --forward-viewer --viewer-receiver A
```

or

```bash
python test_avp_multi_receiver.py --ip 192.168.1.29 --mode dual --forward-viewer --viewer-receiver B
```

## CLI Reference

```text
--ip               Vision Pro IP address, required
--duration         Test duration in seconds, default 60
--hz               Main-process print frequency in Hz, default 5
--record           Pass record=True to VisionProStreamer
--mode             single or dual, default dual
--forward-viewer   Forward one receiver's right-hand data to a UDP viewer
--viewer-ip        Viewer target IP, default 127.0.0.1
--viewer-port      Viewer target UDP port, default 5005
--viewer-rate      Viewer forwarding rate in Hz, default 30
--viewer-receiver  Which receiver forwards to the viewer, A or B, default A
```

## What the Script Prints

Each receiver reports fields such as:

- `poll_fps`: how fast the script reads `vps.latest`
- `valid_fps`: how often non-empty valid dict data is received
- `changed_fps`: how often key hand fields actually change numerically
- `unchanged_count`: how many consecutive reads had unchanged key data
- `right_fingers_shape` / `left_fingers_shape`
- `last_error`
- `reported_recently`

## How to Read the Result

### Likely healthy

- Both receivers keep printing status
- Both receivers show `reported_recently=True`
- `valid_fps` stays stable
- No persistent connection errors

### Possible problem

- One receiver stops reporting
- One receiver has very low `valid_fps`
- One receiver keeps returning invalid data
- One receiver repeatedly shows connection-related errors

## Warnings Explained

### `WARNING: data unchanged across reads`

This only means the key hand fields did not change numerically across
consecutive reads.

This is not automatically a failure.

If the user is holding still, healthy data can also look unchanged.

### `WARNING: source frame not advancing`

This warning is only used when the incoming data contains a detectable top-level
source frame field such as `timestamp`, `frame_id`, `seq`, or `counter`, and
that value stops advancing.

This is more suspicious than simple unchanged joint values.

## Data Format Notes

The script observes raw Vision Pro hand data such as:

- `right_wrist`: usually shape `(1, 4, 4)`
- `left_wrist`: usually shape `(1, 4, 4)`
- `right_fingers`: usually shape `(25, 4, 4)`
- `left_fingers`: usually shape `(25, 4, 4)`

For `right_fingers` / `left_fingers`, `(25, 4, 4)` means:

- `25`: 25 hand joints / keypoints / bone transforms
- each entry is a `4x4` homogeneous transform matrix

From each `4x4` matrix:

- `matrix[:3, 3]` gives the 3D position
- `matrix[:3, :3]` gives the orientation

For viewer forwarding, this script converts Vision Pro's 25-point right hand
into a MediaPipe-like 21-point skeleton and sends it over UDP.

## Typical Workflows

### Test whether dual receivers are supported

```bash
python test_avp_multi_receiver.py --ip 192.168.1.29 --duration 300
```

### Keep Linux as the main receiver and use Windows only for visualization

Run the Linux-side main receiver separately, and use this script on Windows with
viewer forwarding enabled at a modest rate such as `20` or `30 Hz`.

Example:

```bash
python test_avp_multi_receiver.py --ip 192.168.1.29 --mode single --duration 600 --forward-viewer --viewer-rate 20
```

## Troubleshooting

### No data arrives

- Check the Vision Pro IP
- Make sure the Vision Pro app is running
- Confirm both devices are on the same network
- Try `--mode single` first

### Viewer looks wrong when translating the hand

The current forwarding path treats `right_fingers` as hand-local data and avoids
subtracting the absolute world wrist position again. This matches the observed
`avp_stream` behavior more closely for visualization.

### Windows viewer receives nothing

- Check `--viewer-ip` and `--viewer-port`
- Check Windows Firewall for UDP
- Try `127.0.0.1` first on one machine

## Exit Behavior

- The script exits normally when `--duration` is reached
- Press `Ctrl+C` to stop early
- On Windows, `avp_stream` may print its own signal-related messages during shutdown

## License / Usage

This repository currently contains a standalone utility script intended for
testing, debugging, and visualization workflows around `avp_stream` and Apple
Vision Pro hand tracking.
