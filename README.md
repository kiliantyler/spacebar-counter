# Running against Polypuff's 2025/08/28 stream

This section documents how the full-stream analysis was produced for reproducibility and auditing.

## Acquisition

- Downloaded the stream locally for the sole purpose of counting spacebar presses. A typical command to retrieve a single MP4 with soft constraints on quality:

```bash
yt-dlp -f 'bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best' -o full_stream.mp4 '<STREAM_URL>'
```

## ROI selection rationale

- The visual spacebar indicator sits near the top-left of the frame. For this stream, a tight ROI avoids background yellow objects and improves performance. The ROI used was:

```
--roi '230,210,120,40'
```

If your video differs, first run with `--select-roi` to draw a bounding box over the indicator once; it will be saved as `full_stream.mp4.spacebar_roi.json` and reused.

## Detector settings used

- Solid coverage required to avoid distant yellow patches: `--require-solid`
- Strict solid coverage ratio: `--solid-threshold 0.85`
- Debounce for held keys and flicker: `--min-release-frames 3` (default)
- Save only rising-edge frames (one image per counted press): `--save-only-counted`
- Save ROI crops for quick inspection: `--save-pressed-roi-only`
- Progress bar for long runs: `--progress`

## Command and result

```bash
./analyze_spacebar.py ./full_stream.mp4 --roi '230,210,120,40' \
  --save-pressed-dir ./f --save-pressed-roi-only \
  --require-solid --save-only-counted --progress
Total spacebar presses: 37984
```

# Spacebar Press Counter (Video)

This script analyzes a video and counts how many times the spacebar is pressed. It assumes there is a visual spacebar indicator in the top-left (or any user-selected) region that changes color (e.g., yellow) when pressed.

## Features

- ROI (Region of Interest) selection: interactively draw or pass coordinates; saved as a sidecar JSON.
- Robust press detection: counts on rising edges only (no repeats while held).
- Debounce: require N consecutive release frames to avoid flicker double-counts.
- Solid coverage gate: require most of the ROI to be the target color to avoid background false positives.
- Debug exports: save frames when pressed or only those actually counted as presses; optional ROI-only crops.
- Progress bar (optional) and per-press timestamps (optional).

## Requirements

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Select ROI once and visualize:

```bash
./analyze_spacebar.py ./sample.mp4 --select-roi --show --progress
```

This saves the ROI next to your video as `sample.mp4.spacebar_roi.json`.

2. Analyze using saved ROI and stricter detection, saving only counted frames (rising edges) as ROI crops:

```bash
./analyze_spacebar.py ./sample.mp4 \
  --require-solid --solid-threshold 0.85 \
  --min-release-frames 3 \
  --save-only-counted \
  --save-pressed-dir ./f \
  --save-pressed-roi-only \
  --progress
```

Output includes `Total spacebar presses: N`. If `--save-only-counted` is used, one image per counted press is written to `./f` named like `counted_f000123_t4.100.jpg`.

## CLI Reference

- `video` (positional): Path to the input video.
- `--roi "x y w h"` or `--roi "x,y,w,h"`: Manually set ROI (pixels).
- `--select-roi`: Open a window to draw the ROI; saves to `<video>.spacebar_roi.json`.
- `--on-threshold FLOAT` (default `0.035`): Yellow ratio to consider PRESSED.
- `--off-threshold FLOAT` (default `0.020`): Yellow ratio to consider RELEASED.
- `--min-release-frames INT` (default `3`): Consecutive non-pressed frames required before a new press can count.
- `--require-solid`: Require solid coverage across ROI to count as pressed (recommended).
- `--solid-threshold FLOAT` (default `0.85`): Minimum solid coverage ratio when `--require-solid` is set.
- `--save-pressed-dir DIR`: Directory to write images of detected press frames.
- `--save-only-counted`: Save only frames that are actually counted (rising edges). Without this, every pressed frame is saved.
- `--save-pressed-roi-only`: Save only the ROI crop instead of the whole frame.
- `--print-times`: Print timestamps (seconds) for each counted press.
- `--show`: Show a live visualization with ROI box, yellow ratio (y), solid ratio (s), and running count.
- `--progress`: Show a progress bar (requires `tqdm`).

## How Detection Works (Overview)

1. Convert ROI to HSV and threshold for yellow.
2. Compute `y` (yellow coverage ratio) and `s` (solid coverage ratio using close+erode).
3. A frame is considered "pressed" if:
   - `y > on-threshold`, and if `--require-solid` then `s ≥ solid-threshold`.
4. A press is only counted on a rising edge (transition from not-pressed to pressed) and only after `min-release-frames` consecutive release frames.

Tip: During `--show`, you’ll see `y=` and `s=` values. Aim for `on-threshold > off-threshold` with a small gap (e.g., `0.035` vs `0.020`).

## Tuning & Troubleshooting

- Getting 0 presses:

  - Lower `--on-threshold` slightly or lower `--solid-threshold` if your indicator isn’t very saturated.
  - Reduce `--min-release-frames` (e.g., to `1`) if the indicator flickers.

- False positives (counts when it shouldn’t):

  - Increase `--solid-threshold` (e.g., `0.90`) and/or `--on-threshold`.
  - Tighten the ROI to include only the spacebar indicator.

- Too many saved images:

  - Use `--save-only-counted` to write only the rising-edge frames (one per counted press).

- Color not matching:

  - The HSV thresholds live in the script near:
    ```python
    lower = np.array([15, 70, 120], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    ```
    Adjust these if your indicator color differs.

- Visualization window:
  - Press `ESC` to exit early when `--show` is used.

## Outputs

- Console:

  - `Total spacebar presses: N`
  - Optional per-press timestamps with `--print-times`.

- Files:
  - ROI sidecar: `<video>.spacebar_roi.json` (stores `x, y, w, h`).
  - Saved images (if `--save-pressed-dir` is set):
    - `counted_*.jpg` when `--save-only-counted` is used.
    - Otherwise `pressed_*.jpg` for every pressed frame.
