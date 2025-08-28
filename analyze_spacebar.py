#! /usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def load_or_select_roi(
    cap: cv2.VideoCapture,
    video_path: str,
    roi_arg: Optional[List[int]],
    select_interactively: bool,
) -> ROI:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame from video.")

    # Try to load saved ROI next to the video
    roi_sidecar = f"{video_path}.spacebar_roi.json"
    if not select_interactively and roi_arg is None and os.path.exists(roi_sidecar):
        with open(roi_sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ROI(**data)

    if roi_arg is not None:
        x, y, w, h = roi_arg
        return ROI(x, y, w, h)

    if select_interactively:
        # Show a scaled preview if the frame is large to make ROI selection easier
        display = frame.copy()
        title = "Select ROI covering the SPACEBAR highlight area, then press ENTER"
        r = cv2.selectROI(title, display, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        if r is None or r == (0, 0, 0, 0):
            raise RuntimeError("No ROI selected.")
        x, y, w, h = map(int, r)
        roi = ROI(x, y, w, h)
        with open(roi_sidecar, "w", encoding="utf-8") as f:
            json.dump(roi.__dict__, f)
        return roi

    # Fallback: a conservative top-left region (works if overlay is in top-left)
    h_frame, w_frame = frame.shape[:2]
    default_w = int(w_frame * 0.28)
    default_h = int(h_frame * 0.22)
    return ROI(10, 10, max(40, default_w), max(40, default_h))


def count_presses(
    cap: cv2.VideoCapture,
    roi: ROI,
    on_threshold: float,
    off_threshold: float,
    min_release_frames: int,
    show: bool,
    show_progress: bool,
    record_times: bool,
    save_dir: Optional[str],
    save_roi_only: bool,
    require_solid: bool,
    solid_threshold: float,
    save_only_counted: bool,
) -> Tuple[int, List[float]]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    x, y, w, h = roi.as_tuple()

    # Press counting is strictly on rising edge: previous frame not pressed,
    # current frame pressed. We also require a minimum number of consecutive
    # release frames before a new press can be counted to avoid flicker.
    pressed = False
    count = 0
    not_pressed_streak = 1_000_000  # allow first press
    press_times: List[float] = []

    kernel = np.ones((3, 3), np.uint8)
    solid_kernel = np.ones((5, 5), np.uint8)
    frame_index = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Optional progress bar
    pbar = None
    if show_progress:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        try:
            from tqdm import tqdm  # type: ignore

            pbar = tqdm(
                total=total_frames if total_frames > 0 else None,
                unit="frame",
                leave=False,
                desc="Analyzing",
            )
        except Exception:
            pbar = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1

        roi_img = frame[y : y + h, x : x + w]
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # Yellow range in HSV. Tune if needed via CLI.
        lower = np.array([15, 70, 120], dtype=np.uint8)
        upper = np.array([40, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Coverage of yellow pixels in ROI
        yellow_ratio = float(np.count_nonzero(mask)) / float(mask.size)

        # "Solid" coverage: enforce contiguity and interior fullness to avoid
        # counting background speckles or distant yellow objects
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        eroded = cv2.erode(closed, solid_kernel, iterations=1)
        solid_ratio = float(np.count_nonzero(eroded)) / float(eroded.size)

        # Decide next state using the existing not_pressed_streak (from previous frames)
        next_pressed = pressed
        if pressed:
            # Stay pressed until confidence drops below thresholds
            below_off = yellow_ratio < off_threshold
            if require_solid:
                below_off = below_off or (solid_ratio < max(0.6 * solid_threshold, solid_threshold - 0.1))
            if below_off:
                next_pressed = False
        else:
            # Rising edge only when above thresholds and enough release frames observed
            meets_on = yellow_ratio > on_threshold
            if require_solid:
                meets_on = meets_on and (solid_ratio >= solid_threshold)
            if meets_on and not_pressed_streak >= min_release_frames:
                next_pressed = True

        # Rising edge: count only when transitioning False -> True
        is_rising_edge = (not pressed and next_pressed)
        if is_rising_edge:
            count += 1
            if record_times:
                press_times.append(frame_index / fps)

        # Visualization and saving. Use next state for visuals so the box color
        # matches what will be in effect after this frame's decision.
        if show or (
            save_dir
            and (
                (save_only_counted and is_rising_edge)
                or (not save_only_counted and next_pressed)
            )
        ):
            vis = frame.copy()
            color_box = (0, 220, 0) if next_pressed else (60, 60, 220)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color_box, 2)
            cv2.putText(
                vis,
                f"y={yellow_ratio:.3f} s={solid_ratio:.3f} count={count}",
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_box,
                1,
                cv2.LINE_AA,
            )
            if show:
                cv2.imshow("spacebar-counter", vis)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break
            if save_dir and ((save_only_counted and is_rising_edge) or (not save_only_counted and next_pressed)):
                out_img = roi_img if save_roi_only else vis
                out_path = os.path.join(
                    save_dir,
                    (
                        f"counted_f{frame_index:06d}_t{frame_index / fps:.3f}.jpg"
                        if save_only_counted
                        else f"pressed_f{frame_index:06d}_t{frame_index / fps:.3f}.jpg"
                    ),
                )
                cv2.imwrite(out_path, out_img)

        pressed = next_pressed

        # Update consecutive release frame streak AFTER deciding next state,
        # so that the decision uses the streak from prior frames.
        if yellow_ratio < off_threshold:
            not_pressed_streak += 1
        else:
            not_pressed_streak = 0

        if pbar is not None:
            try:
                pbar.update(1)
                pbar.set_postfix_str(f"count={count}")
            except Exception:
                pass

        

    if pbar is not None:
        try:
            pbar.close()
        except Exception:
            pass

    if show:
        cv2.destroyAllWindows()

    return count, press_times


def parse_roi(s: str) -> List[int]:
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must have four integers: x y w h")
    vals = list(map(int, parts))
    if any(v < 0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
        raise argparse.ArgumentTypeError("Invalid ROI values")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count SPACEBAR presses in a video by detecting the yellow highlight "
            "of the overlay in a user-specified ROI."
        )
    )
    parser.add_argument("video", help="Path to the input video (e.g., sample.mp4)")
    parser.add_argument(
        "--roi",
        type=parse_roi,
        help="ROI as 'x y w h' or 'x,y,w,h' (pixels). If omitted, tries saved ROI or fallback.",
    )
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Open a window to select and save the ROI for this video",
    )
    parser.add_argument(
        "--on-threshold",
        type=float,
        default=0.035,
        help="Yellow pixel ratio to consider the key PRESSED (default: 0.035)",
    )
    parser.add_argument(
        "--off-threshold",
        type=float,
        default=0.020,
        help="Yellow pixel ratio to consider the key RELEASED (default: 0.020)",
    )
    parser.add_argument(
        "--min-release-frames",
        type=int,
        default=3,
        help=(
            "Require this many consecutive release frames before a new press is counted "
            "(default: 3). Prevents multiple counts when holding or slight flicker."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Visualize detection while processing",
    )
    parser.add_argument(
        "--print-times",
        action="store_true",
        help="Print timestamps (seconds) of each detected press",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar while analyzing",
    )
    parser.add_argument(
        "--save-pressed-dir",
        type=str,
        default=None,
        help=(
            "Directory to save images for frames detected as PRESSED. "
            "Creates it if missing."
        ),
    )
    parser.add_argument(
        "--save-pressed-roi-only",
        action="store_true",
        help="If set, save only the ROI crop instead of the whole frame",
    )
    parser.add_argument(
        "--require-solid",
        action="store_true",
        help=(
            "Require solid coverage across the ROI to count as PRESSED. Helps avoid "
            "false positives from small yellow objects in the background."
        ),
    )
    parser.add_argument(
        "--solid-threshold",
        type=float,
        default=0.85,
        help=(
            "Minimum solid coverage ratio within ROI when --require-solid is used "
            "(default: 0.85)."
        ),
    )
    parser.add_argument(
        "--save-only-counted",
        action="store_true",
        help=(
            "When saving images, write only frames that are actually COUNTED as presses "
            "(rising-edge frames)."
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Failed to open video.")

    try:
        roi = load_or_select_roi(
            cap=cap,
            video_path=args.video,
            roi_arg=args.roi,
            select_interactively=args.select_roi,
        )
    except Exception as e:  # pragma: no cover - UX errors
        cap.release()
        raise

    # Prepare output directory if requested
    save_dir = None
    if args.save_pressed_dir:
        save_dir = args.save_pressed_dir
        os.makedirs(save_dir, exist_ok=True)

    count, press_times = count_presses(
        cap=cap,
        roi=roi,
        on_threshold=args.on_threshold,
        off_threshold=args.off_threshold,
        min_release_frames=args.min_release_frames,
        show=args.show,
        show_progress=args.progress,
        record_times=args.print_times,
        save_dir=save_dir,
        save_roi_only=args.save_pressed_roi_only,
        require_solid=args.require_solid,
        solid_threshold=args.solid_threshold,
        save_only_counted=args.save_only_counted,
    )
    cap.release()

    print(f"Total spacebar presses: {count}")
    if args.print_times and press_times:
        for t in press_times:
            print(f"{t:.3f}")


if __name__ == "main":  # support running via python -m if needed
    main()

if __name__ == "__main__":
    main()


