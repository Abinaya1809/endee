"""
Vehicle Detection using YOLOv8
================================
Detects vehicles in images, videos, or webcam feed.
Supports: car, truck, bus, motorcycle, bicycle

Usage:
  python vehicle_detector.py --source image.jpg
  python vehicle_detector.py --source video.mp4
  python vehicle_detector.py --source 0          # webcam
  python vehicle_detector.py --source image.jpg --save
"""

import argparse
import cv2
import time
from pathlib import Path
from collections import Counter

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Install ultralytics: pip install ultralytics")

# ─── Vehicle class IDs in COCO dataset (used by YOLOv8) ──────────────────────
VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    1:  "bicycle",
}

# ─── Distinct colors per vehicle type (BGR) ──────────────────────────────────
CLASS_COLORS = {
    "car":        (0,   200, 255),   # amber
    "motorcycle": (255, 100,   0),   # blue
    "bus":        (0,   255, 100),   # green
    "truck":      (100,   0, 255),   # red
    "bicycle":    (255, 200,   0),   # cyan
}


class VehicleDetector:
    def __init__(self, model_size: str = "n", conf: float = 0.4, iou: float = 0.45):
        """
        Args:
            model_size: YOLOv8 variant — n(ano), s(mall), m(edium), l(arge), x(tra-large)
            conf:       Confidence threshold (0-1)
            iou:        IoU threshold for NMS
        """
        model_name = f"yolov8{model_size}.pt"
        print(f"[INFO] Loading {model_name} …")
        self.model = YOLO(model_name)
        self.conf  = conf
        self.iou   = iou
        self.total_count = 0
        print("[INFO] Model ready.")
        self.crossed_ids = set()
        self.total_count = 0
        

    # ── draw detections on a frame ────────────────────────────────────────────
    def _draw(self, frame, results):
        counts = Counter()
        for box in results[0].boxes:
            cls_id = int(box.cls)
            if cls_id not in VEHICLE_CLASSES:
                continue
            label   = VEHICLE_CLASSES[cls_id]
            conf    = float(box.conf)
            color   = CLASS_COLORS[label]
            counts[label] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            text    = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Summary overlay
        summary = "  ".join(f"{v}:{c}" for v, c in counts.most_common())
        total   = sum(counts.values())
        overlay = f"Vehicles: {total}   [{summary}]"
        cv2.putText(frame, overlay, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, overlay, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        return frame, counts

    # ── image ─────────────────────────────────────────────────────────────────
    def detect_image(self, path: str, save: bool = False, show: bool = True):
        frame   = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        frame, counts = self._draw(frame, results)

        print(f"\n[RESULT] {path}")
        print(f"  Total vehicles detected : {sum(counts.values())}")
        for v, c in counts.most_common():
            print(f"    {v:15s}: {c}")

        if save:
            out = Path(path).stem + "_detected.jpg"
            cv2.imwrite(out, frame)
            print(f"  Saved → {out}")

        if show:
            cv2.imshow("Vehicle Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return frame, counts

    # ── video / webcam ────────────────────────────────────────────────────────
    def detect_video(self, source, save: bool = False, show: bool = True):
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            raise IOError(f"Cannot open source: {source}")

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None
        if save:
            out_path = Path(str(source)).stem + "_detected.mp4" if not str(source).isdigit() else "webcam_detected.mp4"
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            print(f"[INFO] Saving output → {out_path}")

        frame_id = 0
        t_start  = time.time()
        print("[INFO] Running … press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
            frame, counts = self._draw(frame, results)

            # FPS counter
            elapsed = time.time() - t_start
            fps_live = frame_id / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps_live:.1f}", (w - 120, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 100), 2)

            if writer:
                writer.write(frame)
            if show:
                cv2.imshow("Vehicle Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Interrupted by user.")
                    break

            frame_id += 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Processed {frame_id} frames in {time.time()-t_start:.1f}s")

    # ── convenience dispatcher ────────────────────────────────────────────────
    def detect(self, source: str, save: bool = False, show: bool = True):
        VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        if str(source).isdigit():
            self.detect_video(source, save=save, show=show)
        elif Path(source).suffix.lower() in VIDEO_EXTS:
            self.detect_video(source, save=save, show=show)
        else:
            self.detect_image(source, save=save, show=show)


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection with YOLOv8")
    parser.add_argument("--source", required=True,
                        help="Path to image/video, or camera index (0, 1, …)")
    parser.add_argument("--model",  default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size (default: n)")
    parser.add_argument("--conf",   type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    parser.add_argument("--iou",    type=float, default=0.45,
                        help="IoU threshold (default: 0.45)")
    parser.add_argument("--save",   action="store_true",
                        help="Save annotated output")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable live window (useful for headless servers)")
    args = parser.parse_args()

    detector = VehicleDetector(model_size=args.model,
                                conf=args.conf,
                                iou=args.iou)
    detector.detect(args.source,
                    save=args.save,
                    show=not args.no_show)


if __name__ == "__main__":
    main()
