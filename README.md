# YOLO + OpenCV Unique-Object Video Counter

A command-line tool that detects objects in a video with **YOLOv8**,
tracks them across frames with a lightweight custom IoU tracker, and
reports **how many unique objects of each class appeared in the
video** — not just how many detections were made per frame. The
output is an annotated video plus a JSON/CSV summary report.

> Built for the Computer Vision course "Build Your Own Project"
> assignment.
> **Author:** Arya Mishra | **Registration No.:** 24BAI10415

---

## 1. Overview

Most simple YOLO demo scripts just draw a box on every detected
object in every frame. That over-counts moving objects (the same car
gets "counted" in every frame it's visible). This project adds a
tracking layer on top of YOLO so that each physical object is
assigned a stable ID and only counted **once**, giving a genuinely
useful "total unique objects seen" number for the whole video.

## 2. Features

- Runs any Ultralytics YOLOv8 `.pt` model (defaults to the small,
  fast `yolov8n.pt`, downloaded automatically on first run).
- Custom centroid/IoU tracker (no extra tracking dependency) that
  assigns a persistent ID per object.
- Live on-screen panel showing the running unique count per class.
- JSON + CSV export of the final counts.
- Configurable via command-line flags: confidence threshold, target
  classes, frame resize, frame skipping, CPU/GPU device, etc.
- Unit-tested tracker and counter logic (`tests/`).
- Runs entirely from the terminal — no GUI required.

## 3. Technologies / Tools Used

| Component        | Technology                     |
|-------------------|--------------------------------|
| Object detection  | YOLOv8 (Ultralytics)            |
| Video I/O & drawing | OpenCV (`opencv-python`)      |
| Language          | Python 3.9+                    |
| Testing           | `pytest`                        |
| Data export       | Built-in `json` / `csv` modules |

## 4. Project Structure

```
yolo-object-counter/
├── main.py                  # CLI entry point
├── requirements.txt
├── README.md
├── statement.md
├── src/
│   ├── config.py             # Central configuration dataclass
│   ├── detector.py           # Module 1: YOLO detection wrapper
│   ├── tracker.py            # Module 2: IoU-based object tracker
│   ├── counter.py            # Module 2b: unique-occurrence counting
│   ├── visualizer.py         # Module 3: drawing + JSON/CSV export
│   ├── video_io.py           # VideoReader / VideoWriter helpers
│   └── logger_utils.py       # Rotating file + console logger
├── tests/
│   ├── test_tracker.py       # Unit tests for the tracker
│   └── test_counter.py       # Unit tests for the counter
├── docs/                     # Architecture / workflow / UML diagrams
└── sample_media/             # Put a sample input video here
```

## 5. Environment Setup

### 5.1 Prerequisites

- Python 3.9 or newer
- `pip` (comes with Python)
- (Optional) An NVIDIA GPU + CUDA if you want faster inference — the
  project runs fine on CPU for short clips.

### 5.2 Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 5.3 Create a virtual environment (recommended)

```bash
python3 -m venv venv

# Activate it:
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 5.4 Install dependencies

```bash
pip install -r requirements.txt
```

This installs `ultralytics` (which pulls in PyTorch), `opencv-python`,
`numpy`, and `pytest`.

> The first time you run the tool, Ultralytics will automatically
> download the `yolov8n.pt` weights file (~6 MB) if it isn't already
> present in the working directory. An internet connection is
> required for this one-time download only.

## 6. Configuration

All defaults live in `src/config.py` and can be overridden with CLI
flags (see `python main.py --help`). The most commonly used ones:

| Flag | Default | Meaning |
|---|---|---|
| `--input` / `-i` | *(required)* | Path to input video, or `0` for a webcam |
| `--output` / `-o` | `output/annotated_output.mp4` | Path to save the annotated video |
| `--model` / `-m` | `yolov8n.pt` | YOLO weights to use |
| `--conf` | `0.40` | Detection confidence threshold |
| `--classes` | *(all)* | Comma-separated class filter, e.g. `person,car` |
| `--resize-width` | `960` | Resize frames for speed (`0` = keep original) |
| `--skip` | `1` | Process every Nth frame |
| `--device` | `cpu` | `cpu` or `cuda:0` |

## 7. How to Run

Place a video file inside `sample_media/` (or point `--input` at any
path on your machine), then run:

```bash
python main.py --input sample_media/input_sample.mp4 \
                --output output/annotated_output.mp4 \
                --model yolov8n.pt \
                --conf 0.4
```

To use a live webcam instead of a file:

```bash
python main.py --input 0
```

To only count people and cars, resized to 640px wide, skipping every
other frame for speed:

```bash
python main.py --input sample_media/input_sample.mp4 \
                --classes person,car --resize-width 640 --skip 2
```

### Output

After the run finishes you will find, under `output/`:

- `annotated_output.mp4` — the video with bounding boxes, track IDs
  and the live unique-count panel drawn on every frame.
- `count_report.json` — final tally, e.g.:
  ```json
  {
    "total_unique_objects": 7,
    "counts_by_class": { "car": 4, "person": 3 },
    "video_stats": { "total_frames_processed": 452, "source_fps": 29.97 }
  }
  ```
- `count_report.csv` — the same tally in CSV form.
- `run.log` — a timestamped log of the run.

## 8. Instructions for Testing

Unit tests cover the tracker's ID-assignment logic and the counter's
unique-counting logic (these do not require the YOLO weights or a
video file, so they run instantly):

```bash
pip install pytest   # already in requirements.txt
python -m pytest tests/ -v
```

Expected result: all 12 tests pass, e.g.

```
tests/test_tracker.py::test_iou_identical_boxes_is_one PASSED
tests/test_tracker.py::test_same_object_moving_slightly_keeps_same_id PASSED
tests/test_tracker.py::test_track_is_dropped_after_max_disappeared_frames PASSED
tests/test_counter.py::test_same_track_id_across_frames_is_not_double_counted PASSED
...
12 passed
```

To manually sanity-check the full pipeline end to end, run `main.py`
on any short (10–30 second) test clip and confirm that:
1. `output/annotated_output.mp4` plays back with boxes and IDs drawn.
2. The unique count in `count_report.json` roughly matches what you
   count by eye when watching the clip.

## 9. Screenshots

See `docs/sample_input_frame.png` and `docs/sample_output_frame.png`
for a representative before/after frame, and the project report PDF
for the full set of design diagrams and results.

## 10. Design Documentation

Architecture, workflow, and UML diagrams live in `docs/`:
- `architecture_diagram.png`
- `workflow_diagram.png`
- `use_case_diagram.png`
- `class_diagram.png`
- `sequence_diagram.png`

## 11. Known Limitations / Future Enhancements

- The tracker is IoU-based and can lose an ID if an object is fully
  occluded for longer than `--max-disappeared` frames, or if two
  objects of the same class cross paths.
- No re-identification (an object leaving and re-entering the frame
  is counted as a new object) — a future enhancement could add
  appearance-based re-ID embeddings.
- No GUI — this is intentional, per the assignment's "must run from
  the command line" requirement.
