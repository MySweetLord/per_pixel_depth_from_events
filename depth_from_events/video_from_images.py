import cv2
import numpy as np
from pathlib import Path
import re
import tqdm


# Configuration
folders = [
    Path("C:\\Users\\Razer\\Documents\\ENGINEERING PHYSICS\\MSc\\MSC THESIS NCS\\depth_from_events\\logs\\images\\validate_run1\\cmax_accumulated_events"),
    Path("C:\\Users\\Razer\\Documents\\ENGINEERING PHYSICS\\MSc\\MSC THESIS NCS\\depth_from_events\\logs\\images\\validate_run1\\cmax_image_warped_events_t"),
    Path("C:\\Users\\Razer\\Documents\\ENGINEERING PHYSICS\\MSc\\MSC THESIS NCS\\depth_from_events\\logs\\images\\validate_run1\\flow"),
    Path("C:\\Users\\Razer\\Documents\\ENGINEERING PHYSICS\\MSc\\MSC THESIS NCS\\depth_from_events\\logs\\images\\validate_run1\\disparity"),
    Path("C:\\Users\\Razer\\Documents\\ENGINEERING PHYSICS\\MSc\\MSC THESIS NCS\\depth_from_events\\logs\\images\\validate_run1\\events"),
]

labels = [
    "cmax_accumulated_events",
    "cmax_image_warped_events_t",
    "flow",
    "disparity",
    "events",
    "event-masked disparity",
]

fps = 50
frame_width = 320
frame_height = 240
header_height = 30
output_path = "C:\\Users\\Razer\\Documents\\ENGINEERING PHYSICS\\MSc\\MSC THESIS NCS\\depth_from_events\\logs\\images\\run1_video.mp4"

# Natural sorting: frame_2.png before frame_10.png
def natural_key(path):
    return [
        int(s) if s.isdigit() else s.lower()
        for s in re.split(r"(\d+)", path.name)
    ]

# Load image paths
sequences = [sorted(folder.glob("*.png"), key=natural_key) for folder in folders]

# Check sequences
n_frames = len(sequences[0])
assert n_frames > 0, "No PNG images found"
assert all(len(seq) == n_frames for seq in sequences), \
    "Folders must contain the same number of images"

# Initialize video writer with a 2x3 panel layout.
columns     = 3
rows        = 2
cell_height = frame_height + header_height
video_width = columns * frame_width
video_height = rows * cell_height

writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (video_width, video_height))

if not writer.isOpened():
    raise RuntimeError("Could not open video writer")

try:
    for i in tqdm.tqdm(range(n_frames), desc="Generating video"):

        canvas = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        images = []

        for j, sequence in enumerate(sequences):
            img = cv2.imread(str(sequence[i]))
            if img is None:
                raise ValueError(f"Cannot read {sequence[i]}")
            images.append(img)

        event_mask = cv2.cvtColor(images[4], cv2.COLOR_BGR2GRAY) > 0
        if event_mask.shape != images[3].shape[:2]:

            event_mask = cv2.resize(
                event_mask.astype(np.uint8),
                (images[3].shape[1], images[3].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        images.append(cv2.bitwise_and(images[3], images[3], mask=event_mask.astype(np.uint8)))

        for j, img in enumerate(images):
            # Resize while preserving aspect ratio
            h, w = img.shape[:2]
            scale = min(frame_width / w, frame_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            img = cv2.resize(img, (new_w, new_h))

            column = j % columns
            row = j // columns
            x = column * frame_width + (frame_width - new_w) // 2
            y = row * cell_height + header_height + (frame_height - new_h) // 2

            canvas[y:y+new_h, x:x+new_w] = img

            # Add panel label
            cv2.putText(
                canvas, labels[j],
                (column * frame_width + 10, row * cell_height + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )

        writer.write(canvas)
finally:
    writer.release()

print(f"Saved video to {output_path}")