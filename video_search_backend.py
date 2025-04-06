from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import sqlite3
import cv2
from ultralytics import YOLO
import subprocess

app = FastAPI()

# Mount static folder to serve clips
app.mount(
    "/uploaded_videos", StaticFiles(directory="uploaded_videos"), name="uploaded_videos"
)

# Directories and paths
VIDEO_DIR = "uploaded_videos"
VIDEO_PATH = os.path.join(VIDEO_DIR, "video.mp4")
DB_PATH = os.path.join(VIDEO_DIR, "detections.db")
CLIPS_DIR = os.path.join(VIDEO_DIR, "clips")
MUSIC_PATH = os.path.join(VIDEO_DIR, "background.mp3")  # Add your music file here
FINAL_OUTPUT = os.path.join(VIDEO_DIR, "final_video.mp4")

# Ensure folders exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT NOT NULL,
        timestamp REAL NOT NULL,
        confidence REAL NOT NULL
    )
"""
)
conn.commit()


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("video_search_ui.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    # Save video
    with open(VIDEO_PATH, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Clear previous data
    cursor.execute("DELETE FROM detections")
    conn.commit()

    # Process video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # one frame per second
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            results = model(frame)
            detected_labels = set()
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < 0.4:
                    continue
                cls_id = int(box.cls[0])
                label = results[0].names[cls_id].lower()
                if label in detected_labels:
                    continue
                detected_labels.add(label)
                timestamp = frame_num / fps
                cursor.execute(
                    "INSERT INTO detections (label, timestamp, confidence) VALUES (?, ?, ?)",
                    (label, timestamp, conf),
                )
        frame_num += 1

    conn.commit()
    cap.release()
    # After processing, save label list to a text file
    cursor.execute("SELECT DISTINCT label FROM detections")
    labels = [row[0] for row in cursor.fetchall()]
    with open(os.path.join(VIDEO_DIR, "detected_labels.txt"), "w") as f:
        for label in labels:
            f.write(label + "\n")

    return {
        "status": "success",
        "labels_file": "uploaded_videos/detected_labels.txt",
        "labels": labels,
    }


@app.get("/stitch")
async def stitch_clips(q: str = Query(...)):
    cursor.execute(
        "SELECT timestamp FROM detections WHERE label LIKE ? ORDER BY timestamp",
        (f"%{q.lower()}%",),
    )
    times = [round(row[0], 2) for row in cursor.fetchall()]

    if not times:
        return JSONResponse(content={"error": "No clips to stitch"}, status_code=404)

    clips = []
    clip_start = times[0]
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > 2:
            clips.append((clip_start, times[i - 1]))
            clip_start = times[i]
    clips.append((clip_start, times[-1]))

    input_txt = os.path.join(CLIPS_DIR, "input.txt")
    with open(input_txt, "w") as f:
        for i, (start, end) in enumerate(clips):
            duration = end - start + 1
            if duration < 2:
                continue
            clip_path = os.path.join(CLIPS_DIR, f"{q.lower()}_{i}.mp4")
            print(
                f"Creating clip: {clip_path}, start: {start}, duration: {duration}"
            )  # Debug print
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start),
                    "-i",
                    VIDEO_PATH,
                    "-t",
                    str(duration),
                    "-an",
                    "-c:v",
                    "libx264",
                    clip_path,
                ]
            )
            f.write(f"file '{os.path.abspath(clip_path)}'\n")

    # Stitch all clips
    stitched = os.path.join(CLIPS_DIR, f"{q.lower()}_stitched.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            input_txt,
            "-c",
            "copy",
            stitched,
        ]
    )

    # Add background music
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            stitched,
            "-i",
            MUSIC_PATH,
            "-shortest",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            FINAL_OUTPUT,
        ]
    )

    return {"status": "success", "final_video": "/final"}


@app.get("/video-status")
async def video_status():
    return {"videoExists": os.path.exists(VIDEO_PATH)}


@app.get("/video")
async def get_uploaded_video():
    if os.path.exists(VIDEO_PATH):
        return FileResponse(VIDEO_PATH, media_type="video/mp4")
    return JSONResponse(content={"error": "Video not found"}, status_code=404)


@app.get("/search")
async def search_video(q: str = Query(...)):
    cursor.execute(
        "SELECT timestamp FROM detections WHERE label LIKE ? ORDER BY timestamp",
        (f"%{q.lower()}%",),
    )
    times = [round(row[0], 2) for row in cursor.fetchall()]

    if not times:
        return JSONResponse(content={"results": []})

    grouped = []
    group = [times[0]]
    for i in range(1, len(times)):
        if times[i] - times[i - 1] <= 2:
            group.append(times[i])
        else:
            if group[-1] - group[0] >= 2:
                grouped.append((group[0], group[-1]))
            group = [times[i]]
    if group[-1] - group[0] >= 2:
        grouped.append((group[0], group[-1]))

    results = []
    for i, (start, end) in enumerate(grouped):
        duration = end - start + 1
        if duration < 2:
            continue
        clip_path = os.path.join(CLIPS_DIR, f"{q.lower()}_{i}.mp4")
        # Generate clip on-the-fly
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-i",
                VIDEO_PATH,
                "-t",
                str(duration),
                "-an",
                "-c:v",
                "libx264",
                clip_path,
            ]
        )
        clip_url = f"/uploaded_videos/clips/{q.lower()}_{i}.mp4"
        results.append(
            {
                "label": q.lower(),
                "time": start,
                "start": start,
                "end": end,
                "clip": clip_url,
            }
        )

    return {"results": results}


@app.get("/final")
async def get_final_video():
    if os.path.exists(FINAL_OUTPUT):
        return FileResponse(FINAL_OUTPUT, media_type="video/mp4")
    return JSONResponse(content={"error": "Final video not found"}, status_code=404)
