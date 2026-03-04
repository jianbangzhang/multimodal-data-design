"""媒体工具：图片 base64 编码、视频帧提取。"""
import base64
from pathlib import Path
from typing import Optional


def encode_image(path: str) -> tuple[str, str]:
    """
    将图片文件编码为 base64，返回 (base64_data, media_type)。
    支持 jpg / jpeg / png / gif / webp。
    """
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode()
    ext = Path(path).suffix.lower().lstrip(".")
    mt = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    return b64, mt


def extract_uniform_frames(video_path: str, n: int = 8,
                           out_dir: Optional[str] = None,
                           jpeg_quality: int = 90) -> list[str]:
    """
    从视频中均匀采样 N 帧，保存为 JPEG，返回帧路径列表。
    这是业界主流做法（Video-LLaVA / VideoChat 均使用均匀采样）。
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("视频处理需要安装 opencv-python：pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # 默认输出目录
    if out_dir is None:
        out_dir = str(Path("data/raw/video_frames") / Path(video_path).stem)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    frame_paths = []
    indices = [int(total * i / n) for i in range(n)]

    for rank, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        t = idx / fps
        path = str(Path(out_dir) / f"f{rank:02d}_t{t:.1f}s.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        frame_paths.append(path)

    cap.release()
    return frame_paths


def get_image_files(directory: str) -> list[str]:
    """递归获取目录下所有图片文件（jpg / jpeg / png / webp）。"""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        str(p) for p in Path(directory).rglob("*")
        if p.suffix.lower() in exts
    )


def get_video_files(directory: str) -> list[str]:
    """获取目录下所有视频文件（mp4 / avi / mov / mkv）。"""
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(
        str(p) for p in Path(directory).rglob("*")
        if p.suffix.lower() in exts
    )


def get_audio_files(directory: str) -> list[str]:
    """获取目录下所有音频文件（wav / mp3 / flac / m4a）。"""
    exts = {".wav", ".mp3", ".flac", ".m4a"}
    return sorted(
        str(p) for p in Path(directory).rglob("*")
        if p.suffix.lower() in exts
    )
