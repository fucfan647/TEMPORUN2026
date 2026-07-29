from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from .sampling import resize_keep_aspect


class VideoFrameReaderCache:
    def __init__(
        self,
        max_open: int = 8,
        max_image_side: int = 512,
        decoder_backend: str = "opencv",
        device: str = "cpu",
    ):
        if decoder_backend not in {"opencv", "torchcodec"}:
            raise ValueError("decoder_backend must be 'opencv' or 'torchcodec'")
        self.max_open = max(1, int(max_open))
        self.max_image_side = int(max_image_side)
        self.decoder_backend = decoder_backend
        self.device = device
        self._captures: OrderedDict[str, cv2.VideoCapture] = OrderedDict()
        self._decoders: OrderedDict[str, Any] = OrderedDict()
        self.torchcodec_failures = 0

    def _capture(self, video_path: Path) -> cv2.VideoCapture | None:
        key = str(video_path)
        capture = self._captures.pop(key, None)
        if capture is not None:
            self._captures[key] = capture
            return capture

        capture = cv2.VideoCapture(key)
        if not capture.isOpened():
            capture.release()
            return None
        self._captures[key] = capture
        while len(self._captures) > self.max_open:
            _, old_capture = self._captures.popitem(last=False)
            old_capture.release()
        return capture

    def _decoder(self, video_path: Path):
        from torchcodec.decoders import VideoDecoder

        key = str(video_path)
        decoder = self._decoders.pop(key, None)
        if decoder is not None:
            self._decoders[key] = decoder
            return decoder
        decoder = VideoDecoder(
            video_path,
            dimension_order="NHWC",
            device=self.device,
            seek_mode="exact",
            num_ffmpeg_threads=1,
        )
        self._decoders[key] = decoder
        while len(self._decoders) > self.max_open:
            self._decoders.popitem(last=False)
        return decoder

    def _read_torchcodec(
        self,
        video_path: Path,
        frame_ms: int,
    ) -> Image.Image | None:
        try:
            decoder = self._decoder(video_path)
            frame = decoder.get_frames_played_at([max(frame_ms, 0) / 1000.0]).data[0]
            array = frame.detach().to("cpu").contiguous().numpy()
            return resize_keep_aspect(Image.fromarray(array), self.max_image_side)
        except Exception:
            self.torchcodec_failures += 1
            return None

    def _read_opencv(
        self,
        video_path: Path,
        frame_ms: int,
    ) -> Image.Image | None:
        capture = self._capture(video_path)
        if capture is None:
            return None
        capture.set(cv2.CAP_PROP_POS_MSEC, float(frame_ms))
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return resize_keep_aspect(Image.fromarray(frame), self.max_image_side)

    def read(self, video_path: Path, frame_ms: int) -> Image.Image | None:
        if self.decoder_backend == "torchcodec":
            image = self._read_torchcodec(video_path, frame_ms)
            if image is not None:
                return image
        return self._read_opencv(video_path, frame_ms)

    def close(self) -> None:
        while self._captures:
            _, capture = self._captures.popitem(last=False)
            capture.release()
        self._decoders.clear()

    def __enter__(self) -> "VideoFrameReaderCache":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
