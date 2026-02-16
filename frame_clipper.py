"""
FrameClipper Node for ComfyUI

Simple node to shorten video clips by specifying the number of frames to keep.
Clean and straightforward - just cuts the video to the specified frame count.
"""

import torch
import logging
from typing import Tuple

try:
    from .utils import validate_image_tensor
except ImportError:
    # Fallback for direct execution
    from utils import validate_image_tensor

class FrameClipper:
    """
    Simple frame clipping node for ComfyUI.
    
    Takes a video sequence and outputs only the specified number of frames.
    Perfect for shortening clips without complex processing.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("IMAGE", {
                    "tooltip": "Source video sequence to clip (IMAGE tensor format)"
                }),
                "frame_count": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Number of frames to keep from the start of the video"
                }),
            },
            "optional": {
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Starting frame index (0 = from beginning)"
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Optional audio to clip in sync with video"
                }),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.01,
                    "tooltip": "Target FPS for audio synchronization (use the same FPS from video source)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("clipped_video", "clipped_audio")
    FUNCTION = "clip_frames"
    CATEGORY = "image/video"
    DESCRIPTION = "Simple node to shorten video clips by specifying frame count, with optional audio support"
    
    def clip_frames(self, source_video: torch.Tensor, frame_count: int = 30,
                   start_frame: int = 0, audio=None, fps: float = 30.0) -> Tuple[torch.Tensor, dict]:
        """
        Clip video to specified number of frames with optional audio.

        Args:
            source_video: Source video tensor [B, H, W, C]
            frame_count: Number of frames to keep
            start_frame: Starting frame index
            audio: Optional audio dict with 'waveform' and 'sample_rate'
            fps: Target FPS for audio synchronization

        Returns:
            Tuple of (clipped video tensor, clipped audio dict or None)
        """
        try:
            # Validate input
            validate_image_tensor(source_video, "source_video")

            if source_video.shape[0] == 0:
                logging.warning("Source video is empty")
                return (source_video, None)

            source_length = source_video.shape[0]

            # Ensure start_frame is within bounds
            start_frame = max(0, min(start_frame, source_length - 1))

            # Calculate end frame
            end_frame = min(start_frame + frame_count, source_length)

            # Clip the video
            clipped_video = source_video[start_frame:end_frame]

            actual_frames = clipped_video.shape[0]
            logging.info(f"Clipped video from {source_length} frames to {actual_frames} frames "
                        f"(frames {start_frame} to {end_frame-1})")

            # Clip audio if provided (using target FPS for synchronization)
            clipped_audio = None
            if audio is not None:
                try:
                    waveform = audio['waveform']
                    sample_rate = audio['sample_rate']
                    total_samples = waveform.shape[2]

                    # Calculate precise sample positions based on frame timing and target FPS
                    start_time = start_frame / fps
                    end_time = end_frame / fps

                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)

                    # Ensure we don't exceed bounds
                    start_sample = max(0, min(start_sample, total_samples))
                    end_sample = max(start_sample, min(end_sample, total_samples))

                    # Clip audio waveform
                    clipped_waveform = waveform[:, :, start_sample:end_sample]

                    clipped_audio = {
                        'waveform': clipped_waveform,
                        'sample_rate': sample_rate
                    }

                    logging.info(f"Clipped audio from {total_samples} samples to {clipped_waveform.shape[2]} samples "
                                f"(FPS: {fps}, time: {start_time:.3f}s to {end_time:.3f}s)")
                except Exception as audio_error:
                    logging.warning(f"Failed to clip audio: {str(audio_error)}")
                    clipped_audio = None

            return (clipped_video, clipped_audio)

        except Exception as e:
            logging.error(f"FrameClipper error: {str(e)}")
            raise RuntimeError(f"Frame clipping failed: {str(e)}")
