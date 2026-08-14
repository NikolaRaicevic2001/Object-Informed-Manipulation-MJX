"""Writing a run to an mp4: the ffmpeg pipe, and the offscreen renderer.

`VideoRecorder` is the encoder alone -- give it raw frames, it gives you a
file. `OffscreenRecorder` is what every MuJoCo world actually uses: it owns
a `mujoco.Renderer` with no viewer and no display, strides frames so
playback is real time, and composites the plan overlay.

Both live here rather than in a world package because all four record the
same way. `OffscreenRecorder` was in `oim.worlds.sim3d.run`, and the object-only
world imported it across that boundary to film its MuJoCo plant.
"""

import os
import subprocess
from datetime import datetime
from typing import List, Optional, Sequence, Tuple, Union

import mujoco

from oim.runtime.overlay import BlockTrace, PlanOverlay


class VideoRecorder:
    """Class for recording videos using FFmpeg."""

    def __init__(
        self,
        output_dir: str,
        width: int = 720,
        height: int = 480,
        fps: float = 30.0,
        prefix: str = "simulation",
        filename: Optional[str] = None,
    ):
        """Initialize the video recorder.

        Args:
            output_dir: Directory to save the video.
            width: Width of the video in pixels.
            height: Height of the video in pixels.
            fps: Frames per second.
            prefix: Filename prefix, before the timestamp. Defaults to
                "simulation" (the original, task-agnostic name) so existing
                callers are unaffected; pass e.g.
                "pusht3d_xarm6_shelf_gap_admm" for a name that identifies
                the task and method too.
            filename: Exact base name (no extension), used verbatim instead
                of `{prefix}_{timestamp}`. Pass the base name already
                computed for a run's plot and results JSON so all three
                files share one timestamp rather than each stamping its own.
        """
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.prefix = prefix
        self.filename = filename

        self.ffmpeg_process = None
        self.video_path = None
        self.is_recording = False

    def start(self) -> bool:
        """Start recording the video.

        Returns:
            True if recording started successfully, False otherwise.
        """
        if self.is_recording:
            print("Warning: Recording already in progress")
            return True

        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Use the caller's exact base name if given, else stamp our own.
        if self.filename is not None:
            base = self.filename
        else:
            base = f"{self.prefix}_{datetime.now():%Y%m%d_%H%M%S}"
        self.video_path = os.path.join(self.output_dir, f"{base}.mp4")

        # Check if FFmpeg is available
        try:
            # Test FFmpeg availability
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            # Set up FFmpeg process
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-f",
                "rawvideo",  # Input format
                "-vcodec",
                "rawvideo",  # Input codec
                "-s",
                f"{self.width}x{self.height}",  # Frame dimensions
                "-pix_fmt",
                "rgb24",  # Pixel format
                "-r",
                str(self.fps),  # FPS
                "-i",
                "-",  # Input from pipe
                "-an",  # No audio
                "-vcodec",
                "h264",  # H.264 codec
                "-crf",
                "1",  # Quality (1=highest)
                "-preset",
                "slow",  # Encoding speed/compression tradeoff
                "-movflags",
                "+faststart",  # Optimize for web playback
                "-pix_fmt",
                "yuv420p",  # Standard compatible format
                "-profile:v",
                "high",  # H.264 profile
                "-tune",
                "film",  # Encoding optimization
                "-loglevel",
                "error",  # Suppress output except errors
                self.video_path,
            ]

            self.ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.is_recording = True
            print(f"Recording video to {self.video_path}")
            return True

        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: FFmpeg not found. Video recording disabled.")
            self.is_recording = False
            return False

    def add_frame(self, frame: bytes) -> bool:
        """Add a frame to the video.

        Args:
            frame: Raw RGB frame data.

        Returns:
            True if the frame was added successfully, False otherwise.
        """
        if (
            not self.is_recording
            or self.ffmpeg_process is None
            or self.ffmpeg_process.stdin is None
        ):
            return False

        try:
            self.ffmpeg_process.stdin.write(frame)
            return True
        except (BrokenPipeError, IOError):
            print("Warning: Failed to write frame to video")
            self.is_recording = False
            return False

    def stop(self) -> bool:
        """Stop recording and finalize the video.

        Returns:
            True if the video was finalized successfully, False otherwise.
        """
        if not self.is_recording or self.ffmpeg_process is None:
            return False

        try:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            print(f"Video saved to {self.video_path}")
            self.is_recording = False
            return True
        except (subprocess.TimeoutExpired, BrokenPipeError, IOError) as e:
            print(f"Warning: Error finalizing video: {e}")
            # Try to terminate the process if it's still running
            try:
                self.ffmpeg_process.terminate()
            except Exception:
                pass
            self.is_recording = False
            return False


class OffscreenRecorder:
    """Renders `mujoco.MjData` to an mp4 with no viewer and no display.

    Frames are strided so playback is real time: the simulator produces
    `1/timestep` frames per simulated second (100 at the push-T timestep),
    far more than a video needs, so every `stride`-th one is kept and the
    encoder is told the matching frame rate. Recording a frame per physics
    step and encoding at the *replanning* rate -- which is what
    `run_interactive` does -- yields a video that plays back
    `sim_steps_per_replan` times slower than reality.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        output_dir: str,
        base_name: str,
        target_fps: float,
        size: Tuple[int, int],
        camera: Optional[Union[str, int]],
        overlay: Optional[PlanOverlay] = None,
    ) -> None:
        """Set up the offscreen renderer and start the encoder.

        Args:
            mj_model: The execution model to render.
            output_dir: Directory for the mp4.
            base_name: Filename stem, shared with the run's plot/results.
            target_fps: Desired playback frame rate; the achieved rate is
                the nearest one an integer stride allows.
            size: (width, height) in pixels.
            camera: Model camera name or id. None uses the default free
                camera, which frames the whole scene.
            overlay: If given, the ADMM plan overlay to composite into every
                frame. Feed it with `set_plans` once per control step.

        Raises:
            RuntimeError: If no OpenGL backend is available.
        """
        width, height = size
        step_fps = 1.0 / mj_model.opt.timestep
        self.stride = max(1, round(step_fps / target_fps))
        self._i = 0

        # Must be set before the Renderer allocates its buffer.
        mj_model.vis.global_.offwidth = max(
            width, mj_model.vis.global_.offwidth
        )
        mj_model.vis.global_.offheight = max(
            height, mj_model.vis.global_.offheight
        )
        try:
            self.renderer = mujoco.Renderer(
                mj_model, height=height, width=width
            )
        except Exception as e:  # no GL context available
            raise RuntimeError(
                f"Could not create an offscreen renderer ({e}). On a machine "
                "with no display, set MUJOCO_GL=egl (or osmesa for software "
                "rendering) before running."
            ) from e

        self.camera = mujoco.MjvCamera()
        if camera is None:
            mujoco.mjv_defaultFreeCamera(mj_model, self.camera)
        else:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = (
                camera
                if isinstance(camera, int)
                else mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera
                )
            )
            if self.camera.fixedcamid < 0:
                raise ValueError(f"No camera named {camera!r} in the model")

        self.recorder = VideoRecorder(
            output_dir=output_dir,
            width=width,
            height=height,
            fps=step_fps / self.stride,
            filename=base_name,
        )
        self.active = self.recorder.start()
        self.overlay = overlay
        self._traces: List[BlockTrace] = []

    def set_plans(self, traces: Sequence[BlockTrace]) -> None:
        """Hold the blocks to composite into subsequent frames.

        Called once per control step, while frames are captured once per
        physics step -- so the same plans are drawn across the substeps they
        were computed for, which is exactly their period of validity.

        Args:
            traces: One `BlockTrace` per block, from
                `oim.runtime.overlay.traces_for`.
        """
        self._traces = list(traces)

    def capture(self, mj_data: mujoco.MjData) -> None:
        """Render and encode one frame, if this physics step is on-stride."""
        on_stride = self._i % self.stride == 0
        self._i += 1
        if not (self.active and on_stride):
            return
        self.renderer.update_scene(mj_data, self.camera)
        # After update_scene, not before: it rebuilds the scene from the
        # model and resets ngeom, discarding anything added earlier. This
        # is what puts the trajectories in the mp4 as well as the viewer.
        if self.overlay is not None and self._traces:
            self.overlay.draw(self.renderer.scene, self._traces)
        self.recorder.add_frame(self.renderer.render().tobytes())

    def close(self) -> None:
        """Finalize the mp4 and release the GL context."""
        if self.active:
            self.recorder.stop()
            self.active = False
        self.renderer.close()
