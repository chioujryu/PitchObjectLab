import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import rf_detr_canonical_output as canonical  # noqa: E402


CATEGORIES = [
    {"id": 0, "name": "football"},
    {"id": 1, "name": "player"},
]


def media_probe(
    *,
    width=100,
    height=50,
    frames=2,
    duration=0.08,
    audio=False,
    stream_index=0,
):
    return {
        "probe_backend": "fake",
        "format": {
            "name": "mov,mp4",
            "long_name": "MP4",
            "duration_seconds": duration,
            "size_bytes": 123,
            "bit_rate": 456,
        },
        "video": {
            "stream_index": stream_index,
            "codec_name": "h264",
            "codec_long_name": "H.264",
            "profile": "High",
            "decoded_width": width,
            "decoded_height": height,
            "display_width": width,
            "display_height": height,
            "pixel_format": "yuv420p",
            "fps_rational": "25/1",
            "fps_float": 25.0,
            "nominal_fps_rational": "25/1",
            "time_base": "1/12800",
            "is_variable_frame_rate": False,
            "frame_count": frames,
            "duration_seconds": duration,
            "rotation_degrees": 0.0,
        },
        "audio": {
            "has_audio": audio,
            "stream_index": 3 if audio else None,
            "codec_name": "aac" if audio else None,
            "sample_rate_hz": 48000 if audio else None,
            "channels": 2 if audio else None,
            "channel_layout": "stereo" if audio else None,
            "duration_seconds": duration if audio else None,
        },
        "probe_warnings": [],
    }


class CanonicalConfigTest(unittest.TestCase):
    def test_defaults_and_cli_overrides(self):
        cfg = canonical.parse_canonical_output_config({"inference": {}})
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.directory, "canonical_v2")
        self.assertEqual(cfg.media.video_codec, "libx264")
        self.assertEqual(cfg.media.crf, 18)

        parser = argparse.ArgumentParser()
        canonical.add_canonical_output_cli_arguments(parser)
        args = parser.parse_args(
            [
                "--no-canonical-output",
                "--canonical-output-dir",
                "vlm/canonical",
                "--ffmpeg-path",
                "/opt/ffmpeg",
                "--ffprobe-path",
                "/opt/ffprobe",
            ]
        )
        overridden = canonical.apply_canonical_output_cli_overrides(cfg, args)
        self.assertFalse(overridden.enabled)
        self.assertEqual(overridden.directory, "vlm/canonical")
        self.assertEqual(overridden.ffprobe_path, "/opt/ffprobe")

    def test_output_directory_cannot_escape_run(self):
        for directory in ("/tmp/canonical", "../canonical", "a/../../b", r"C:\outside"):
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "relative path"):
                canonical.parse_canonical_output_config({"canonical_output": {"directory": directory}})

    def test_explicit_missing_ffprobe_fails_preflight(self):
        cfg = canonical.parse_canonical_output_config(
            {
                "canonical_output": {
                    "ffmpeg_path": "auto",
                    "ffprobe_path": "/missing/ffprobe",
                }
            }
        )
        with patch.object(
            canonical.shutil,
            "which",
            side_effect=lambda value: "/usr/bin/ffmpeg" if value == "ffmpeg" else None,
        ):
            with self.assertRaisesRegex(RuntimeError, "ffprobe"):
                canonical.preflight(cfg)

    def test_missing_ffmpeg_and_missing_configured_encoder_fail_preflight(self):
        cfg = canonical.parse_canonical_output_config({})
        with patch.object(canonical.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "requires ffmpeg"):
                canonical.preflight(cfg)

        def runner(_command, **_kwargs):
            return SimpleNamespace(
                returncode=0,
                stdout=" V..... libx264\n",
                stderr="",
            )

        with patch.object(
            canonical,
            "_resolve_executable",
            side_effect=["/bin/ffmpeg", "/bin/ffprobe"],
        ):
            with self.assertRaisesRegex(RuntimeError, "aac"):
                canonical.preflight(cfg, runner=runner)

    def test_preflight_verifies_both_encoders(self):
        cfg = canonical.parse_canonical_output_config({})
        calls = []

        def runner(command, **_kwargs):
            calls.append(command)
            return SimpleNamespace(
                returncode=0,
                stdout=" V..... libx264\n A..... aac\n",
                stderr="",
            )

        with patch.object(
            canonical,
            "_resolve_executable",
            side_effect=["/bin/ffmpeg", "/bin/ffprobe"],
        ):
            tools = canonical.preflight(cfg, runner=runner)
        self.assertEqual(tools.ffmpeg_path, "/bin/ffmpeg")
        self.assertIn("-encoders", calls[0])


class CanonicalGeometryTest(unittest.TestCase):
    def test_xyxy_geometry_preserves_raw_pixels_and_clamps_normalized(self):
        geometry = canonical.geometry_from_xyxy([-10.0, 20.0, 120.0, 60.0], 100, 50)
        self.assertEqual(geometry["bbox_xyxy_pixels"], [-10.0, 20.0, 120.0, 60.0])
        self.assertEqual(geometry["bbox_xyxy_normalized"], [0.0, 0.4, 1.0, 1.0])
        self.assertEqual(geometry["center_pixels"], [55.0, 40.0])
        self.assertEqual(geometry["center_normalized"], [0.55, 0.8])
        self.assertEqual(geometry["area_pixels"], 5200.0)
        self.assertAlmostEqual(geometry["area_normalized"], 0.6)
        self.assertTrue(geometry["was_clipped"])
        self.assertTrue(geometry["in_frame"])

    def test_xywh_conversion_uses_continuous_edges(self):
        self.assertEqual(
            canonical.xywh_to_xyxy([10.25, 20.5, 5.5, 7.25]),
            [10.25, 20.5, 15.75, 27.75],
        )

    def test_legacy_detection_keeps_detector_attributes_not_tracker_fields(self):
        raw = {
            "category_id": 0,
            "score": 0.7,
            "bbox": [10.0, 20.0, 4.0, 6.0],
            "area": 24.0,
            "track_id": 8,
            "track_hits": 3,
            "track_center_x": 12.0,
            "track_status": "confirmed",
            "association_stage": "high",
            "first_stage_score": 0.6,
            "recheck_passed": True,
        }
        row = canonical.build_detection_rows_from_legacy([raw], CATEGORIES, 100, 50)[0]
        self.assertEqual(row["bbox_xyxy_pixels"], [10.0, 20.0, 14.0, 26.0])
        self.assertEqual(
            row["attributes"],
            {"first_stage_score": 0.6, "recheck_passed": True},
        )

    def test_center_only_predicted_state_has_null_bbox_not_fake_box(self):
        states = canonical.build_tracker_state_rows(
            [
                {
                    "track_id": 4,
                    "observation": "predicted",
                    "bbox": None,
                    "center": [105.0, 25.0],
                    "detection_index": None,
                }
            ],
            100,
            50,
        )
        self.assertEqual(states[0]["center_pixels"], [105.0, 25.0])
        self.assertEqual(states[0]["center_normalized"], [1.0, 0.5])
        self.assertIsNone(states[0]["bbox_xyxy_pixels"])
        self.assertIsNone(states[0]["area_pixels"])
        self.assertNotIn("score", states[0])

    def test_hybrid_observed_state_is_remapped_by_track_id(self):
        detections = [
            {"category_id": 1, "bbox": [0, 0, 10, 20], "track_id": None},
            {"category_id": 0, "bbox": [10, 10, 4, 4], "track_id": 7},
        ]
        states = canonical.build_tracker_state_rows(
            [
                {
                    "track_id": 7,
                    "observation": "observed",
                    "bbox": [10, 10, 4, 4],
                    "center": [12, 12],
                    "association": {"detection_index": 0},
                }
            ],
            100,
            50,
            detections=detections,
            categories=CATEGORIES,
        )
        self.assertEqual(states[0]["detection_index"], 1)
        self.assertEqual(states[0]["category_id"], 0)

    def test_camera_motion_accepts_integration_affine_key(self):
        motion = canonical.normalize_camera_motion(
            {
                "affine_2x3_pixels": [[0.0, -2.0, 3.0], [2.0, 0.0, 4.0]],
                "success": True,
                "inliers": 42,
            }
        )
        self.assertEqual(motion["translation_pixels"], [3.0, 4.0])
        self.assertEqual(motion["scale"], 2.0)
        self.assertEqual(motion["rotation_clockwise_degrees"], 90.0)
        self.assertEqual(motion["inliers"], 42)


class CanonicalTrajectoryTest(unittest.TestCase):
    @staticmethod
    def state(center_x, *, provenance="observed", detection_index=0):
        bbox = [center_x - 2.0, 23.0, center_x + 2.0, 27.0] if provenance == "observed" else None
        return canonical.build_track_state_row(
            track_id=1,
            provenance=provenance,
            bbox_xyxy=bbox,
            frame_width=100,
            frame_height=50,
            center_pixels=[center_x, 25.0],
            detection_index=(detection_index if provenance == "observed" else None),
            category_id=0,
            category_name="football",
        )

    def test_real_dt_motion_uses_unclamped_signed_normalization(self):
        accumulator = canonical.TrajectoryAccumulator(100, 50)
        first = self.state(90.0)
        second = self.state(110.0)
        third = self.state(130.0)
        motion0 = accumulator.add_state(
            first,
            segment_frame_index=0,
            source_frame_index=20,
            source_timestamp_seconds=10.0,
            segment_timestamp_seconds=0.0,
            detector_score=0.5,
        )
        motion1 = accumulator.add_state(
            second,
            segment_frame_index=1,
            source_frame_index=21,
            source_timestamp_seconds=10.5,
            segment_timestamp_seconds=0.5,
            detector_score=0.7,
        )
        motion2 = accumulator.add_state(
            third,
            segment_frame_index=2,
            source_frame_index=22,
            source_timestamp_seconds=11.0,
            segment_timestamp_seconds=1.0,
            detector_score=0.9,
        )
        self.assertIsNone(motion0["velocity_pixels_per_second"])
        self.assertEqual(motion1["velocity_pixels_per_second"], [40.0, 0.0])
        self.assertEqual(motion1["velocity_normalized_per_second"], [0.4, 0.0])
        self.assertEqual(motion1["direction_8way"], "east")
        self.assertEqual(motion2["acceleration_pixels_per_second_squared"], [0.0, 0.0])
        track = accumulator.track_rows("video")[0]
        self.assertEqual(track["path_length_pixels"], 40.0)
        self.assertAlmostEqual(track["path_length_normalized"], 0.4)
        self.assertEqual(track["net_displacement_pixels"], 40.0)
        self.assertEqual(track["net_displacement_vector_normalized"], [0.4, 0.0])
        self.assertEqual(
            [point["detector_score"] for point in track["points"]],
            [0.5, 0.7, 0.9],
        )

    def test_gap_nulls_motion_but_records_gap_and_real_seconds_since_observed(self):
        accumulator = canonical.TrajectoryAccumulator(100, 50)
        observed = self.state(10.0)
        accumulator.add_state(
            observed,
            segment_frame_index=0,
            source_frame_index=100,
            source_timestamp_seconds=5.0,
            segment_timestamp_seconds=0.0,
            detector_score=0.8,
        )
        predicted = self.state(20.0, provenance="predicted")
        motion = accumulator.add_state(
            predicted,
            segment_frame_index=2,
            source_frame_index=102,
            source_timestamp_seconds=5.25,
            segment_timestamp_seconds=0.25,
            detector_score=None,
        )
        self.assertTrue(all(value is None for value in motion.values()))
        self.assertEqual(predicted["seconds_since_observed"], 0.25)
        track = accumulator.track_rows("video")[0]
        self.assertEqual(track["max_gap_frames"], 1)
        self.assertEqual(track["max_gap_seconds"], 0.25)
        self.assertIsNone(track["points"][1]["detector_score"])

    def test_tracker_capabilities_distinguish_center_and_bbox_prediction(self):
        circle = canonical.tracker_capabilities("circle")
        hybrid = canonical.tracker_capabilities("hybrid")
        self.assertTrue(circle["predicted_center"])
        self.assertFalse(circle["predicted_bbox"])
        self.assertTrue(hybrid["predicted_bbox"])
        for algorithm in ("ocsort", "deepocsort", "botsort", "bytetrack"):
            with self.subTest(algorithm=algorithm):
                boxmot = canonical.tracker_capabilities(algorithm)
                self.assertTrue(boxmot["observed_track_states"])
                self.assertFalse(boxmot["predicted_track_states"])
                self.assertFalse(boxmot["predicted_center"])
                self.assertFalse(boxmot["predicted_bbox"])


class CanonicalMediaTest(unittest.TestCase):
    def test_video_id_is_stable_and_range_scoped(self):
        first = canonical.make_video_id("/videos/match.mp4", canonical.VideoSelection(10, 20, 1.0, 2.0))
        again = canonical.make_video_id("/videos/match.mp4", canonical.VideoSelection(10, 20, 1.0, 2.0))
        other = canonical.make_video_id("/videos/match.mp4", canonical.VideoSelection(20, 30, 2.0, 3.0))
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_ffprobe_source_is_fast_but_output_can_count_frames(self):
        commands = []
        payload = {
            "streams": [
                {
                    "index": 2,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30000/1001",
                    "r_frame_rate": "30/1",
                    "time_base": "1/90000",
                    "nb_frames": "10",
                    "tags": {"rotate": "90"},
                    "disposition": {"default": 1},
                },
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "mp3",
                    "sample_rate": "44100",
                    "channels": 2,
                    "disposition": {"default": 0},
                },
                {
                    "index": 3,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "48000",
                    "channels": 2,
                    "disposition": {"default": 1},
                },
            ],
            "format": {
                "format_name": "mov,mp4",
                "duration": "1.0",
                "size": "100",
            },
        }

        def runner(command, **_kwargs):
            commands.append(command)
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

        with patch.object(canonical, "_resolve_executable", return_value="/bin/ffprobe"):
            probe = canonical.probe_media(
                "source.mp4",
                ffprobe_path="/bin/ffprobe",
                runner=runner,
            )
            canonical.probe_media(
                "media.mp4",
                ffprobe_path="/bin/ffprobe",
                count_frames=True,
                runner=runner,
            )
        self.assertNotIn("-count_frames", commands[0])
        self.assertIn("-count_frames", commands[1])
        self.assertEqual(probe["video"]["decoded_width"], 1920)
        self.assertEqual(probe["video"]["display_width"], 1080)
        self.assertTrue(probe["video"]["is_variable_frame_rate"])
        self.assertTrue(probe["audio"]["has_audio"])
        self.assertEqual(probe["audio"]["stream_index"], 3)

    def test_transcode_uses_selected_streams_real_times_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "media.mp4"
            commands = []

            def runner(command, **_kwargs):
                commands.append(command)
                Path(command[-1]).write_bytes(b"mp4")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            output_probe = media_probe(
                width=100,
                height=50,
                frames=60,
                duration=2.0,
                audio=True,
            )
            probe_calls = []

            def probe_fn(_source, **kwargs):
                probe_calls.append(kwargs)
                return output_probe

            canonical.transcode_clean_media(
                "source.mp4",
                output,
                canonical.VideoSelection(30, 90, 1.0, 3.0),
                canonical.CanonicalMediaConfig(),
                canonical.CanonicalToolchain("/bin/ffmpeg", "/bin/ffprobe", "libx264", "aac"),
                expected_frames=60,
                expected_width=100,
                expected_height=50,
                source_probe=media_probe(
                    frames=300,
                    duration=10.0,
                    audio=True,
                    stream_index=2,
                ),
                runner=runner,
                probe_fn=probe_fn,
            )
            filter_graph = commands[0][commands[0].index("-filter_complex") + 1]
            self.assertIn("[0:2]trim=start_frame=30:end_frame=90", filter_graph)
            self.assertIn("[0:3]atrim=start=1.000000000000", filter_graph)
            self.assertIn("apad,atrim=duration=2.000000000000", filter_graph)
            self.assertNotIn("whole_dur", filter_graph)
            self.assertTrue(probe_calls[0]["count_frames"])

    def test_transcode_pads_audio_shorter_than_selected_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "media.mp4"
            commands = []

            def runner(command, **_kwargs):
                commands.append(command)
                Path(command[-1]).write_bytes(b"mp4")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            source_probe = media_probe(frames=90, duration=3.0, audio=True)
            source_probe["audio"]["duration_seconds"] = 2.8
            result = canonical.transcode_clean_media(
                "source.mp4",
                output,
                canonical.VideoSelection(0, 90, 0.0, 3.0),
                canonical.CanonicalMediaConfig(),
                canonical.CanonicalToolchain("/bin/ffmpeg", "/bin/ffprobe", "libx264", "aac"),
                expected_frames=90,
                expected_width=100,
                expected_height=50,
                source_probe=source_probe,
                runner=runner,
                probe_fn=lambda *_args, **_kwargs: media_probe(
                    frames=90,
                    duration=3.0,
                    audio=True,
                ),
            )
            filter_graph = commands[0][commands[0].index("-filter_complex") + 1]
            self.assertIn("apad,atrim=duration=3.000000000000", filter_graph)
            self.assertNotIn("whole_dur", filter_graph)
            self.assertEqual(result["video"]["frame_count"], 90)
            self.assertEqual(result["audio"]["duration_seconds"], 3.0)

    def test_transcode_without_audio_uses_an_and_codec_failure_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "media.mp4"
            commands = []

            def runner(command, **_kwargs):
                commands.append(command)
                Path(command[-1]).write_bytes(b"mp4")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            canonical.transcode_clean_media(
                "source.mp4",
                output,
                canonical.VideoSelection(0, 50, 0.0, 2.0),
                canonical.CanonicalMediaConfig(),
                canonical.CanonicalToolchain("/bin/ffmpeg", "/bin/ffprobe", "libx264", "aac"),
                expected_frames=50,
                expected_width=100,
                expected_height=50,
                source_probe=media_probe(frames=250, duration=10.0, audio=False),
                runner=runner,
                probe_fn=lambda *_args, **_kwargs: media_probe(frames=50, duration=2.0, audio=False),
            )
            self.assertIn("-an", commands[0])
            self.assertNotIn("[a]", commands[0])
            self.assertNotIn("-c:a", commands[0])
            filter_graph = commands[0][commands[0].index("-filter_complex") + 1]
            self.assertNotIn("apad", filter_graph)

            def failing_runner(_command, **_kwargs):
                return SimpleNamespace(returncode=1, stdout="", stderr="unknown encoder")

            with self.assertRaisesRegex(RuntimeError, "unknown encoder"):
                canonical.transcode_clean_media(
                    "source.mp4",
                    Path(tmp) / "failed.mp4",
                    canonical.VideoSelection(0, 50, 0.0, 2.0),
                    canonical.CanonicalMediaConfig(),
                    canonical.CanonicalToolchain("/bin/ffmpeg", "/bin/ffprobe", "libx264", "aac"),
                    expected_frames=50,
                    source_probe=media_probe(frames=250, duration=10.0, audio=False),
                    runner=failing_runner,
                )

    def test_clean_media_rejects_audio_video_drift_beyond_one_frame(self):
        probe = media_probe(frames=50, duration=2.0, audio=True)
        probe["audio"]["duration_seconds"] = 2.2
        with self.assertRaisesRegex(
            RuntimeError,
            r"audio/video duration.*video=2\.000000s, audio=2\.200000s, "
            r"difference=0\.200000s, tolerance=0\.041000s",
        ):
            canonical._validate_clean_media(
                probe,
                expected_frames=50,
                expected_width=100,
                expected_height=50,
                expected_audio=True,
                expected_duration=2.0,
            )

    def test_transcode_rejects_wrong_selected_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "media.mp4"

            def runner(command, **_kwargs):
                Path(command[-1]).write_bytes(b"mp4")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with self.assertRaisesRegex(RuntimeError, "selected duration"):
                canonical.transcode_clean_media(
                    "source.mp4",
                    output,
                    canonical.VideoSelection(0, 50, 0.0, 2.0),
                    canonical.CanonicalMediaConfig(),
                    canonical.CanonicalToolchain("/bin/ffmpeg", "/bin/ffprobe", "libx264", "aac"),
                    expected_frames=50,
                    expected_width=100,
                    expected_height=50,
                    source_probe=media_probe(frames=250, duration=10.0, audio=False),
                    runner=runner,
                    probe_fn=lambda *_args, **_kwargs: media_probe(frames=50, duration=3.0, audio=False),
                )

    def test_real_ffmpeg_pads_short_audio_to_video_duration(self):
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            self.skipTest("FFmpeg and FFprobe are required for the real-media regression test")
        try:
            toolchain = canonical.preflight(
                canonical.CanonicalOutputConfig(
                    ffmpeg_path=ffmpeg,
                    ffprobe_path=ffprobe,
                )
            )
        except RuntimeError as exc:
            self.skipTest(f"Canonical FFmpeg encoders are unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "short-audio-source.mp4"
            output = Path(tmp) / "media.mp4"
            fixture_command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=30:duration=3",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=1000:sample_rate=48000:duration=2.8",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(source),
            ]
            fixture_result = subprocess.run(
                fixture_command,
                capture_output=True,
                text=True,
                check=False,
            )
            if fixture_result.returncode != 0:
                self.skipTest(f"FFmpeg lavfi fixture is unavailable: {fixture_result.stderr.strip()}")

            source_probe = canonical.probe_media(
                source,
                ffprobe_path=toolchain.ffprobe_path or "auto",
                ffmpeg_path=toolchain.ffmpeg_path or "auto",
            )
            self.assertAlmostEqual(source_probe["video"]["duration_seconds"], 3.0, places=3)
            self.assertAlmostEqual(source_probe["audio"]["duration_seconds"], 2.8, places=3)
            output_probe = canonical.transcode_clean_media(
                source,
                output,
                canonical.VideoSelection(0, 90, 0.0, 3.0),
                canonical.CanonicalMediaConfig(preset="ultrafast"),
                toolchain,
                expected_frames=90,
                expected_width=160,
                expected_height=90,
                source_probe=source_probe,
            )
            self.assertEqual(output_probe["video"]["frame_count"], 90)
            self.assertAlmostEqual(output_probe["video"]["duration_seconds"], 3.0, places=3)
            self.assertAlmostEqual(output_probe["audio"]["duration_seconds"], 3.0, places=3)

    def test_real_ffmpeg_repairs_compressed_audio_pts_drift(self):
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            self.skipTest("FFmpeg and FFprobe are required for the real-media regression test")
        try:
            toolchain = canonical.preflight(
                canonical.CanonicalOutputConfig(
                    ffmpeg_path=ffmpeg,
                    ffprobe_path=ffprobe,
                )
            )
        except RuntimeError as exc:
            self.skipTest(f"Canonical FFmpeg encoders are unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "compressed-audio-pts-source.mp4"
            output = Path(tmp) / "media.mp4"
            fixture_command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=30:duration=3",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=1000:sample_rate=48000:duration=3",
                "-filter_complex",
                "[1:a:0]asetpts=PTS*0.95[a]",
                "-map",
                "0:v:0",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(source),
            ]
            fixture_result = subprocess.run(
                fixture_command,
                capture_output=True,
                text=True,
                check=False,
            )
            if fixture_result.returncode != 0:
                self.skipTest(f"FFmpeg lavfi fixture is unavailable: {fixture_result.stderr.strip()}")

            source_probe = canonical.probe_media(
                source,
                ffprobe_path=toolchain.ffprobe_path or "auto",
                ffmpeg_path=toolchain.ffmpeg_path or "auto",
            )
            one_frame_tolerance = 1.0 / 30.0 + 1.0e-3
            self.assertEqual(source_probe["video"]["frame_count"], 90)
            self.assertAlmostEqual(
                source_probe["audio"]["duration_seconds"],
                2.85,
                delta=one_frame_tolerance,
            )
            output_probe = canonical.transcode_clean_media(
                source,
                output,
                canonical.VideoSelection(0, 90, 0.0, 3.0),
                canonical.CanonicalMediaConfig(preset="ultrafast"),
                toolchain,
                expected_frames=90,
                expected_width=160,
                expected_height=90,
                source_probe=source_probe,
            )
            video_duration = output_probe["video"]["duration_seconds"]
            audio_duration = output_probe["audio"]["duration_seconds"]
            self.assertEqual(output_probe["video"]["frame_count"], 90)
            self.assertAlmostEqual(video_duration, 3.0, delta=one_frame_tolerance)
            self.assertAlmostEqual(audio_duration, 3.0, delta=one_frame_tolerance)
            self.assertLessEqual(abs(video_duration - audio_duration), one_frame_tolerance)


class CanonicalWriterTest(unittest.TestCase):
    def make_run(self, root):
        cfg = canonical.parse_canonical_output_config({})

        def probe_fn(_source, **_kwargs):
            return media_probe(frames=2, duration=0.08, audio=True)

        def transcode_fn(
            _source,
            output,
            _selection,
            _media_cfg,
            _toolchain,
            **_kwargs,
        ):
            Path(output).write_bytes(b"clean mp4")
            return media_probe(frames=2, duration=0.08, audio=True)

        return canonical.CanonicalRunWriter(
            root,
            cfg,
            CATEGORIES,
            {
                "name": "rf_detr_inference",
                "detector": {
                    "checkpoint": "model.pth",
                    "config": "merged_config.yaml",
                },
            },
            toolchain=canonical.CanonicalToolchain("/bin/ffmpeg", "/bin/ffprobe", "libx264", "aac"),
            probe_fn=probe_fn,
            transcode_fn=transcode_fn,
        )

    def test_stream_bundle_is_atomic_complete_and_manifested(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp)
            writer = run.start_video(
                source="match.mp4",
                source_path="/cache/match.mp4",
                width=100,
                height=50,
                input_fps=25.0,
                source_frame_count=1000,
                frame_window={
                    "start_frame": 10,
                    "end_frame": 12,
                    "start_seconds": 5.0,
                    "effective_end_seconds": 5.08,
                },
                detection_fps=25.0,
                frame_interval=1,
                output_fps=12.5,
                tracking_config={
                    "enabled": True,
                    "algorithm": "hybrid",
                    "hybrid_options": {"hypothesis": {"lookahead_seconds": 0.5}},
                },
            )
            first = writer.write_frame(
                0,
                10,
                source_timestamp_seconds=5.125,
                timestamp_source="decoder_pts",
                detections=[
                    {
                        "category_id": 1,
                        "score": 0.8,
                        "bbox": [0, 0, 10, 20],
                    },
                    {
                        "category_id": 0,
                        "score": 0.9,
                        "bbox": [10, 10, 4, 4],
                        "track_id": 7,
                        "track_hits": 2,
                        "track_status": "confirmed",
                        "recheck_passed": True,
                    },
                ],
                track_states=[
                    {
                        "track_id": 7,
                        "observation": "observed",
                        "bbox": [10, 10, 4, 4],
                        "center": [12, 12],
                        "hits": 2,
                        "status": "confirmed",
                        "association": {"detection_index": 0},
                    }
                ],
                camera_motion={
                    "affine_2x3_pixels": [[1, 0, 2], [0, 1, 3]],
                    "success": True,
                    "inliers": 20,
                },
            )
            second = writer.write_frame(
                1,
                11,
                source_timestamp_seconds=5.165,
                timestamp_source="decoder_pts",
                detection_ran=False,
                detections=[],
                track_states=[
                    {
                        "track_id": 7,
                        "observation": "predicted",
                        "bbox": None,
                        "center": [14, 12],
                        "hits": 2,
                        "status": "lost",
                    }
                ],
            )
            self.assertEqual(first["segment_timestamp_seconds"], 0.0)
            self.assertAlmostEqual(second["segment_timestamp_seconds"], 0.04)
            self.assertEqual(first["track_states"][0]["detection_index"], 1)
            self.assertNotIn("track_id", first["detections"][1]["attributes"])
            self.assertIsNone(second["track_states"][0]["bbox_xyxy_pixels"])
            self.assertAlmostEqual(second["track_states"][0]["seconds_since_observed"], 0.04)

            summary = writer.finalize(
                {"actual_decoded_frame_count": 2},
                annotated_output="videos/match_pred.mp4",
            )
            bundle = Path(tmp) / "canonical_v2" / summary["video_id"]
            self.assertFalse((Path(tmp) / "canonical_v2" / f"{summary['video_id']}.partial").exists())
            for name in ("metadata.json", "frames.jsonl", "tracks.jsonl", "media.mp4"):
                self.assertTrue((bundle / name).is_file(), name)

            frame_rows = [
                json.loads(line) for line in (bundle / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(frame_rows), 2)
            self.assertFalse(frame_rows[1]["detection_ran"])
            track = json.loads((bundle / "tracks.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(track["observed_point_count"], 1)
            self.assertEqual(track["predicted_point_count"], 1)
            self.assertEqual(
                [point["detector_score"] for point in track["points"]],
                [0.9, None],
            )

            metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], "2.0.0")
            self.assertEqual(metadata["selected_segment"]["start_seconds"], 5.125)
            self.assertAlmostEqual(metadata["selected_segment"]["end_seconds"], 5.205)
            self.assertEqual(
                metadata["selected_segment"]["requested"]["start_seconds"],
                5.0,
            )
            self.assertEqual(metadata["tracker"]["offline"]["lookahead_seconds"], 0.5)
            self.assertTrue(metadata["tracker"]["offline"]["confirmation_backfill"])
            self.assertEqual(metadata["processing"]["annotated_output_fps"], 12.5)

            manifest = run.finish_manifest()
            self.assertEqual(manifest["video_count"], 1)
            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(manifest["media_count"], 1)
            self.assertTrue((Path(tmp) / "canonical_v2" / "manifest.json").is_file())

    def test_multi_source_manifest_paths_and_track_ids_are_video_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp)
            summaries = []
            for source in ("first.mp4", "second.mp4"):
                writer = run.start_video(
                    source=source,
                    width=100,
                    height=50,
                    input_fps=25.0,
                    frame_window={"start_frame": 0, "end_frame": 2},
                    tracking_config={"enabled": True, "algorithm": "circle"},
                )
                for frame_index in range(2):
                    writer.write_frame(
                        frame_index,
                        frame_index,
                        source_timestamp_seconds=frame_index / 25.0,
                        detections=[
                            {
                                "category_id": 0,
                                "score": 0.9,
                                "bbox": [10 + frame_index, 10, 4, 4],
                                "track_id": 1,
                            }
                        ],
                    )
                summaries.append(writer.finalize())

            manifest = run.finish_manifest()
            self.assertEqual(manifest["video_count"], 2)
            self.assertEqual(len({summary["video_id"] for summary in summaries}), 2)
            for summary in summaries:
                self.assertFalse(Path(summary["bundle_path"]).is_absolute())
                bundle = Path(tmp) / "canonical_v2" / summary["bundle_path"]
                track = json.loads((bundle / "tracks.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(track["track_id"], 1)
                self.assertEqual(track["scope"], "video_id")
                self.assertEqual(track["video_id"], summary["video_id"])

    def test_missing_track_state_frame_nulls_later_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp)
            writer = run.start_video(
                source="gap.mp4",
                width=100,
                height=50,
                input_fps=25.0,
                frame_window={"start_frame": 0, "end_frame": 3},
            )
            detection = {
                "category_id": 0,
                "score": 0.8,
                "bbox": [10, 10, 4, 4],
                "track_id": 1,
            }
            writer.write_frame(0, 0, source_timestamp_seconds=0.0, detections=[detection])
            writer.write_frame(
                1,
                1,
                source_timestamp_seconds=0.04,
                detections=[],
                track_states=[],
            )
            row = writer.write_frame(
                2,
                2,
                source_timestamp_seconds=0.08,
                detections=[{**detection, "bbox": [20, 10, 4, 4]}],
            )
            self.assertTrue(all(value is None for value in row["track_states"][0]["motion"].values()))
            writer.abort()

    def test_abort_removes_partial_and_manifest_rejects_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp)
            writer = run.start_video(
                source="abort.mp4",
                width=100,
                height=50,
                input_fps=25.0,
                frame_window={"start_frame": 0, "end_frame": 1},
            )
            partial = writer.partial_path
            self.assertTrue(partial.is_dir())
            with self.assertRaisesRegex(RuntimeError, "active"):
                run.finish_manifest()
            writer.abort()
            self.assertFalse(partial.exists())
            self.assertEqual(run.finish_manifest()["video_count"], 0)

    def test_source_frame_indices_must_be_contiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp)
            writer = run.start_video(
                source="indices.mp4",
                width=100,
                height=50,
                input_fps=25.0,
                frame_window={"start_frame": 10, "end_frame": 13},
            )
            writer.write_frame(0, 10, source_timestamp_seconds=1.0)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                writer.write_frame(1, 12, source_timestamp_seconds=1.08)
            writer.abort()


if __name__ == "__main__":
    unittest.main()
