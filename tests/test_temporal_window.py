import unittest

from src.models.llava_onevision_summarizer import build_centered_window


class TemporalWindowTests(unittest.TestCase):
    def test_center_window_is_301_frames_at_30_fps(self):
        window = build_centered_window(
            target_frame=500,
            total_frames=1000,
            fps=30,
            duration_seconds=10,
            num_samples=16,
        )
        self.assertEqual((window.start_frame, window.end_frame), (350, 650))
        self.assertEqual(len(window.sampled_indices), 16)
        self.assertEqual(window.sampled_indices[0], 350)
        self.assertEqual(window.sampled_indices[-1], 650)

    def test_first_frame_clips_without_underflow(self):
        window = build_centered_window(
            target_frame=0, total_frames=100, fps=30, num_samples=16
        )
        self.assertEqual(window.start_frame, 0)
        self.assertEqual(window.end_frame, 99)
        self.assertTrue(all(0 <= index < 100 for index in window.sampled_indices))

    def test_last_frame_clips_without_overflow(self):
        window = build_centered_window(
            target_frame=99, total_frames=100, fps=30, num_samples=16
        )
        self.assertEqual(window.start_frame, 0)
        self.assertEqual(window.end_frame, 99)
        self.assertEqual(window.sampled_indices[-1], 99)

    def test_reported_fps_controls_timestamp_window(self):
        window = build_centered_window(
            target_frame=250, total_frames=500, fps=25, duration_seconds=10, num_samples=8
        )
        self.assertEqual((window.start_frame, window.end_frame), (125, 375))


if __name__ == "__main__":
    unittest.main()
