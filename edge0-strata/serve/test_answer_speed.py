"""The one definition of answer speed in the server: produced tokens over the engine's decode time, never over a time that
holds the prompt being read (no GPU or model pack).

    python -m unittest serve.test_answer_speed -v
"""
import unittest

from serve.server import answer_speed, request_timings


class AnswerSpeed(unittest.TestCase):
    # A real request from the owner PC (Gyro-S, MTP): 308 prompt tokens read in 1212 ms, 73 generated in 1108 ms.
    REAL = {"generated": 73, "decode_ms": 1108.0, "prompt_ms": 1212.0, "drafts_accepted": 45, "drafts_offered": 68}

    def test_the_engines_decode_time_is_the_denominator(self):
        self.assertEqual(answer_speed(self.REAL, 73, 2.3), 65.9)

    def test_it_matches_the_timings_and_metrics_figure(self):
        self.assertEqual(answer_speed(self.REAL, 73), request_timings(308, 73, self.REAL)["predicted_per_second"])

    def test_prompt_reading_time_is_not_in_it(self):
        wall = (1212.0 + 1108.0) / 1000
        self.assertLess(round(73 / wall, 1), 40)          # the number a wall-clock division would show
        self.assertEqual(answer_speed(self.REAL, 73, wall), 65.9)

    def test_drafts_are_not_counted(self):
        # 68 drafted, 45 accepted: the engine's count (73) is what the model kept; the speed never uses the draft figures.
        self.assertEqual(answer_speed({**self.REAL, "drafts_offered": 500}, 73), 65.9)

    def test_a_one_token_reply_has_no_absurd_rate(self):
        self.assertIsNone(answer_speed({"generated": 1, "decode_ms": 0.8}, 1, None))     # a 1-token probe: 1 / 0.8 ms was 1,250 tok/s
        self.assertIsNone(answer_speed({}, 1, 1e-6))
        self.assertIsNone(answer_speed({}, 1, None))

    def test_without_a_done_line_the_server_clock_from_the_first_token_is_used(self):
        self.assertEqual(answer_speed({}, 50, 2.0), 25.0)


if __name__ == "__main__":
    unittest.main()
