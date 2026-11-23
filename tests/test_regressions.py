import unittest
from unittest.mock import Mock, patch

from transcriber import PROVIDER_DEFAULTS, Recording, TranscriptionAPI, ordered_live_items


class RecordingRegressions(unittest.TestCase):
    def setUp(self):
        api = TranscriptionAPI.from_settings({
            **PROVIDER_DEFAULTS,
            "TRANSCRIPTION_MODE": "http",
            "TRANSCRIPTION_URL": "https://example.invalid/audio",
        })
        self.recording = Recording(api)

    def test_out_of_order_commits_follow_predecessors(self):
        previous = {"third": "second", "first": None, "second": "first"}
        parts = {"third": "three", "uncommitted": "four", "first": "one", "second": "two"}
        self.assertEqual(ordered_live_items(previous, parts), ["first", "second", "third", "uncommitted"])

    def test_reads_short_playback_packet_and_drains_buffer_at_stop(self):
        audio = Mock()
        audio.get_default_wasapi_loopback.return_value = {
            "defaultSampleRate": 16_000, "maxInputChannels": 1, "index": 7,
        }
        stream = Mock()
        stream.get_read_available.side_effect = [160, 1_920]

        def read(frame_count, **_kwargs):
            self.recording.stop_requested.set()
            return b"\x01\x00" * frame_count

        stream.read.side_effect = read
        audio_context = Mock(__enter__=Mock(return_value=audio), __exit__=Mock(return_value=False))
        audio.open.return_value = Mock(__enter__=Mock(return_value=stream), __exit__=Mock(return_value=False))
        with patch("transcriber.pyaudio.PyAudio", return_value=audio_context):
            self.recording.capture()

        self.assertTrue(self.recording.capture_done.is_set())
        self.assertEqual([call.args[0] for call in stream.read.call_args_list], [160, 800, 800, 320])
        self.assertEqual(self.recording.audio.get_nowait()[0], b"\x01\x00" * 2_080)

    def test_capture_failure_does_not_start_provider(self):
        with patch("transcriber.pyaudio.PyAudio", side_effect=OSError("no output")):
            self.recording.capture()
        with patch.object(self.recording, "transcribe_http") as provider:
            self.recording.transcribe()
        provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
