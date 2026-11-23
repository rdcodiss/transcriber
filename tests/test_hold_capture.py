import threading
import unittest
from unittest.mock import Mock, patch

from transcriber import PROVIDER_DEFAULTS, Recording, TranscriptionAPI


class HoldCaptureTests(unittest.TestCase):
    def test_fast_repeated_holds_capture_separate_chunks_without_reading_during_release(self):
        api = TranscriptionAPI.from_settings({
            **PROVIDER_DEFAULTS,
            "TRANSCRIPTION_MODE": "http",
            "TRANSCRIPTION_URL": "https://example.invalid/audio",
        })
        recording = Recording(api, hold_to_listen=True)
        gate = recording.hold_gate

        class Stream:
            holds = 0
            reads = 0
            active = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def start_stream(self):
                self.active = True
                self.holds += 1

            def stop_stream(self):
                self.active = False

            def is_active(self):
                return self.active

            def get_read_available(self):
                if not self.active:
                    raise AssertionError("Requested data with the key released")
                return 100 if self.reads < self.holds else 0

            def read(self, frames):
                # Report a packet of a different amplitude per hold.
                self.reads += 1
                return bytes((self.holds, 0)) * frames

        stream = Stream()
        audio = Mock()
        audio.get_default_wasapi_loopback.return_value = {
            "defaultSampleRate": 16_000, "maxInputChannels": 1, "index": 7,
        }
        audio.open.return_value = stream
        context = Mock(__enter__=Mock(return_value=audio), __exit__=Mock(return_value=False))
        with patch("transcriber.pyaudio.PyAudio", return_value=context):
            worker = threading.Thread(target=recording.capture, daemon=True)
            worker.start()
            try:
                for value in (1, 2):
                    while True:
                        event, _ = recording.events.get(timeout=2)
                        if event == "level":
                            break
                    gate.clear()
                    self.assertEqual(recording.audio.get(timeout=2)[0], bytes((value, 0)) * 100)
                    if value == 1:
                        gate.set()
                self.assertTrue(recording.audio.empty())
            finally:
                recording.stop_requested.set()
                worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(audio.open.call_args.kwargs["start"])
        self.assertEqual(stream.reads, 2)


if __name__ == "__main__":
    unittest.main()
