import unittest
from unittest.mock import patch

import httpx

from transcriber import PROVIDER_DEFAULTS, PROVIDER_PRESETS, Recording, TranscriptionAPI, TranslationAPI


def audio_api(provider):
    return TranscriptionAPI.from_settings({
        **PROVIDER_DEFAULTS, **PROVIDER_PRESETS[provider], "TRANSCRIPTION_API_KEY": "sample-audio-key",
    })


class TranslationTests(unittest.TestCase):
    def test_reuses_the_text_provider_key_and_preserves_call_sign_in_prompt(self):
        translator = TranslationAPI.for_recording(audio_api("Groq"), "Portuguese (Brazil)")

        def respond(request):
            self.assertEqual(request.url.host, "api.groq.com")
            self.assertEqual(request.headers["Authorization"], "Bearer sample-audio-key")
            self.assertIn(b"Portuguese (Brazil)", request.content)
            self.assertIn(b"AZU123, runway 27", request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "AZU123, pista 27"}}]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            self.assertEqual(translator.translate(client, "AZU123, runway 27"), "AZU123, pista 27")

    def test_audio_only_provider_uses_separate_key(self):
        api = audio_api("Deepgram")
        with patch("transcriber.read_settings", return_value={"TRANSCRIBER_TRANSLATION_API_KEY": "sample-translation-key"}):
            translator = TranslationAPI.for_recording(api, "Spanish")
        self.assertEqual(translator.headers, {"Authorization": "Bearer sample-translation-key"})
        self.assertTrue(translator.url.startswith("https://api.openai.com/"))
        self.assertIsNone(TranslationAPI.for_recording(api, ""))

    def test_translation_worker_preserves_original_on_failure(self):
        api = audio_api("OpenAI")
        recording = Recording(api, translator=TranslationAPI.for_recording(api, "Portuguese (Brazil)"))
        for text in ("First part", "Second part", "Third part"):
            recording.publish_text(text)
        recording.translation_queue.put(None)
        with patch.object(TranslationAPI, "translate", side_effect=["Primeira parte", ValueError("unavailable")]):
            recording.translate_pending()
        events = [recording.events.get_nowait() for _ in range(5)]
        self.assertEqual([text for event, text in events if event == "text"],
                         ["Primeira parte", "Second part", "Third part"])
        self.assertEqual([event for event, _ in events], ["text", "warning", "text", "text", "translation_done"])


if __name__ == "__main__":
    unittest.main()
