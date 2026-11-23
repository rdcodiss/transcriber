import unittest
from types import SimpleNamespace

from transcriber import Hotkey


class HotkeyPickerTests(unittest.TestCase):
    def test_loads_existing_function_key_and_new_mouse_buttons(self):
        self.assertEqual(Hotkey.from_setting("F9"), Hotkey(0x78, "F9"))
        for code in (1, 2, 4, 5, 6):
            mouse = Hotkey.from_setting(f"{code}:mouse")
            self.assertTrue(mouse.is_mouse)
            self.assertEqual(Hotkey.from_setting(mouse.as_setting()), mouse)

    def test_preserves_keyboard_virtual_key_and_modifier_side(self):
        self.assertEqual(Hotkey.from_keyboard(SimpleNamespace(keycode=0x4B, keysym="k")), Hotkey(0x4B, "K"))
        self.assertEqual(Hotkey.from_keyboard(SimpleNamespace(keycode=0x11, keysym="Control_R")),
                         Hotkey(0xA3, "Right Ctrl"))

    def test_ignores_non_keyboard_events_and_invalid_preferences(self):
        self.assertIsNone(Hotkey.from_keyboard(SimpleNamespace(keycode=0xE7, keysym="??")))
        for invalid in ("", "Off", "999:Bad key", "65:Text\n", "65:"):
            self.assertIsNone(Hotkey.from_setting(invalid))


if __name__ == "__main__":
    unittest.main()
