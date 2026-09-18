from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config_store
import profiles
from app_data import user_data_dir


class UserDataPathTests(unittest.TestCase):
    def test_environment_override_controls_user_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"DELTA_EXIT_ASSISTANT_DATA_DIR": tmp}):
                self.assertEqual(user_data_dir(), Path(tmp))

    def test_config_is_saved_under_user_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("config_store.user_data_dir", return_value=root):
                config_store.save_config({"selected_profile_id": "delta"})
                self.assertEqual(config_store.load_config()["selected_profile_id"], "delta")
                self.assertTrue((root / "config.json").exists())

    def test_bundled_profiles_load_without_user_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("profiles.user_data_dir", return_value=Path(tmp)):
                profile_ids = {profile.id for profile in profiles.load_profiles_from_assets()}
        self.assertTrue({"delta", "valorant"}.issubset(profile_ids))

    def test_user_template_overrides_bundled_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            override = root / "assets" / "templates" / "delta" / "success.png"
            override.parent.mkdir(parents=True)
            override.write_bytes(b"custom")
            with patch("profiles.user_data_dir", return_value=root):
                resolved = Path(profiles.resolve_resource_path("assets/templates/delta/success.png"))
        self.assertEqual(resolved, override)


if __name__ == "__main__":
    unittest.main()
