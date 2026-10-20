"""Offline tests: no real photos, Box credentials, or network calls."""
import unittest
from unittest.mock import patch
from avatar.box_storage import (
    BoxStorageError, new_participant_id, upload_study_image,
)


class BoxStorageTests(unittest.TestCase):
    def test_ids_are_random_and_valid(self):
        a, b = new_participant_id(), new_participant_id()
        self.assertNotEqual(a, b)
        self.assertEqual(len(a), 34)

    def test_rejects_invalid_kind(self):
        with self.assertRaises(ValueError):
            upload_study_image(b"fake", participant_id=new_participant_id(),
                               image_kind="other", content_type="image/png")

    def test_rejects_oversize(self):
        with self.assertRaises(ValueError):
            upload_study_image(b"x" * (10 * 1024 * 1024 + 1),
                               participant_id=new_participant_id(),
                               image_kind="original", content_type="image/png")

    def test_no_config_no_upload(self):
        with patch.dict("os.environ", {"BOX_ACCESS_TOKEN": "", "BOX_STUDY_FOLDER_ID": ""}):
            with self.assertRaises(BoxStorageError):
                upload_study_image(b"synthetic", participant_id=new_participant_id(),
                                   image_kind="original", content_type="image/png")


if __name__ == "__main__":
    unittest.main()
