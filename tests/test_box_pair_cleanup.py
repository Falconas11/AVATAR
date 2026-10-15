"""Offline mocked Box image pair tests; no real images or network."""
import unittest
from unittest.mock import patch, ANY
from avatar.box_integration import store_pair
from avatar.box_storage import BoxUploadResult

PID = "P-" + "a" * 32

class PairTests(unittest.TestCase):
    def run_pair(self):
        return store_pair(original=b"synthetic", aged=b"synthetic",
                          original_content_type="image/jpeg", aged_content_type="image/png",
                          token_provider=object(), participant_id=PID)

    @patch("avatar.box_integration.delete_study_image")
    @patch("avatar.box_integration.upload_study_image")
    def test_success(self, upload, delete):
        upload.side_effect = [
            BoxUploadResult("101", PID, "original"),
            BoxUploadResult("102", PID, "aged")]
        pair = self.run_pair()
        self.assertEqual(pair.participant_id, PID)
        self.assertEqual(pair.aged.file_id, "102")
        delete.assert_not_called()

    @patch("avatar.box_integration.delete_study_image")
    @patch("avatar.box_integration.upload_study_image")
    def test_second_failure_cleanup(self, upload, delete):
        upload.side_effect = [BoxUploadResult("101", PID, "original"), RuntimeError("upload failed")]
        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            self.run_pair()
        delete.assert_called_once_with("101", token_provider=ANY)

    @patch("avatar.box_integration.delete_study_image")
    @patch("avatar.box_integration.upload_study_image")
    def test_cleanup_failure_reported(self, upload, delete):
        upload.side_effect = [BoxUploadResult("101", PID, "original"), RuntimeError("upload failed")]
        delete.side_effect = RuntimeError("cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "Manual reconciliation required"):
            self.run_pair()

    @patch("avatar.box_integration.delete_study_image")
    @patch("avatar.box_integration.upload_study_image")
    def test_first_failure_no_cleanup(self, upload, delete):
        upload.side_effect = RuntimeError("first failed")
        with self.assertRaisesRegex(RuntimeError, "first failed"):
            self.run_pair()
        delete.assert_not_called()

if __name__ == "__main__":
    unittest.main()
