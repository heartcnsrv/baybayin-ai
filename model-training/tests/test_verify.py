import io
import json
import os
import unittest
import urllib.error
from unittest import mock

import numpy as np

from baybayin import verify

DRAFT = {"baybayin": "ᜐ", "tagalog": "sa", "english": "to",
         "words": [{"line": 0, "baybayin": "ᜐ", "reading": "sa", "tagalog": "sa", "confidence": 0.9, "box": [0, 0, 5, 5]}]}
IMG = np.full((40, 60, 3), 255, np.uint8)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("OPENAI_API_KEY", None)
        self.no_dotenv = mock.patch.object(verify, "load_env", lambda: None)
        self.no_dotenv.start()

    def tearDown(self):
        self.no_dotenv.stop()
        self.env.stop()

    def test_without_key_check_is_off(self):
        out = verify.verify(IMG, DRAFT)
        self.assertEqual(out["status"], "unavailable")
        self.assertNotIn("verified", out)

    def test_placeholder_key_counts_as_missing(self):
        os.environ["OPENAI_API_KEY"] = "replace-with-a-new-secret-key"
        self.assertEqual(verify.verify(IMG, DRAFT)["status"], "unavailable")

    def test_successful_review_is_returned(self):
        os.environ["OPENAI_API_KEY"] = "sk-test"
        reply = {"baybayin": "ᜐ ᜅ", "tagalog": "sa nga", "english": "to the",
                 "all_symbols_detected": False, "changes": ["added nga"], "confidence": "high", "note": ""}
        payload = json.dumps({"choices": [{"message": {"content": json.dumps(reply)}}]}).encode()
        with mock.patch("urllib.request.urlopen", return_value=_Resp(payload)) as call:
            out = verify.verify(IMG, DRAFT)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["verified"]["tagalog"], "sa nga")
        self.assertEqual(out["verified"]["changes"], ["added nga"])
        sent = json.loads(call.call_args[0][0].data)
        self.assertEqual(sent["messages"][1]["content"][1]["type"], "image_url")
        self.assertEqual(call.call_args[0][0].get_header("Authorization"), "Bearer sk-test")

    def test_api_error_falls_back(self):
        os.environ["OPENAI_API_KEY"] = "sk-test"
        err = urllib.error.HTTPError(verify.API_URL, 401, "Unauthorized", {}, io.BytesIO(b'{"error":{"message":"bad key"}}'))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            out = verify.verify(IMG, DRAFT)
        self.assertEqual(out["status"], "error")
        self.assertIn("bad key", out["message"])


if __name__ == "__main__":
    unittest.main()
