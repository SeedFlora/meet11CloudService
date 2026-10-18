"""Configuration and unavailable-database regressions without a running database."""

from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import HTTPException
from psycopg import OperationalError
import app
import rag


class LocalConfigurationTests(unittest.TestCase):
    def test_password_comes_from_env_without_secret_file(self):
        with patch.dict(os.environ, {"DB_PASSWORD": "test-password"}):
            with patch("rag.psycopg.connect") as connect:
                rag.connect()
        self.assertEqual(connect.call_args.kwargs["password"], "test-password")

    def test_missing_password_has_setup_instruction(self):
        with patch.dict(os.environ, {"DB_PASSWORD": ""}):
            with self.assertRaisesRegex(RuntimeError, "Salin .env.example"):
                rag.connect()

    def test_readiness_reports_unconfigured_database_as_503(self):
        with patch.dict(os.environ, {"DB_PASSWORD": ""}):
            with self.assertRaises(HTTPException) as raised:
                app.ready()
        self.assertEqual(raised.exception.status_code, 503)

    def test_query_database_failure_is_503_without_raw_details(self):
        with patch("app.answer_question", side_effect=OperationalError("private connection details")):
            with self.assertRaises(HTTPException) as raised:
                app.ask(app.AskInput(question="Apa itu container?"))
        self.assertEqual(raised.exception.status_code, 503)
        self.assertNotIn("private connection details", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
