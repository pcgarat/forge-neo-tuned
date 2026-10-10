#!/usr/bin/env python3
"""Tests del asistente de arranque (parseo y reglas de recomendación)."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import choose_profile as c  # noqa: E402


class ParseResolutionTests(unittest.TestCase):
    def test_square(self):
        self.assertEqual(c.parse_resolution("1024"), 1024)

    def test_landscape_returns_long_side(self):
        self.assertEqual(c.parse_resolution("1280x720"), 1280)

    def test_portrait_returns_long_side(self):
        self.assertEqual(c.parse_resolution("832x1216"), 1216)

    def test_uppercase_x(self):
        self.assertEqual(c.parse_resolution("1024X1024"), 1024)

    def test_bad_format_raises(self):
        with self.assertRaises(ValueError):
            c.parse_resolution("grande")


class RecommendTests(unittest.TestCase):
    def test_video_forces_ck(self):
        rec = c.recommend(c.Answers(model="wan", kind="video", resolution=832))
        self.assertEqual(rec.attn, "ck")

    def test_wan_is_video_even_if_kind_image(self):
        rec = c.recommend(c.Answers(model="wan", kind="imagen", resolution=768))
        self.assertEqual(rec.attn, "ck")

    def test_low_res_image_uses_flash(self):
        rec = c.recommend(c.Answers(model="klein", resolution=1024))
        self.assertEqual(rec.attn, "flash")

    def test_krea2_forces_ck_at_any_resolution(self):
        for res in (768, 1024, 1280):
            rec = c.recommend(c.Answers(model="krea2", resolution=res))
            self.assertEqual(rec.attn, "ck", f"krea2 a {res} px debería usar ck")

    def test_high_res_image_uses_ck(self):
        rec = c.recommend(c.Answers(model="klein", resolution=1280))
        self.assertEqual(rec.attn, "ck")

    def test_long_sequence_uses_ck(self):
        rec = c.recommend(c.Answers(model="krea2", resolution=1024, long_sequence=True))
        self.assertEqual(rec.attn, "ck")

    def test_api_enables_warmup(self):
        rec = c.recommend(c.Answers(model="klein", api=True))
        self.assertTrue(rec.warmup)

    def test_krea2_always_warns_fast_fp8(self):
        rec = c.recommend(c.Answers(model="krea2", resolution=1024))
        self.assertTrue(any("--fast-fp8" in n for n in rec.notes))

    def test_video_notes_sparse_incompatibility(self):
        rec = c.recommend(c.Answers(model="wan", kind="video"))
        self.assertTrue(any("Sparse Attention" in n for n in rec.notes))

    def test_vram_defaults_to_auto(self):
        rec = c.recommend(c.Answers())
        self.assertEqual(rec.vram, "auto")

    def test_stream_on_by_default(self):
        rec = c.recommend(c.Answers())
        self.assertTrue(rec.stream)

    def test_stream_off_propagates_and_notes(self):
        rec = c.recommend(c.Answers(stream=False))
        self.assertFalse(rec.stream)
        self.assertTrue(any("STREAM=off" in n for n in rec.notes))


class MakeLinesTests(unittest.TestCase):
    def test_flash_plain(self):
        rec = c.Recommendation(attn="flash", attn_reason="")
        self.assertEqual(rec.make_lines(), ["ATTN=flash", "VRAM=auto"])

    def test_ck_with_warmup(self):
        rec = c.Recommendation(attn="ck", attn_reason="", warmup=True)
        self.assertEqual(rec.make_lines(), ["ATTN=ck", "VRAM=auto", "WARMUP=1"])

    def test_stream_off_appended(self):
        rec = c.Recommendation(attn="flash", attn_reason="", stream=False)
        self.assertEqual(rec.make_lines(), ["ATTN=flash", "VRAM=auto", "STREAM=off"])

    def test_stream_on_not_emitted(self):
        rec = c.Recommendation(attn="flash", attn_reason="", stream=True)
        self.assertNotIn("STREAM=off", rec.make_lines())


class AskStreamTests(unittest.TestCase):
    """El prompt debe ir a stderr: stdout lo captura el `eval` del Makefile."""

    def test_prompt_goes_to_stderr_not_stdout(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("builtins.input", return_value="1280"), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            value = c._ask("Lado mayor", 1024, c.parse_resolution)
        self.assertEqual(value, 1280)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("Lado mayor", err.getvalue())


if __name__ == "__main__":
    unittest.main()
