#!/usr/bin/env python3
"""Tests del benchmark de atención (deducción de backend, payload, tabla, informe)."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_attn as b  # noqa: E402

_GB = 1024**3


class BackendLabelTests(unittest.TestCase):
    def test_override_wins(self):
        self.assertEqual(b.backend_label({"use_ck_attention": True}, "flash"), "flash")

    def test_ck(self):
        self.assertEqual(b.backend_label({"use_ck_attention": True}, None), "ck-int8")

    def test_pytorch_sdpa(self):
        self.assertEqual(
            b.backend_label({"use_pytorch_cross_attention": True}, None), "pytorch-sdpa"
        )

    def test_sage_explicit_function(self):
        self.assertEqual(b.backend_label({"sage_function": "fp16_cuda"}, None), "sage-fp16_cuda")

    def test_sage_auto_is_not_distinguishable(self):
        self.assertEqual(b.backend_label({"sage_function": "auto"}, None), "default")

    def test_default(self):
        self.assertEqual(b.backend_label({}, None), "default")


class BuildBodyTests(unittest.TestCase):
    def test_fixed_fields_and_seed(self):
        body = b.build_body(1024, 1024, 8, 42)
        self.assertEqual(body["seed"], 42)
        self.assertEqual(body["steps"], 8)
        self.assertEqual(body["width"], 1024)
        self.assertFalse(body["save_images"])
        self.assertFalse(body["send_images"])

    def test_checkpoint_override_restored(self):
        body = b.build_body(768, 768, 8, 1, checkpoint="krea2")
        self.assertEqual(body["override_settings"]["sd_model_checkpoint"], "krea2")
        self.assertTrue(body["override_settings_restore_afterwards"])

    def test_no_override_when_no_checkpoint(self):
        self.assertNotIn("override_settings", b.build_body(768, 768, 8, 1))


class MemoryParsingTests(unittest.TestCase):
    def test_vram_peak(self):
        mem = {"cuda": {"reserved": {"peak": 5 * _GB}}}
        self.assertEqual(b.vram_peak_from_memory(mem), 5 * _GB)

    def test_vram_peak_unavailable(self):
        self.assertIsNone(b.vram_peak_from_memory({"cuda": {"error": "unavailable"}}))
        self.assertIsNone(b.vram_peak_from_memory({}))

    def test_ooms(self):
        mem = {"cuda": {"events": {"oom": 3, "retries": 1}}}
        self.assertEqual(b.ooms_from_memory(mem), 3)

    def test_ooms_missing(self):
        self.assertIsNone(b.ooms_from_memory({"cuda": {}}))


class FormatTableTests(unittest.TestCase):
    def test_row_and_derived_metrics(self):
        r = b.BenchResult(
            label="ck-int8",
            endpoint="txt2img",
            width=1280,
            height=1280,
            steps=8,
            seed=1,
            runs=[10.0, 20.0, 30.0],
            vram_peak=6 * _GB,
            ooms=0,
        )
        self.assertEqual(r.median, 20.0)
        self.assertAlmostEqual(r.it_s, 8 / 20.0)
        table = b.format_table([r])
        self.assertIn("| ck-int8 | txt2img | 1280x1280 | 8 | 1 |", table)
        self.assertIn("| 20.00 |", table)
        self.assertIn("6.00", table)

    def test_missing_vram_is_na(self):
        r = b.BenchResult(label="x", endpoint="txt2img", width=768, height=768, steps=8, seed=1, runs=[1.0])
        table = b.format_table([r])
        self.assertIn("n/d", table)


class ResolveSizeTests(unittest.TestCase):
    def test_args_win(self):
        ns = argparse.Namespace(width=1024, height=1024, steps=20)
        width, height, steps, checkpoint = b.resolve_size(ns, data_path="/nonexistent")
        self.assertEqual((width, height, steps), (1024, 1024, 20))
        self.assertIsNone(checkpoint)

    def test_falls_back_to_defaults_without_params(self):
        ns = argparse.Namespace(width=0, height=0, steps=0)
        width, height, steps, checkpoint = b.resolve_size(ns, data_path="/nonexistent")
        self.assertEqual((width, height, steps), (1024, 1024, 8))
        self.assertIsNone(checkpoint)

    def test_reads_last_gen_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "params.txt").write_text(
                "a cat\nSteps: 8, Size: 832x1216, Model: krea2\n", encoding="utf-8"
            )
            ns = argparse.Namespace(width=0, height=0, steps=0)
            width, height, steps, checkpoint = b.resolve_size(ns, data_path=tmp)
            self.assertEqual((width, height), (832, 1216))
            self.assertEqual(checkpoint, "krea2")


class AppendReportTests(unittest.TestCase):
    def _result(self, label: str) -> b.BenchResult:
        return b.BenchResult(
            label=label, endpoint="txt2img", width=768, height=768, steps=8, seed=1, runs=[5.0]
        )

    def test_creates_report_with_date_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "report.md")
            b.append_report(path, [self._result("flash")])
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("# Última modificación: "))
            self.assertIn("# Resultados", text)
            self.assertIn("| flash |", text)

    def test_appends_rows_on_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "report.md")
            b.append_report(path, [self._result("flash")])
            b.append_report(path, [self._result("ck-int8")])
            text = path.read_text(encoding="utf-8")
            self.assertIn("| flash |", text)
            self.assertIn("| ck-int8 |", text)
            self.assertEqual(text.count("| Backend |"), 2)


if __name__ == "__main__":
    unittest.main()
