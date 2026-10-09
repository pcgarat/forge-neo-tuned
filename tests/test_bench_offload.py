#!/usr/bin/env python3
"""Tests del benchmark de offload (parseo de resoluciones, log y tabla)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_offload as b  # noqa: E402

_GB = 1024**3


class ParseSizesTests(unittest.TestCase):
    def test_single(self):
        self.assertEqual(b.parse_sizes("768x768"), [(768, 768)])

    def test_multiple_with_spaces(self):
        self.assertEqual(
            b.parse_sizes("768x768, 1024x1024 ,1280x1280"),
            [(768, 768), (1024, 1024), (1280, 1280)],
        )

    def test_uppercase_x(self):
        self.assertEqual(b.parse_sizes("832X1216"), [(832, 1216)])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            b.parse_sizes("")

    def test_bad_format_raises(self):
        with self.assertRaises(ValueError):
            b.parse_sizes("768")

    def test_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            b.parse_sizes("ancho x alto")


class ParseOffloadLogTests(unittest.TestCase):
    def test_loaded_partially_with_usable(self):
        log = (
            "loaded partially; 5120.15 MB usable, 2048.00 MB loaded, "
            "3072.00 MB offloaded, 256.00 MB buffer reserved, lowvram patches: 12"
        )
        stats = b.parse_offload_log(log)
        self.assertEqual(stats.loaded_partially, 1)
        self.assertAlmostEqual(stats.max_loaded_mb, 2048.0)
        self.assertAlmostEqual(stats.max_offloaded_mb, 3072.0)
        self.assertAlmostEqual(stats.max_buffer_mb, 256.0)
        self.assertEqual(stats.max_patches, 12)
        self.assertTrue(stats.seen)

    def test_loaded_partially_without_usable(self):
        log = (
            "loaded partially; 1024.00 MB loaded, 2048.00 MB offloaded, "
            "128.00 MB buffer reserved, lowvram patches: 3"
        )
        stats = b.parse_offload_log(log)
        self.assertEqual(stats.loaded_partially, 1)
        self.assertAlmostEqual(stats.max_offloaded_mb, 2048.0)

    def test_max_across_lines(self):
        log = (
            "loaded partially; 1000.00 MB loaded, 100.00 MB offloaded, 10.00 MB buffer reserved, lowvram patches: 1\n"
            "loaded partially; 2000.00 MB loaded, 900.00 MB offloaded, 90.00 MB buffer reserved, lowvram patches: 7"
        )
        stats = b.parse_offload_log(log)
        self.assertEqual(stats.loaded_partially, 2)
        self.assertAlmostEqual(stats.max_offloaded_mb, 900.0)
        self.assertEqual(stats.max_patches, 7)

    def test_unloaded_partially_accumulates(self):
        log = (
            "Unloaded partially: 100.00 MB freed, 20.00 MB remains loaded, 5.00 MB buffer reserved, lowvram patches: 2\n"
            "Unloaded partially: 250.00 MB freed, 0.00 MB remains loaded, 0.00 MB buffer reserved, lowvram patches: 8"
        )
        stats = b.parse_offload_log(log)
        self.assertEqual(stats.unloaded_events, 2)
        self.assertAlmostEqual(stats.total_freed_mb, 350.0)

    def test_pinning_and_streams(self):
        log = (
            "Pinned Memory: 30220 MB\n"
            "Using async weight offloading with 2 streams\n"
            "Total VRAM 8188 MB, total RAM 32000 MB\n"
        )
        stats = b.parse_offload_log(log)
        self.assertEqual(stats.pinned_memory_mb, 30220)
        self.assertEqual(stats.streams, 2)
        self.assertEqual(stats.total_vram_mb, 8188)
        self.assertEqual(stats.total_ram_mb, 32000)

    def test_strips_ansi(self):
        log = "\x1b[32mloaded partially; 10.00 MB loaded, 20.00 MB offloaded, 1.00 MB buffer reserved, lowvram patches: 1\x1b[0m"
        stats = b.parse_offload_log(log)
        self.assertEqual(stats.loaded_partially, 1)

    def test_empty_log(self):
        stats = b.parse_offload_log("")
        self.assertFalse(stats.seen)
        self.assertIsNone(stats.pinned_memory_mb)
        self.assertIsNone(stats.streams)


class FormatTableTests(unittest.TestCase):
    def _result(self) -> b.SizeResult:
        return b.SizeResult(
            label="offload-stream",
            width=1024,
            height=1024,
            steps=8,
            seed=1,
            runs=[10.0, 20.0, 30.0],
            vram_peak=6 * _GB,
            ooms=0,
        )

    def test_row_and_metrics(self):
        r = self._result()
        self.assertEqual(r.median, 20.0)
        self.assertAlmostEqual(r.it_s, 8 / 20.0)
        table = b.format_table([r])
        self.assertIn("| offload-stream | 1024x1024 | 8 | 1 |", table)
        self.assertIn("| 20.00 |", table)
        self.assertIn("6.00", table)

    def test_missing_vram_is_na(self):
        r = b.SizeResult(label="x", width=768, height=768, steps=8, seed=1, runs=[1.0])
        table = b.format_table([r])
        self.assertIn("n/d", table)


class FormatContextTests(unittest.TestCase):
    def test_includes_pinning_and_streams(self):
        stats = b.OffloadStats(pinned_memory_mb=8192, streams=2)
        text = b.format_context({"gpu": "RTX 4060, 8188 MiB", "swappiness": "10", "memlock_kb": "unlimited"}, stats)
        self.assertIn("8192 MB", text)
        self.assertIn("2 streams", text)
        self.assertIn("RTX 4060", text)

    def test_reports_disabled_when_absent(self):
        stats = b.OffloadStats()
        text = b.format_context({}, stats)
        self.assertIn("no aplicado", text)
        self.assertIn("desactivado", text)


class SummaryTests(unittest.TestCase):
    def test_empty_summary(self):
        self.assertIn("Sin líneas de offload", b.format_offload_summary(b.OffloadStats()))

    def test_summary_with_stats(self):
        stats = b.OffloadStats(
            loaded_partially=1,
            unloaded_events=1,
            max_offloaded_mb=3072.0,
            total_freed_mb=100.0,
            total_vram_mb=8188,
            total_ram_mb=32000,
        )
        text = b.format_offload_summary(stats)
        self.assertIn("3072 MB", text)
        self.assertIn("8188", text)


class AppendReportTests(unittest.TestCase):
    def _result(self) -> b.SizeResult:
        return b.SizeResult(label="flash", width=768, height=768, steps=8, seed=1, runs=[5.0])

    def test_creates_report_with_date_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "offload.md")
            b.append_report(path, label="flash", context={}, results=[self._result()], stats=b.OffloadStats())
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("# Última modificación: "))
            self.assertIn("# Resultados", text)
            self.assertIn("## Perfil `flash`", text)
            self.assertIn("| flash | 768x768 |", text)

    def test_appends_blocks_on_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "offload.md")
            b.append_report(path, label="flash", context={}, results=[self._result()], stats=b.OffloadStats())
            b.append_report(path, label="stream", context={}, results=[self._result()], stats=b.OffloadStats())
            text = path.read_text(encoding="utf-8")
            self.assertIn("## Perfil `flash`", text)
            self.assertIn("## Perfil `stream`", text)
            self.assertEqual(text.count("# Resultados"), 1)


if __name__ == "__main__":
    unittest.main()
