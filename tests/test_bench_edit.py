#!/usr/bin/env python3
"""Tests del benchmark de Identity Edit (LoRA en el prompt, verificación en log, informe)."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_edit as b  # noqa: E402

_GB = 1024**3
_LORA = "krea2_identity_edit_v1_2_r64"


class LoraInjectionTests(unittest.TestCase):
    def test_tag_added(self):
        out = b.inject_lora("make it a kitchen", _LORA)
        self.assertTrue(out.startswith("<lora:krea2_identity_edit_v1_2_r64:1>"))
        self.assertTrue(out.endswith("make it a kitchen"))

    def test_no_lora_leaves_prompt_untouched(self):
        self.assertEqual(b.inject_lora("a prompt", None), "a prompt")

    def test_idempotent(self):
        once = b.inject_lora("a prompt", _LORA)
        self.assertEqual(b.inject_lora(once, _LORA), once)

    def test_weight_is_formatted(self):
        self.assertIn(":0.75>", b.inject_lora("x", _LORA, 0.75))


class ScriptArgsTests(unittest.TestCase):
    """El orden de args debe seguir el `return` del ui() de krea2_edit.py."""

    def test_positional_order_and_count(self):
        args = b.edit_script_args(768, 4.0, "fit source to output (v1.2)", "B64")
        self.assertEqual(len(args), 8)
        self.assertIs(args[0], True)
        self.assertEqual(args[1], "B64")
        self.assertIsNone(args[2])
        self.assertEqual(args[3], 768)
        self.assertEqual(args[4], "fit source to output (v1.2)")
        self.assertEqual(args[5], 4.0)
        self.assertEqual(args[6], 1.0)
        self.assertIs(args[7], False)

    def test_second_reference_goes_in_slot_three(self):
        args = b.edit_script_args(384, 2.0, "fit source to output (v1.2)", "A", second_b64="B")
        self.assertEqual(args[2], "B")

    def test_prep_disabled(self):
        # El face-ref prep lanza dos generaciones anidadas: nunca debe medirse con él.
        self.assertIs(b.edit_script_args(768, 4.0, "off", "x")[7], False)


class BodyTests(unittest.TestCase):
    def _body(self, **kw):
        params = dict(
            width=640,
            height=640,
            steps=8,
            seed=777,
            source_b64="B64",
            grounding_px=768,
            ref_boost=4.0,
            ar_mode="fit source to output (v1.2)",
            lora=_LORA,
        )
        params.update(kw)
        return b.build_edit_body(**params)

    def test_edit_script_present(self):
        body = self._body()
        self.assertIn("Krea2 Identity Edit", body["alwayson_scripts"])

    def test_lora_in_prompt(self):
        self.assertIn(f"<lora:{_LORA}:1>", self._body()["prompt"])

    def test_no_lora_when_disabled(self):
        self.assertNotIn("<lora:", self._body(lora=None)["prompt"])

    def test_fixed_comparison_fields(self):
        body = self._body()
        self.assertEqual(body["seed"], 777)
        self.assertEqual(body["steps"], 8)
        self.assertFalse(body["send_images"])
        self.assertTrue(body["do_not_save_samples"])

    def test_grounding_and_boost_reach_the_script(self):
        args = self._body(grounding_px=384, ref_boost=2.0)["alwayson_scripts"]["Krea2 Identity Edit"]["args"]
        self.assertEqual(args[3], 384)
        self.assertEqual(args[5], 2.0)


class TeDeviceTests(unittest.TestCase):
    def test_explicit_flag_wins(self):
        self.assertEqual(b.te_device_from_flags({"text_enc_device": "cuda", "lowvram": True}), "cuda")

    def test_lowvram_implies_cpu(self):
        self.assertIn("cpu", b.te_device_from_flags({"lowvram": True}))

    def test_auto_without_lowvram(self):
        self.assertEqual(b.te_device_from_flags({}), "auto")


class LoraLoadedTests(unittest.TestCase):
    def test_detects_loaded(self):
        log = f"[LORA] Loaded {_LORA}.safetensors for KModel-UNet with 256 keys"
        self.assertTrue(b.lora_loaded(log, _LORA))

    def test_detects_without_suffix_in_log(self):
        self.assertTrue(b.lora_loaded(f"[LORA] Loaded {_LORA} for KModel-UNet", _LORA))

    def test_accepts_name_with_suffix(self):
        self.assertTrue(b.lora_loaded(f"[LORA] Loaded {_LORA}.safetensors", f"{_LORA}.safetensors"))

    def test_missing_is_false(self):
        self.assertFalse(b.lora_loaded("[LORA] Loaded something_else.safetensors", _LORA))

    def test_ansi_codes_do_not_break_match(self):
        log = f"\x1b[0m[LORA] Loaded {_LORA}.safetensors for KModel"
        self.assertTrue(b.lora_loaded(log, _LORA))

    def test_no_lora_name_is_unknown(self):
        self.assertIsNone(b.lora_loaded("[LORA] Loaded x", None))


class GuardTests(unittest.TestCase):
    """Sin pruebas en el `info` la medición no vale: debe abortar, no reportar un número falso."""

    _OK_INFO = {
        "infotexts": [f'Krea2 Edit: "1 source(s) [abc]", Lora hashes: "{_LORA}: 1"']
    }

    def test_aborts_when_lora_missing(self):
        orig = b.run_edit_once
        b.run_edit_once = lambda *a, **k: (1.0, {"infotexts": ['Krea2 Edit: "1 source(s) [a]"']})
        try:
            with self.assertRaises(b.BenchError):
                b.benchmark(
                    "http://x",
                    label="t",
                    width=640,
                    height=640,
                    steps=8,
                    seed=1,
                    runs=1,
                    source_b64="B64",
                    grounding_px=768,
                    ref_boost=4.0,
                    ar_mode="off",
                    lora=_LORA,
                    container="forge-neo",
                    warmup=False,
                )
        finally:
            b.run_edit_once = orig

    def test_passes_with_evidence(self):
        orig_run, orig_mem = b.run_edit_once, b.fetch_memory
        b.run_edit_once = lambda *a, **k: (1.0, self._OK_INFO)
        b.fetch_memory = lambda *a, **k: {}
        try:
            result = b.benchmark(
                "http://x",
                label="t",
                width=640,
                height=640,
                steps=8,
                seed=1,
                runs=2,
                source_b64="B64",
                grounding_px=768,
                ref_boost=4.0,
                ar_mode="off",
                lora=_LORA,
                container="",
                warmup=True,
            )
        finally:
            b.run_edit_once, b.fetch_memory = orig_run, orig_mem
        self.assertEqual(result.runs, [1.0, 1.0])

    def test_checkpoint_is_not_faked_through_override(self):
        # `override_settings.sd_model_checkpoint` da HTTP 500 en Neo 2.29: no se intenta.
        body = b.build_edit_body(
            width=640, height=640, steps=8, seed=1, source_b64="B",
            grounding_px=768, ref_boost=4.0, ar_mode="off", lora=None,
        )
        self.assertNotIn("override_settings", body)

    def test_rejects_non_krea2_checkpoint(self):
        options = {"sd_model_checkpoint": "FLUX/klein_bigLove_klein4_Q5_K_M.gguf"}
        with self.assertRaises(b.BenchError) as ctx:
            b.require_krea2_checkpoint(options)
        self.assertIn("klein_bigLove", str(ctx.exception))
        self.assertIn("make krea2", str(ctx.exception))

    def test_accepts_krea2_checkpoint(self):
        options = {"sd_model_checkpoint": "Krea/krea2_krea2Int4Convrot_v10Turbo.safetensors"}
        self.assertEqual(
            b.require_krea2_checkpoint(options),
            "Krea/krea2_krea2Int4Convrot_v10Turbo.safetensors",
        )

    def test_rejects_missing_checkpoint(self):
        with self.assertRaises(b.BenchError):
            b.require_krea2_checkpoint({})

    def test_krea2_detection_is_case_insensitive(self):
        self.assertTrue(b.is_krea2_checkpoint("Krea/Krea2_Turbo.safetensors"))
        self.assertFalse(b.is_krea2_checkpoint("ponyRealism.safetensors"))


class InfotextTests(unittest.TestCase):
    """La verificación autoritativa: el `info` de la respuesta, no el log."""

    _INFO = {
        "infotexts": [
            "<lora:krea2_identity_edit_v1_2_r64:1> make them stand in a kitchen\n"
            "Steps: 8, Sampler: Euler, CFG scale: 1.0, Size: 640x640, "
            'Krea2 Edit: "1 source(s) [705606], grounding_px 768, AR fit, ref_boost 4", '
            'Lora hashes: "krea2_identity_edit_v1_2_r64: 6adf9a69cc95", Version: neo-2.29'
        ]
    }

    def test_parses_loras_and_sources(self):
        parsed = b.parse_infotext(self._INFO["infotexts"][0])
        self.assertEqual(parsed["loras"], [_LORA])
        self.assertEqual(parsed["sources"], 1)
        self.assertEqual(parsed["grounding_px"], 768)

    def test_parses_multiple_loras(self):
        parsed = b.parse_infotext('Lora hashes: "a: 1, b: 2"')
        self.assertEqual(parsed["loras"], ["a", "b"])

    def test_missing_fields_are_none(self):
        parsed = b.parse_infotext("Steps: 8")
        self.assertEqual(parsed["loras"], [])
        self.assertIsNone(parsed["sources"])
        self.assertIsNone(parsed["grounding_px"])

    def test_verify_passes_with_evidence(self):
        b.verify_edit_infotext(self._INFO, _LORA)  # no debe lanzar

    def test_verify_rejects_missing_lora(self):
        info = {"infotexts": ['Krea2 Edit: "1 source(s) [abc]", Lora hashes: "otro: 1"']}
        with self.assertRaises(b.BenchError) as ctx:
            b.verify_edit_infotext(info, _LORA)
        self.assertIn("Lora hashes", str(ctx.exception))

    def test_verify_rejects_missing_reference(self):
        info = {"infotexts": [f'Lora hashes: "{_LORA}: 1"']}
        with self.assertRaises(b.BenchError) as ctx:
            b.verify_edit_infotext(info, _LORA)
        self.assertIn("source(s)", str(ctx.exception))

    def test_verify_error_names_the_checkpoint(self):
        info = {"infotexts": ['Model: FLUX/klein_bigLove_klein4_Q5_K_M.gguf, Steps: 8']}
        with self.assertRaises(b.BenchError) as ctx:
            b.verify_edit_infotext(info, _LORA)
        self.assertIn("klein_bigLove", str(ctx.exception))
        self.assertIn("Mismatch", str(ctx.exception))

    def test_verify_allows_no_lora_run_when_expected(self):
        info = {"infotexts": ['Krea2 Edit: "1 source(s) [abc]"']}
        b.verify_edit_infotext(info, None)  # comparación sin LoRA es legítima

    def test_verify_accepts_name_with_suffix(self):
        info = {
            "infotexts": [
                f'Krea2 Edit: "1 source(s) [a]", Lora hashes: "{_LORA}.safetensors: 1"'
            ]
        }
        b.verify_edit_infotext(info, f"{_LORA}.safetensors")

    def test_verify_rejects_empty_info(self):
        with self.assertRaises(b.BenchError):
            b.verify_edit_infotext({}, _LORA)


class FormatTests(unittest.TestCase):
    def _result(self, label: str) -> b.EditResult:
        return b.EditResult(
            label=label,
            width=640,
            height=640,
            steps=8,
            seed=777,
            grounding_px=768,
            ref_boost=4.0,
            lora=_LORA,
            te_device="cuda",
            runs=[30.0, 40.0],
            vram_peak=5 * _GB,
            ooms=0,
        )

    def test_row_contains_key_columns(self):
        table = b.format_table([self._result("te-gpu")])
        self.assertIn("| te-gpu | identity-edit | 640x640 | 8 |", table)
        self.assertIn("| cuda |", table)
        self.assertIn(_LORA, table)
        self.assertIn("| 35.00 |", table)

    def test_missing_vram_is_na(self):
        r = self._result("x")
        r.vram_peak = None
        self.assertIn("n/d", b.format_table([r]))

    def test_row_lists_all_applied_loras(self):
        r = self._result("multi")
        r.loras_seen = [_LORA, "krea2_turbo_4step_rank_64_lora"]
        self.assertIn(f"{_LORA}, krea2_turbo_4step_rank_64_lora", b.format_table([r]))

    def test_row_falls_back_to_requested_lora(self):
        r = self._result("single")
        r.loras_seen = []
        self.assertIn(_LORA, b.format_table([r]))

    def test_no_lora_renders_dash(self):
        r = self._result("x")
        r.lora = None
        self.assertIn("| — |", b.format_table([r]))

    def test_report_single_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "r.md")
            b.append_report(path, [self._result("a")])
            b.append_report(path, [self._result("b")])
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("| Config |"), 1)
            self.assertIn("| a |", text)
            self.assertIn("| b |", text)
            self.assertTrue(text.startswith("# Última modificación: "))
            self.assertIn("sin el tag el modelo edita sin LoRA", text)


class ParseArTests(unittest.TestCase):
    def test_accepts_known_mode(self):
        self.assertEqual(b._parse_ar("fit source to output (v1.2)"), "fit source to output (v1.2)")

    def test_rejects_unknown(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            b._parse_ar("inventado")


if __name__ == "__main__":
    unittest.main()
