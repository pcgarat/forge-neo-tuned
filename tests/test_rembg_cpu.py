#!/usr/bin/env python3
"""Tests del parche de rembg (proveedor CPU + caché de sesiones).

No necesita Forge ni GPU: se inyecta un rembg falso en sys.modules.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "builtin-extensions" / "forge-neo-rembg-cpu" / "scripts" / "rembg_cpu.py"


def load_module():
    """Carga el script como módulo nuevo (evita el estado global entre tests)."""
    spec = importlib.util.spec_from_file_location("rembg_cpu_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSession:
    def __init__(self, model_name, providers):
        self.model_name = model_name
        self.providers = providers


class FakeRembg:
    """Doble mínimo de rembg: session_factory, bg y el paquete raíz."""

    def __init__(self):
        self.calls = []

        def fake_new_session(model_name="bria-rmbg", *args, sess_opts=None, **kwargs):
            self.calls.append((model_name, kwargs.get("providers")))
            return FakeSession(model_name, kwargs.get("providers"))

        self.calls_made = self.calls
        self.session_factory = type(sys)("rembg.session_factory")
        self.session_factory.new_session = fake_new_session
        self.bg = type(sys)("rembg.bg")
        self.bg.new_session = fake_new_session

    def new_session(self, *a, **kw):  # pragma: no cover - reasignado por apply()
        raise AssertionError("apply() debería haber reasignado este atributo")


class ProvidersTests(unittest.TestCase):
    def test_default_es_cpu(self):
        module = load_module()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REMBG_PROVIDER", None)
            self.assertEqual(module.providers(), ["CPUExecutionProvider"])

    def test_gpu_explicito(self):
        module = load_module()
        for valor in ("gpu", "cuda", "GPU"):
            with self.subTest(valor=valor):
                with mock.patch.dict(os.environ, {"REMBG_PROVIDER": valor}):
                    self.assertEqual(
                        module.providers(),
                        ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )

    def test_valor_desconocido_cae_en_cpu(self):
        module = load_module()
        with mock.patch.dict(os.environ, {"REMBG_PROVIDER": "tpu"}):
            self.assertEqual(module.providers(), ["CPUExecutionProvider"])


class U2netHomeTests(unittest.TestCase):
    def test_fija_u2net_home_si_falta(self):
        module = load_module()
        fake_paths = type(sys)("modules.paths_internal")
        fake_paths.models_path = "/data/Models"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("U2NET_HOME", None)
            with mock.patch.dict(sys.modules, {"modules.paths_internal": fake_paths}):
                module._install_u2net_home()
            self.assertEqual(os.environ["U2NET_HOME"], "/data/Models/u2net")

    def test_respeta_u2net_home_existente(self):
        module = load_module()
        fake_paths = type(sys)("modules.paths_internal")
        fake_paths.models_path = "/data/Models"
        with mock.patch.dict(os.environ, {"U2NET_HOME": "/otra/ruta"}):
            with mock.patch.dict(sys.modules, {"modules.paths_internal": fake_paths}):
                module._install_u2net_home()
            self.assertEqual(os.environ["U2NET_HOME"], "/otra/ruta")

    def test_sin_modulo_paths_no_revienta(self):
        module = load_module()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("U2NET_HOME", None)
            with mock.patch.dict(sys.modules, {"modules.paths_internal": None}):
                module._install_u2net_home()
            self.assertNotIn("U2NET_HOME", os.environ)


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeRembg()
        self.patches = mock.patch.dict(
            sys.modules,
            {
                "rembg": self.fake,
                "rembg.bg": self.fake.bg,
                "rembg.session_factory": self.fake.session_factory,
                "modules.paths_internal": None,
            },
        )
        self.patches.start()

    def tearDown(self):
        self.patches.stop()

    def test_apply_inyecta_cpu(self):
        module = load_module()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REMBG_PROVIDER", None)
            self.assertTrue(module.apply())
        self.fake.new_session("u2net")
        self.assertEqual(self.fake.calls[-1][1], ["CPUExecutionProvider"])

    def test_cachea_la_sesion_por_modelo(self):
        module = load_module()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REMBG_PROVIDER", None)
            module.apply()
        primera = self.fake.new_session("u2net")
        segunda = self.fake.new_session("u2net")
        otra = self.fake.new_session("isnet-general-use")
        self.assertIs(primera, segunda)
        self.assertIsNot(primera, otra)
        self.assertEqual(len(self.fake.calls), 2)

    def test_apply_es_idempotente(self):
        module = load_module()
        with mock.patch.dict(os.environ, {}, clear=False):
            module.apply()
            parcheado = self.fake.new_session
            module.apply()
        self.assertIs(self.fake.new_session, parcheado)

    def test_apply_sin_rembg_devuelve_false(self):
        module = load_module()
        with mock.patch.dict(sys.modules, {"rembg": None}, clear=False):
            self.assertFalse(module.apply())


if __name__ == "__main__":
    unittest.main()
