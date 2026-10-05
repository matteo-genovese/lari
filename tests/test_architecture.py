"""Cheap, credential-free guards for the runtime's architectural boundaries."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "lari"


def sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        yield path.relative_to(ROOT).as_posix(), source, ast.parse(source)


class ArchitectureTests(unittest.TestCase):
    def test_llm_requests_belong_to_hermes(self):
        for path, source, tree in sources():
            with self.subTest(module=path):
                for forbidden in ("ask_deepseek", "DEEPSEEK_", "AGENT_BACKEND"):
                    self.assertNotIn(forbidden, source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        # All HTTP call styles: inline URL or a separately assigned URL.
                        if node.func.attr not in {"post", "request", "send"}:
                            continue
                        rendered = ast.unparse(node)
                        if "chat/completions" in rendered:
                            self.assertTrue(path.startswith("lari/hermes/"), rendered)
                            self.assertIn("settings.hermes_api", rendered)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        if "chat/completions" in node.value:
                            self.assertTrue(path.startswith("lari/hermes/"), node.value)
                            self.assertNotIn("://", node.value,
                                             "LLM endpoint must derive from Hermes settings")

    def test_environment_access_belongs_to_config(self):
        for path, _, tree in sources():
            if path == "lari/config.py":
                continue
            os_names = {"os"}
            environment_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    os_names.update(alias.asname or "os" for alias in node.names if alias.name == "os")
                elif isinstance(node, ast.ImportFrom) and node.module == "os":
                    environment_names.update(alias.asname or alias.name for alias in node.names
                                             if alias.name in {"environ", "getenv", "putenv", "unsetenv"})
            for node in ast.walk(tree):
                with self.subTest(module=path, line=getattr(node, "lineno", 0)):
                    if isinstance(node, ast.Name):
                        self.assertNotIn(node.id, environment_names)
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                        if node.value.id in os_names:
                            self.assertNotIn(node.attr, {"environ", "getenv", "putenv", "unsetenv"})

    def test_server_does_not_import_voice_or_http_providers(self):
        tree = ast.parse((PACKAGE / "server.py").read_text())
        forbidden = {"faster_whisper", "vosk", "edge_tts", "httpx"}
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            for module in modules:
                self.assertNotIn(module.split(".")[0], forbidden)

    def test_protocol_literals_belong_to_protocol(self):
        for path, _, tree in sources():
            if path == "lari/protocol.py":
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    for key in node.keys:
                        with self.subTest(module=path, line=node.lineno):
                            self.assertFalse(isinstance(key, ast.Constant) and key.value == "type",
                                             "Construct satellite messages via lari.protocol")

    def test_no_duplicate_legacy_modules(self):
        for name in ("wake_config.py", "stt_backends.py"):
            self.assertFalse((PACKAGE / name).exists(), name)
