"""Cheap, credential-free guards for the runtime's architectural boundaries."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "lari"


# Config owns environment loading; server is the sole runtime composition root.
SETTINGS_OWNERS = {"lari/config.py", "lari/server.py"}


def settings_snapshots(tree):
    """Find module-scope assignments, including branches and annotated values."""
    getters = {"get_settings"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            getters.update(alias.asname or alias.name for alias in node.names
                           if alias.name == "get_settings")

    class ModuleAssignments(ast.NodeVisitor):
        def __init__(self):
            self.violations = []

        def visit_FunctionDef(self, node):
            pass

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

        def visit_Assign(self, node):
            self.check(node)

        visit_AnnAssign = visit_Assign
        visit_AugAssign = visit_Assign
        visit_NamedExpr = visit_Assign

        def check(self, node):
            if node.value is not None:
                for child in ast.walk(node.value):
                    if isinstance(child, ast.Call) and (
                        isinstance(child.func, ast.Name) and child.func.id in getters
                        or isinstance(child.func, ast.Attribute) and child.func.attr == "get_settings"
                    ):
                        self.violations.append(node.lineno)
                        break

    visitor = ModuleAssignments()
    visitor.visit(tree)
    return visitor.violations


def sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        yield path.relative_to(ROOT).as_posix(), source, ast.parse(source)


class ArchitectureTests(unittest.TestCase):
    def test_no_module_settings_snapshots(self):
        for path, _, tree in sources():
            if path in SETTINGS_OWNERS:
                continue
            for line in settings_snapshots(tree):
                self.fail(f"{path}:{line}: module settings snapshot; inject Settings "
                          "via a constructor/function parameter from lari.server or Session")

    def test_snapshot_guard_understands_ast_forms(self):
        for source in (
            "_S = get_settings()", "_WAKE = get_settings().wake_config",
            "x: Settings = get_settings()", "if True:\n    x = get_settings()",
            "from lari.config import get_settings as read\nx = read()",
            "x = config.get_settings()",
        ):
            with self.subTest(source=source):
                self.assertTrue(settings_snapshots(ast.parse(source)))
        self.assertFalse(settings_snapshots(ast.parse(
            "def make():\n    x = get_settings()\n    return x")))
        self.assertFalse(settings_snapshots(ast.parse("x = 'get_settings()'")))

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
