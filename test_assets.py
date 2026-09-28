import asyncio
import unittest
from unittest.mock import patch

import httpx

import server


class MascotAssetRouteTests(unittest.TestCase):
    def setUp(self):
        self.token_patch = patch.object(server, "TOKEN", "unit-test-token")
        self.token_patch.start()
        self.addCleanup(self.token_patch.stop)

    def get(self, path):
        async def request():
            transport = httpx.ASGITransport(app=server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get(path)

        return asyncio.run(request())

    def test_mobile_ui_loads_config_from_the_bridge_not_a_hard_coded_wake_phrase(self):
        response = self.get("/unit-test-token/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"assets/lare-idle.svg", response.content)
        for state in ('listening', 'thinking', 'speaking', 'error'):
            self.assertIn(f'assets/lare-{state}.svg'.encode(), response.content)
        self.assertNotIn(b"Ehi Lari", response.content)
        self.assertNotIn(b"Hey Lari", response.content)

    def test_mascot_svg_is_served_only_with_the_installation_token(self):
        response = self.get("/unit-test-token/assets/lare-concept.svg")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("image/svg+xml"))
        self.assertIn(b"<svg", response.content)
        self.assertIn(b"The Lare", response.content)

    def test_mascot_svg_rejects_an_invalid_token(self):
        response = self.get("/wrong-token/assets/lare-concept.svg")
        self.assertEqual(response.status_code, 403)

    def test_debug_audio_route_requires_the_installation_token(self):
        response = self.get("/wrong-token/debug/last.wav")
        self.assertEqual(response.status_code, 403)

    def test_each_mascot_state_has_a_separate_animated_svg(self):
        import xml.etree.ElementTree as ET
        from pathlib import Path
        design_state = {'idle': 'idle', 'listening': 'listen', 'thinking': 'think',
                        'speaking': 'speak', 'error': 'error'}
        for state, design in design_state.items():
            path = Path(server.BASE_DIR) / 'static' / 'assets' / f'lare-{state}.svg'
            with self.subTest(state=state):
                self.assertTrue(path.is_file())
                root = ET.parse(path).getroot()
                self.assertEqual(root.attrib.get('viewBox', '').split(),
                                 ['0', '0', '380', '380'])
                self.assertEqual(root.attrib.get('data-state'), design)
                body = path.read_text()
                self.assertNotIn('class="demo"', body)
                self.assertIn(f'<g id="{design}-state"', body)
                for other in design_state.values():
                    if other != design:
                        self.assertNotIn(f'<g id="{other}-state"', body)
                response = self.get(f'/unit-test-token/assets/lare-{state}.svg')
                self.assertEqual(response.status_code, 200)
                self.assertIn('image/svg+xml', response.headers['content-type'])
    def test_asset_route_does_not_serve_arbitrary_files(self):
        response = self.get("/unit-test-token/assets/other.svg")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
