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
        self.assertIn(b"assets/lare-concept.svg", response.content)
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

    def test_asset_route_does_not_serve_arbitrary_files(self):
        response = self.get("/unit-test-token/assets/other.svg")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
