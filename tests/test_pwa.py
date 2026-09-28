"""PWA installability: manifest, service worker and icons behind the token."""
import asyncio
import unittest
from unittest.mock import patch

import httpx

from lari import server


class PwaTests(unittest.TestCase):
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

    def test_manifest_is_served_for_the_installation(self):
        response = self.get("/unit-test-token/manifest.webmanifest")
        self.assertEqual(response.status_code, 200)
        self.assertIn("manifest", response.headers["content-type"])
        data = response.json()
        self.assertEqual(data["start_url"], "/unit-test-token/")
        self.assertEqual(data["scope"], "/unit-test-token/")
        self.assertEqual(data["display"], "fullscreen")
        self.assertEqual(data["theme_color"], "#010000")
        icons = data["icons"]
        sources = [i["src"] for i in icons]
        self.assertTrue(any(s.endswith("/assets/icon-192.png") for s in sources))
        self.assertTrue(any(s.endswith("/assets/icon-512.png") for s in sources))
        for source in sources:
            self.assertTrue(source.startswith("/unit-test-token/assets/"))

    def test_service_worker_is_served_with_the_installation_token(self):
        response = self.get("/unit-test-token/sw.js")
        self.assertEqual(response.status_code, 200)
        self.assertIn("javascript", response.headers["content-type"])
        self.assertIn(b"addEventListener", response.content)

    def test_pwa_routes_reject_an_invalid_token(self):
        self.assertEqual(self.get("/wrong-token/manifest.webmanifest").status_code, 403)
        self.assertEqual(self.get("/wrong-token/sw.js").status_code, 403)

    def test_ui_registers_the_pwa(self):
        response = self.get("/unit-test-token/")
        self.assertIn(b"manifest.webmanifest", response.content)
        self.assertIn(b"serviceWorker", response.content)

    def test_pwa_icons_are_served(self):
        for name in ("icon-192.png", "icon-512.png"):
            with self.subTest(icon=name):
                response = self.get(f"/unit-test-token/assets/{name}")
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.headers["content-type"].startswith("image/png"))


if __name__ == "__main__":
    unittest.main()
