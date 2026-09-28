// Lari service worker: makes the phone app installable and keeps the shell
// usable offline.  The voice path is a WebSocket and always needs the network;
// this only guarantees the page opens and can report "disconnected".
const SHELL = ["./", "./assets/logo.jpg", "./assets/favicon.ico", "./assets/lare-idle.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open("lari-shell").then((cache) => cache.addAll(SHELL)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(
    fetch(event.request).catch(() =>
      caches.match(event.request, { ignoreSearch: true }).then((hit) => hit || Response.error())
    )
  );
});
