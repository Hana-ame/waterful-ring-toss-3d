/* Service Worker for Waterful Ring-Toss 3D (PWA installable) */
const CACHE_NAME = 'waterful-ring-toss-3d-v2';
const ASSETS = [
  './',
  './index.html',
  './manifest.json',
  './icon-192.png',
  './icon-512.png',
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)).catch(() => {})
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  // Only handle same-origin GET requests.
  if (event.request.method !== 'GET' || url.origin !== self.location.origin) {
    return;
  }
  // NETWORK-FIRST for navigation & HTML: always try fresh network first so a newly
  // deployed build (new hashed bundle in index.html) is picked up immediately.
  // Only fall back to cache on network failure (offline). This prevents the classic
  // PWA "stale cache → 404 bundle → blank page" bug after every deploy.
  const isNavigation = event.request.mode === 'navigate';
  const isHtml = url.pathname === '/' || url.pathname.endsWith('/index.html');
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Cache successful same-origin responses (assets + HTML).
        if (response && response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone)).catch(() => {});
        }
        return response;
      })
      .catch(() => {
        // Offline: fall back to cache. For navigation, serve cached index.html (SPA-like).
        return caches.match(event.request).then((cached) => {
          if (cached) return cached;
          if (isNavigation || isHtml) return caches.match('./index.html');
          return Response.error();
        });
      })
  );
});
