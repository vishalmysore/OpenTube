// Adds Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers to every same-origin response, which
// makes the page "cross-origin isolated" so ONNX Runtime Web can use multithreaded WebAssembly.
// Copied from layaAsRagJudge, changed to COEP "credentialless" (was "require-corp") so YouTube thumbnails, which
// are served without a Cross-Origin-Resource-Policy header, still load on an isolated page.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const r = e.request;
  if (r.cache === "only-if-cached" && r.mode !== "same-origin") return;
  if (new URL(r.url).origin !== self.location.origin) return; // only same-origin documents need the headers
  e.respondWith(
    fetch(r).then((res) => {
      if (res.status === 0 || res.type === "opaque") return res;
      const h = new Headers(res.headers);
      h.set("Cross-Origin-Embedder-Policy", "credentialless");
      h.set("Cross-Origin-Opener-Policy", "same-origin");
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
    }).catch((err) => { console.error(err); return Response.error(); })
  );
});
