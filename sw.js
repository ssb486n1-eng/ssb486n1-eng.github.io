/* 오프라인 실행 + WASM 멀티스레드용 서비스 워커
   - 앱 파일: 네트워크 우선(업데이트 반영) → 실패 시 캐시
   - 같은 출처 응답에 COOP/COEP 헤더를 붙여 crossOriginIsolated 를 켠다
     → SharedArrayBuffer 사용 가능 → ONNX Runtime WASM 멀티스레드 (헤더를 못 바꾸는 정적 호스팅용)
   - 런타임: 배포본은 같은 출처 ./vendor/ (캐시 우선), 개발 환경은 jsDelivr @3.7.6 (캐시 우선)
   - 모델 가중치(Hugging Face)는 transformers.js 가 자체 Cache Storage 에 저장하므로 관여하지 않음
   - 통역 내용은 어떤 요청에도 포함되지 않음 (GET 정적 파일만 캐시) */
const APP_CACHE = 'li-app-dd88f72';
const RT_CACHE = 'li-runtime-v1';
const SHELL = ['./', './index.html', './core.js', './glossary.js', './engine.mjs', './worker.mjs', './manifest.webmanifest'];
const RT_PREFIX = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/';

function isolate(res) {
  if (!res || res.status === 0 || res.type === 'opaque') return res;
  const h = new Headers(res.headers);
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Embedder-Policy', 'require-corp');
  h.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

self.addEventListener('install', e => {
  e.waitUntil(caches.open(APP_CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys
    .filter(k => k.startsWith('li-') && k !== APP_CACHE && k !== RT_CACHE)
    .map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = req.url;
  if (url.startsWith(RT_PREFIX)) {
    e.respondWith(caches.open(RT_CACHE).then(async c => {
      const hit = await c.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) c.put(req, res.clone());
      return res;
    }));
    return;
  }
  // 같은 출처 런타임(./vendor/, 약 22MB)은 캐시 우선 — 배포마다 캐시 이름이 바뀌므로 새 버전은 자동 교체
  if (new URL(url).origin === self.location.origin && /\/vendor\//.test(new URL(url).pathname)) {
    e.respondWith(caches.open(APP_CACHE).then(async c => {
      const hit = await c.match(req);
      if (hit) return isolate(hit);
      const res = await fetch(req);
      if (res.ok) c.put(req, res.clone());
      return isolate(res);
    }));
    return;
  }
  if (new URL(url).origin === self.location.origin) {
    // GitHub Pages 는 max-age=600 을 보내므로 그냥 fetch 하면 브라우저 HTTP 캐시의 예전 파일이 섞일 수 있다
    // → 항상 재검증(no-cache). 바뀌지 않았으면 304 라 데이터 낭비는 거의 없음
    const fresh = req.mode === 'navigate' ? new Request(req.url, { cache: 'no-cache', credentials: 'same-origin' }) : new Request(req, { cache: 'no-cache' });
    e.respondWith(fetch(fresh).then(res => {
      if (res.ok) { const cl = res.clone(); caches.open(APP_CACHE).then(c => c.put(req, cl)); }
      return isolate(res);
    }).catch(async () => isolate((await caches.match(req)) || (await caches.match('./index.html')))));
  }
});
