/* 모델 추론 워커 — UI 스레드와 분리
   런타임(transformers.js 3.7.6 + ONNX Runtime WASM)은 배포 시 ./vendor/ 에 같은 출처로 함께 올린다
   (GitHub Pages 워크플로가 npm 무결성 검증 후 복사). 개발 환경처럼 vendor 가 없으면 jsDelivr(버전 고정)에서 불러온다.
   같은 출처 런타임 → 외부 스크립트 의존 없음, COOP/COEP 환경에서 WASM 스레드 워커 생성이 확실함 */
import { createEngine } from './engine.mjs?v=dd88f72';

// 런타임을 비동기로 불러오는 동안 도착한 메시지를 잃지 않도록, 수신 함수를 가장 먼저 등록하고 대기열에 쌓는다
const queue = [];
let handle = null;
self.onmessage = (ev) => { if (handle) handle(ev.data); else queue.push(ev.data); };

const CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/';
const VENDOR = new URL('./vendor/', import.meta.url).href;
let T, base;
try { T = await import(VENDOR + 'transformers.min.js'); base = VENDOR; }
catch (e) { T = await import(CDN + 'transformers.min.js'); base = CDN; }

T.env.allowLocalModels = false;
T.env.useBrowserCache = true;
try { T.env.backends.onnx.wasm.wasmPaths = base; } catch (e) {}
const E = createEngine(T);
const progress = p => self.postMessage({ type: 'progress', p: { status: p.status, file: p.file, loaded: p.loaded, total: p.total } });

handle = async (m) => {
  try {
    const r = await E.handle(m, progress);
    self.postMessage(Object.assign({ type: 'ok', id: m.id }, r));
  } catch (e) {
    self.postMessage({ type: 'err', id: m.id, error: String((e && e.message) || e) });
  }
};
self.postMessage({ type: 'boot', runtime: base === VENDOR ? 'vendor' : 'cdn' });
queue.splice(0).forEach(handle);
