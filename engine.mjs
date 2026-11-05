/* =====================================================================
   모델 추론 엔진 — 브라우저 Web Worker(worker.mjs)와 Node 테스트가 같은 코드를 쓴다.
   T: transformers.js 모듈 (@huggingface/transformers 3.7.6)

   슬롯(slot) 단위로 여러 모델을 동시에 올려 둔다.
     asr      : Whisper 초안(빠른 모델)        asr2 : Whisper 확정(정확한 모델, 2단계 인식)
     mt:<s>-<t>: 방향별 번역 모델 (Opus-MT 등)   mt   : 다국어 번역 모델 (NLLB / M2M100)
   ===================================================================== */
export function createEngine(T, opts = {}) {
  const { pipeline, Tensor, env } = T;
  const slots = new Map();      // slot -> { pipe, key, device, task }
  const maxPrompt = opts.maxPromptTokens || 96;

  const toNum = x => (typeof x === 'bigint' ? Number(x) : x);
  function f16(h) {
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    return e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  function get(slot) {
    const s = slots.get(slot);
    if (!s) throw new Error('모델이 로드되지 않음: ' + slot);
    return s.pipe;
  }
  function specialId(tok, text) {
    const ids = tok.encode(text, { add_special_tokens: false });
    return ids.length === 1 ? ids[0] : null;
  }

  // ---- 설정: WASM 스레드 수 (crossOriginIsolated 일 때만 의미 있음)
  function configure(m) {
    try {
      if (m.numThreads && env && env.backends && env.backends.onnx && env.backends.onnx.wasm) env.backends.onnx.wasm.numThreads = m.numThreads;
    } catch (e) { /* 백엔드 초기화 이후에는 바꿀 수 없음 */ }
    return { numThreads: m.numThreads || 0 };
  }

  // ---- 로드: tries 를 순서대로 시도 (예: WebGPU → WASM)
  async function load(m, onProgress) {
    const key = m.task + '|' + m.model + '|' + JSON.stringify(m.tries);
    const cur = slots.get(m.slot);
    if (cur && cur.key === key) return { device: cur.device, cached: true };
    if (cur) { try { cur.pipe.dispose && (await cur.pipe.dispose()); } catch (e) {} slots.delete(m.slot); }
    let last = null;
    for (const o of m.tries) {
      try {
        const pipe = await pipeline(m.task, m.model, Object.assign({}, o, { progress_callback: onProgress }));
        slots.set(m.slot, { pipe, key, device: o.device, task: m.task, warmArgs: m.warmArgs || null });
        let warmMs = 0;
        if (m.warmup) warmMs = await warmup(m.slot);
        return { device: o.device, warmMs };
      } catch (e) { last = e; }
    }
    throw last || new Error('모델 로드 실패: ' + m.model);
  }

  // ---- 예열: WebGPU 는 첫 추론 때 셰이더를 컴파일하므로, 로드 직후 한 번 돌려 첫 발화 지연을 없앤다
  async function warmup(slot) {
    const t0 = now();
    const s = slots.get(slot);
    if (s.task === 'automatic-speech-recognition') {
      await asr({ slot, audio: new Float32Array(16000), language: 'en', maxTokens: 2 });
    } else {
      const p = s.pipe;
      await p('ok', Object.assign({ max_new_tokens: 2 }, s.warmArgs || {})).catch(() => {});
    }
    return now() - t0;
  }
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

  /* ---- 음성 인식
     decoder_input_ids 를 직접 구성해 Whisper 의 '이전 문맥 프롬프트'를 넣는다:
       <|startofprev|> 프롬프트 … <|startoftranscript|> <|ko|> <|transcribe|> <|notimestamps|>
     프롬프트(용어집·직전 발화)는 드문 용어·약어 인식률을 높인다 (Whisper 논문 §4.5 장문 전사에서
     이전 텍스트 조건화, OpenAI Speech-to-text 가이드 'Prompting').
     transformers.js 3.7.6 파이프라인은 prompt_ids 를 지원하지 않아 generate 를 직접 호출한다. */
  // 프롬프트 복사 감지: 모델이 확신이 없을 때 프롬프트 문구를 그대로 출력하는 실패 모드
  const normT = x => String(x || '').toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, '');
  function isPromptCopy(text, prompt) {
    const t = normT(text), p = normT(prompt);
    if (!t || !p) return false;
    return t.length >= 2 && p.includes(t);
  }

  async function asr(m) {
    const r = await asrOnce(m);
    if (m.prompt && isPromptCopy(r.text, m.prompt)) {
      if (m.retryNoPrompt === false) return Object.assign(r, { text: '', promptCopy: true });
      const r2 = await asrOnce(Object.assign({}, m, { prompt: '' }));
      return Object.assign(r2, { ms: r.ms + r2.ms, promptCopy: true });
    }
    return r;
  }

  async function asrOnce(m) {
    const p = get(m.slot);
    const tok = p.tokenizer, gc = p.model.generation_config;
    const t0 = now();
    const { input_features } = await p.processor(m.audio);
    const init = [gc.decoder_start_token_id];
    if (gc.is_multilingual) {
      const lid = gc.lang_to_id && gc.lang_to_id['<|' + m.language + '|>'];
      if (lid == null) throw new Error('Whisper 가 지원하지 않는 언어: ' + m.language);
      init.push(lid, gc.task_to_id.transcribe);
    }
    if (gc.no_timestamps_token_id != null) init.push(gc.no_timestamps_token_id);
    let pre = [];
    if (m.prompt && String(m.prompt).trim()) {
      const sp = specialId(tok, '<|startofprev|>');
      if (sp != null) {
        const ids = tok.encode(' ' + String(m.prompt).trim(), { add_special_tokens: false });
        pre = [sp].concat(ids.slice(-maxPrompt));
      }
    }
    const ids = pre.concat(init);
    const out = await p.model.generate({ inputs: input_features, decoder_input_ids: ids, max_new_tokens: Math.max(2, m.maxTokens || 128) });
    const seq = out.tolist()[0].map(toNum).slice(ids.length);
    const text = tok.decode(seq, { skip_special_tokens: true }).trim();
    return { text, ms: now() - t0, promptTokens: pre.length };
  }

  // ---- 언어 식별: <|startoftranscript|> 다음 토큰의 logit 을 후보 언어끼리만 비교
  async function lid(m) {
    const p = get(m.slot);
    const t0 = now();
    const { input_features } = await p.processor(m.audio);
    const gc = p.model.generation_config;
    const ids = new Tensor('int64', BigInt64Array.from([BigInt(gc.decoder_start_token_id)]), [1, 1]);
    const out = await p.model({ input_features, decoder_input_ids: ids });
    const lg = out.logits, V = lg.dims[lg.dims.length - 1], off = lg.data.length - V;
    const val = i => (lg.data instanceof Uint16Array ? f16(lg.data[off + i]) : Number(lg.data[off + i]));
    const scores = {};
    for (const c of m.cands) {
      const id = gc.lang_to_id && gc.lang_to_id['<|' + c + '|>'];
      scores[c] = id == null ? -1e9 : val(id);
    }
    return { scores, ms: now() - t0 };
  }

  // ---- 번역: 다국어 모델이면 src/tgt 코드 전달, 방향 전용 모델(Opus-MT)이면 생략
  async function mt(m) {
    const p = get(m.slot);
    const t0 = now();
    const args = { max_new_tokens: m.maxTokens || 200 };
    if (m.src) args.src_lang = m.src;
    if (m.tgt) args.tgt_lang = m.tgt;
    const out = await p(m.text, args);
    return { text: (out && out[0] && out[0].translation_text) || '', ms: now() - t0 };
  }

  async function handle(m, onProgress) {
    switch (m.type) {
      case 'config': return configure(m);
      case 'load': return load(m, onProgress);
      case 'asr': return asr(m);
      case 'lid': return lid(m);
      case 'mt': return mt(m);
      case 'unload': { const s = slots.get(m.slot); if (s) { try { s.pipe.dispose && (await s.pipe.dispose()); } catch (e) {} slots.delete(m.slot); } return {}; }
      case 'slots': return { slots: [...slots.entries()].map(([k, v]) => ({ slot: k, device: v.device })) };
      default: throw new Error('알 수 없는 요청: ' + m.type);
    }
  }
  return { handle, asr, lid, mt, load, warmup, slots, isPromptCopy };
}
