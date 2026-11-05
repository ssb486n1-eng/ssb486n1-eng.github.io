/* =====================================================================
   InterpCore — API 없는 실시간 통역의 순수 로직 (DOM·네트워크 의존 없음)
   브라우저: window.InterpCore / Node 테스트: require('./core.js')

   구성
   1) LANGS          : 언어별 코드 매핑 (BCP-47 / NLLB(FLORES-200) / Whisper)
   2) Agreement      : 스트리밍 ASR 가설 안정화 (LocalAgreement-n)
   3) segment()      : 절(clause) 단위 분할 — '살라미' 방식 동시통역
   4) Glossary       : 전문용어 강제(소스 주입) + 칩 표시
   5) hallucination  : Whisper 무음 환각 필터
   6) similarity     : 스피커 에코(자기 통역음 재인식) 차단
   7) 기타           : 지연 통계, 낭독 속도 적응, 내보내기
   ===================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.InterpCore = factory();
})(typeof self !== 'undefined' ? self : globalThis, function () {
  'use strict';

  /* ---------------- 1) 언어 ---------------- */
  // nllb: FLORES-200 코드 (NLLB-200), whisper: Whisper 언어 코드
  const LANGS = {
    ko: { name: '한국어',     bcp: 'ko-KR', nllb: 'kor_Hang', whisper: 'ko', spaced: true },
    en: { name: '영어',       bcp: 'en-US', nllb: 'eng_Latn', whisper: 'en', spaced: true },
    cs: { name: '체코어',     bcp: 'cs-CZ', nllb: 'ces_Latn', whisper: 'cs', spaced: true },
    ja: { name: '일본어',     bcp: 'ja-JP', nllb: 'jpn_Jpan', whisper: 'ja', spaced: false },
    zh: { name: '중국어',     bcp: 'zh-CN', nllb: 'zho_Hans', whisper: 'zh', spaced: false },
    de: { name: '독일어',     bcp: 'de-DE', nllb: 'deu_Latn', whisper: 'de', spaced: true },
    fr: { name: '프랑스어',   bcp: 'fr-FR', nllb: 'fra_Latn', whisper: 'fr', spaced: true },
    es: { name: '스페인어',   bcp: 'es-ES', nllb: 'spa_Latn', whisper: 'es', spaced: true },
    ru: { name: '러시아어',   bcp: 'ru-RU', nllb: 'rus_Cyrl', whisper: 'ru', spaced: true },
    pl: { name: '폴란드어',   bcp: 'pl-PL', nllb: 'pol_Latn', whisper: 'pl', spaced: true },
    sk: { name: '슬로바키아어', bcp: 'sk-SK', nllb: 'slk_Latn', whisper: 'sk', spaced: true },
    vi: { name: '베트남어',   bcp: 'vi-VN', nllb: 'vie_Latn', whisper: 'vi', spaced: true },
  };

  /* ---------------- 공통 토큰화 ---------------- */
  function isSpaced(lang) { return !LANGS[lang] || LANGS[lang].spaced !== false; }
  function toks(text, lang) {
    const t = String(text || '').trim();
    if (!t) return [];
    return isSpaced(lang) ? t.split(/\s+/) : Array.from(t.replace(/\s+/g, ''));
  }
  function join(arr, lang) { return isSpaced(lang) ? arr.join(' ') : arr.join(''); }
  // 비교용 정규화: 소문자, 앞뒤 구두점 제거
  function normTok(w) {
    return String(w).toLowerCase().replace(/^[\p{P}\p{S}]+|[\p{P}\p{S}]+$/gu, '');
  }

  /* ---------------- 2) LocalAgreement ----------------
     스트리밍 Whisper / Web Speech interim 결과는 매번 앞부분이 바뀔 수 있다.
     연속된 n개 가설이 공통으로 동의하는 앞부분(prefix)만 '확정'하고,
     한 번 확정된 부분은 되돌리지 않는다.
     근거: Macháček, Dabre, Bojar (2023) "Turning Whisper into Real-Time
     Transcription System", IJCNLP-AACL demo — LocalAgreement-n 정책. */
  class Agreement {
    constructor(lang, n) { this.lang = lang; this.n = Math.max(2, n || 2); this.hist = []; this.committed = []; }
    reset() { this.hist = []; this.committed = []; }
    update(hyp) {
      const h = toks(hyp, this.lang);
      this.hist.push(h);
      if (this.hist.length > this.n) this.hist.shift();
      const c = this.committed.length;
      // 확정된 부분이 새 가설과 어긋나는지 (앞 단어 교정) — 어긋나도 확정은 유지
      let aligned = 0;
      while (aligned < c && aligned < h.length && normTok(h[aligned]) === normTok(this.committed[aligned])) aligned++;
      if (this.hist.length >= this.n && aligned === c) {
        let k = h.length;
        for (const p of this.hist) {
          let i = 0;
          while (i < k && i < p.length && normTok(p[i]) === normTok(h[i])) i++;
          k = i;
        }
        if (k > c) this.committed = this.committed.concat(h.slice(c, k));
      }
      const tail = h.slice(this.committed.length);
      return { committed: join(this.committed, this.lang), tail: join(tail, this.lang) };
    }
    // 발화 종료: 최종 가설 중 '이미 확정·번역된 부분 이후'만 이어 붙인다.
    // 최종 가설이 앞부분을 고쳤으면(2단계 인식 등) 편집거리 정렬로 확정분의 끝 위치를 찾는다.
    finalize(finalText) {
      const f = toks(finalText, this.lang);
      const c = this.committed.length;
      let i = 0;
      while (i < c && i < f.length && normTok(f[i]) === normTok(this.committed[i])) i++;
      const end = i === c ? c : alignEnd(this.committed, f);
      this.committed = this.committed.concat(f.slice(end));
      return join(this.committed, this.lang);
    }
  }

  // 확정 단어열 A 전체를 B 의 앞부분 B[0..j) 에 정렬했을 때 점수가 최대인 j (semi-global alignment).
  // 일치 +2, 접두 일치(조사 차이 등) +1, 치환·삽입·삭제 -1. 편집거리 최소화는 짧은 j 를 선호해
  // 확정분 끝 단어를 놓치므로 일치에 가점을 준다. 동률이면 작은 j (내용 누락보다 중복이 안전)
  function alignEnd(A, B) {
    const n = A.length, m = B.length;
    const sc = (x, y) => {
      const a = normTok(x), b = normTok(y);
      if (a === b) return 2;
      if (a.length >= 2 && b.length >= 2 && (a.startsWith(b) || b.startsWith(a))) return 1;
      return -1;
    };
    let prev = Array.from({ length: m + 1 }, (_, j) => -j);
    for (let r = 1; r <= n; r++) {
      const cur = [-r];
      for (let j = 1; j <= m; j++) cur[j] = Math.max(prev[j] - 1, cur[j - 1] - 1, prev[j - 1] + sc(A[r - 1], B[j - 1]));
      prev = cur;
    }
    let best = 0;
    for (let j = 1; j <= m; j++) if (prev[j] > prev[best]) best = j;
    return best;
  }

  /* ---------------- 3) 절 단위 분할 ----------------
     문장 끝을 기다리지 않고 의미 단위(절)가 완성되는 즉시 번역·낭독한다.
     동시통역사의 'salami technique'(긴 문장을 짧은 단위로 잘라 바로 옮김)과
     wait-k 정책(Ma et al., 2019, ACL "STACL")의 최대 대기 길이를 결합. */
  // 연결어미: 단일 음절('면','서','고')은 명사(측면·문서·창고)와 겹치므로 활용형만 사용
  const KO_CONNECTIVE = /(?:하고|있고|없고|했고|되고|됐고|이고|않고|었고|았고|였고|겠고|하며|있으며|으며|이며|되며|면서|는데|은데|인데|던데|지만|어서|아서|해서|여서|돼서|니까|으니|므로|으면|이면|하면|되면|려면|다면|도록|다가|자마자|거나|든지|듯이)[,，]?$/;
  // 종결어미: 단독 '요'는 '필요' 등과 겹쳐 제외
  const KO_FINAL = /(?:니다|어요|아요|에요|예요|세요|네요|군요|해요|죠|했다|한다|된다|있다|없다|이다|였다|었다|았다|겠다)[.?!]?$/;
  const CONJ = {
    en: ['and', 'but', 'so', 'because', 'which', 'when', 'if', 'although', 'while', 'then', 'or', 'however', 'whereas', 'unless', 'since'],
    cs: ['a', 'ale', 'protože', 'který', 'která', 'které', 'když', 'pokud', 'že', 'nebo', 'jestli', 'aby', 'takže', 'proto'],
    de: ['und', 'aber', 'weil', 'dass', 'wenn', 'oder', 'denn', 'obwohl', 'damit'],
    fr: ['et', 'mais', 'parce', 'quand', 'si', 'ou', 'donc', 'puisque', 'lorsque'],
    es: ['y', 'pero', 'porque', 'cuando', 'si', 'o', 'aunque', 'entonces'],
    ru: ['и', 'но', 'потому', 'когда', 'если', 'или', 'который', 'поэтому'],
    pl: ['i', 'ale', 'ponieważ', 'kiedy', 'jeśli', 'lub', 'który', 'więc'],
    sk: ['a', 'ale', 'pretože', 'ktorý', 'keď', 'ak', 'alebo', 'takže'],
    vi: ['và', 'nhưng', 'vì', 'khi', 'nếu', 'hoặc'],
  };
  const STRONG_END = /[.!?。？！…]["'”’)\]]?$/;
  const WEAK_END = /[,，;；:：、]["'”’)\]]?$/;

  function segment(text, lang, opt) {
    // clauses=false 이면 문장 부호만으로 자른다 (정확도 우선 모드)
    const o = Object.assign({ maxWords: lang === 'ko' ? 7 : 12, minWords: lang === 'ko' ? 2 : 3, maxChars: 40, final: false, clauses: true }, opt || {});
    const w = toks(text, lang);
    const out = [];
    let start = 0;
    const spaced = isSpaced(lang);
    const conj = new Set(CONJ[lang] || []);
    for (let i = 0; i < w.length; i++) {
      const len = i - start + 1; // 현재 세그먼트 단어 수 (i 포함)
      const tok = w[i];
      let cutAfter = false;
      if (STRONG_END.test(tok)) cutAfter = true;
      else if (!o.clauses) cutAfter = false;
      else if (WEAK_END.test(tok) && len >= o.minWords) cutAfter = true;
      else if (lang === 'ko' && len >= o.minWords + 1 && tok.length >= 2 && (KO_CONNECTIVE.test(tok) || KO_FINAL.test(tok))) cutAfter = true;
      else if (!spaced && /[。，、！？]/.test(tok)) cutAfter = true;
      // 접속사 '앞'에서 자르기 (다음 토큰이 접속사)
      const next = w[i + 1];
      if (!cutAfter && o.clauses && spaced && next && conj.has(normTok(next)) && len >= o.minWords + 1) cutAfter = true;
      // wait-k 상한: 경계가 없어도 너무 오래 기다리지 않는다
      const limit = spaced ? o.maxWords : o.maxChars;
      if (!cutAfter && len >= limit) cutAfter = true;
      if (cutAfter) { out.push(join(w.slice(start, i + 1), lang)); start = i + 1; }
    }
    let rest = join(w.slice(start), lang);
    if (o.final && rest) { out.push(rest); rest = ''; }
    return { segments: out, rest };
  }

  /* ---------------- 4) 용어집 ----------------
     NMT 는 낯선 전문용어를 일반어로 바꾸는 경향이 있다. 번역 전에 원문의
     용어를 '목표 언어 표준 대역'으로 바꿔 넣으면(code-mixed input) 대부분의
     NMT 가 그대로 복사한다 → 용어 일관성 확보. 결과에 대역이 남았는지 검증. */
  const RE_ESC = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const strip = s => String(s || '').replace(/\s*[\(（].*?[\)）]/g, '').split(' / ')[0].trim();

  function buildGlossary(rows) {
    return (rows || []).map(r => {
      const [code, ko, en, cs] = r;
      return { code: (code || '').trim(), ko: strip(ko), en: strip(en), cs: strip(cs),
               full: { ko: String(ko || ''), en: String(en || ''), cs: String(cs || '') } };
    }).filter(g => g.ko || g.en);
  }
  // 사용자 용어 입력: 한 줄에 "코드|한국어|영어|체코어" 또는 "한국어=영어=체코어"
  function parseUserGlossary(text) {
    const rows = [];
    String(text || '').split(/\r?\n/).forEach(line => {
      const l = line.trim(); if (!l || l.startsWith('#')) return;
      const p = l.includes('|') ? l.split('|') : ['', ...l.split('=')];
      if (p.length >= 3) rows.push([p[0] || '', p[1] || '', p[2] || '', p[3] || '']);
    });
    return rows;
  }
  function termIn(g, lang) { return lang === 'ko' ? g.ko : lang === 'en' ? g.en : lang === 'cs' ? g.cs : ''; }
  function findRe(term, lang) {
    if (lang === 'ko') return new RegExp(RE_ESC(term), 'g');
    return new RegExp('(^|[^\\p{L}\\p{N}])(' + RE_ESC(term) + ')(?=$|[^\\p{L}\\p{N}])', 'giu');
  }
  function matches(g, text, lang) {
    const t = termIn(g, lang);
    if (t) {
      if (lang === 'ko' && t.length >= 2 && /[가-힣]/.test(t) && text.includes(t)) return { by: 'term', term: t };
      if (lang !== 'ko' && t.length >= 4 && findRe(t, lang).test(text)) return { by: 'term', term: t };
    }
    if (g.code && g.code.length >= 2 && /^[A-Z0-9&]+$/.test(g.code) &&
        new RegExp('(^|[^A-Za-z])' + RE_ESC(g.code) + '([^A-Za-z]|$)').test(text)) return { by: 'code', term: g.code };
    return null;
  }
  function findTerms(text, lang, gl) {
    const hits = [];
    for (const g of gl) { const m = matches(g, text, lang); if (m) hits.push({ g, m }); }
    hits.sort((a, b) => b.m.term.length - a.m.term.length);
    const seen = new Set();
    return hits.filter(h => { const k = h.g.ko + '|' + h.g.en; if (seen.has(k)) return false; seen.add(k); return true; });
  }
  // 강제 적용 대상: 오인 위험이 큰 짧은 일반어는 제외 (칩 표시만)
  function enforceable(h, src) {
    if (h.m.by === 'code') return true;
    if (src === 'ko') return h.m.term.length >= 3;
    // 영어·체코어 원문: 여러 단어 전문용어만 (pressure·temperature 같은 일반 단어를 치환하면 문장이 섞여 번역이 나빠짐 — 브라우저 실측)
    return h.m.term.includes(' ');
  }
  function applyGlossary(text, src, tgt, gl, enforce) {
    const hits = findTerms(text, src, gl);
    let out = text;
    const forced = [];
    if (enforce !== false && ['ko', 'en', 'cs'].includes(tgt)) {
      const slots = [];
      for (const h of hits) {
        const target = termIn(h.g, tgt);
        if (!target || !enforceable(h, src)) continue;
        const re = h.m.by === 'code'
          ? new RegExp('(^|[^A-Za-z])(' + RE_ESC(h.m.term) + ')(?=[^A-Za-z]|$)', 'g')
          : findRe(h.m.term, src);
        let used = false;
        out = out.replace(re, (...a) => {
          used = true;
          const idx = slots.push(target) - 1;
          const mark = '\u0001' + idx + '\u0002';
          return (src === 'ko' && h.m.by !== 'code') ? mark : (a[1] || '') + mark;
        });
        if (used) forced.push({ g: h.g, target });
      }
      out = out.replace(/\u0001(\d+)\u0002/g, (_, i) => slots[+i]);
    }
    return { text: out, hits: hits.map(h => h.g), forced };
  }
  function verifyTerms(output, forced) {
    const low = String(output || '').toLowerCase();
    return forced.map(f => ({ term: f.target, ok: low.includes(f.target.toLowerCase()) }));
  }

  /* ---------------- 5) Whisper 환각 필터 ----------------
     Whisper 는 무음·잡음 구간에서 학습 데이터의 자막 문구를 지어내는 경향이
     보고되어 있다 (Koenecke et al., 2024, ACM FAccT "Careless Whisper").
     대표 문구 + 동일 토큰 반복 루프를 제거한다. */
  const HALLU = [
    /^(감사합니다|고맙습니다|시청해\s?주셔서\s?감사합니다|구독(과)?\s?좋아요.*|다음\s?영상에서\s?만나요|MBC\s?뉴스.*|이\s?시각\s?세계였습니다)[.!\s]*$/i,
    /^(thank you( (so much|very much))?( for watching)?|thanks for watching|please subscribe.*|subtitles by.*|you)[.!\s]*$/i,
    /^(děkuji( za pozornost)?|titulky.*)[.!\s]*$/i,
    /^[\s♪♫.…\-–—]*$/,
    /^\[(.*?)\]$|^\((.*?)\)$/,
    /^[\[(]?\s*(BLANK_AUDIO|MUSIC|NOISE|SILENCE|INAUDIBLE|음악|박수)\s*[\])]?[.!\s]*$/i,   // 무음 표기 (실측: '[BLANK_AUDIO')
  ];
  function isHallucination(text) {
    const t = String(text || '').trim();
    if (!t) return true;
    if (HALLU.some(re => re.test(t))) return true;
    // 글자 단위 반복 루프 ('지지지지…', 'same.com/same.com/…') 가 본문 대부분을 차지하면 환각
    const loop = t.match(/(.{1,12}?)\1{4,}/u);
    if (loop && loop[0].length >= t.length * 0.4) return true;
    const w = t.split(/\s+/);
    if (w.length >= 6) {
      const freq = {}; w.forEach(x => { const k = normTok(x); freq[k] = (freq[k] || 0) + 1; });
      if (Math.max(...Object.values(freq)) / w.length > 0.6) return true;
    }
    return false;
  }
  // 반복 루프 축약: "네 네 네 네 네" → "네"
  function collapseRepeats(text) {
    return String(text || '').replace(/(.{2,12}?)\1{3,}/gu, '$1').replace(/(\b[\p{L}\p{N}]+(?:\s+[\p{L}\p{N}]+){0,3}\b)(?:[\s,]+\1\b){2,}/gu, '$1');
  }

  /* ---------------- 6) 에코 차단 ----------------
     스피커로 나간 통역 음성이 마이크로 다시 인식되면 무한 반복이 생긴다.
     최근 낭독 문장과 문자 bigram Dice 유사도가 높으면 버린다. */
  function bigrams(s) {
    const t = String(s || '').toLowerCase().replace(/[\s\p{P}]+/gu, '');
    const m = new Map();
    for (let i = 0; i < t.length - 1; i++) { const b = t.substr(i, 2); m.set(b, (m.get(b) || 0) + 1); }
    return { m, n: Math.max(0, t.length - 1) };
  }
  function similarity(a, b) {
    const A = bigrams(a), B = bigrams(b);
    if (!A.n || !B.n) return 0;
    let inter = 0;
    for (const [k, v] of A.m) if (B.m.has(k)) inter += Math.min(v, B.m.get(k));
    return (2 * inter) / (A.n + B.n);
  }
  // 인식된 텍스트가 최근 낭독문(들)의 일부인지: 짧은 인식문은 포함 관계로 판단
  function isEcho(heard, spokenList, threshold) {
    const th = threshold || 0.55;
    const h = String(heard || '').trim();
    if (h.length < 3) return false;
    for (const s of spokenList || []) {
      if (!s) continue;
      if (similarity(h, s) >= th) return true;
      const hs = h.toLowerCase().replace(/[\s\p{P}]+/gu, ''), ss = String(s).toLowerCase().replace(/[\s\p{P}]+/gu, '');
      if (hs.length >= 6 && ss.includes(hs)) return true;
    }
    return false;
  }

  /* ---------------- 7) 지연 통계 / 낭독 속도 ---------------- */
  class Stat {
    constructor(alpha) { this.a = alpha || 0.25; this.v = null; this.n = 0; this.last = null; }
    push(x) { if (!isFinite(x)) return; this.last = x; this.n++; this.v = this.v == null ? x : this.v + this.a * (x - this.v); }
  }
  // 대기 중인 낭독 분량이 쌓이면 속도를 올린다 (통역사의 압축 전략을 단순화)
  function adaptiveRate(base, backlogChars, lang) {
    const cps = lang === 'ko' || lang === 'ja' || lang === 'zh' ? 7 : 14; // 초당 글자 수(대략)
    const backlogSec = backlogChars / (cps * base);
    const r = base * (1 + 0.12 * Math.max(0, backlogSec - 2));
    return Math.min(Math.max(base, r), Math.max(base, 1.7));
  }

  // 오디오 RMS (VAD 용)
  function rms(buf) { let s = 0; for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i]; return Math.sqrt(s / (buf.length || 1)); }

  // 에너지 기반 VAD (적응형 잡음 바닥)
  class Vad {
    constructor(o) {
      // longAfterMs 이상 말이 이어지면 짧은 쉼(shortHangMs)에서도 끊는다 → 긴 발화의 지연·재인식 비용 감소
      this.o = Object.assign({ ratio: 3.0, minAbs: 0.008, hangMs: 650, minSpeechMs: 250, longAfterMs: 5000, shortHangMs: 280 }, o || {});
      this.noise = 0.004; this.speaking = false; this.lastVoice = 0; this.voiceStart = 0;
    }
    // frame RMS 와 시각(ms) → 'start' | 'end' | null
    push(r, t) {
      const th = Math.max(this.o.minAbs, this.noise * this.o.ratio);
      const voiced = r > th;
      if (!this.speaking) this.noise = this.noise * 0.97 + Math.min(r, th) * 0.03;
      if (voiced) {
        if (!this.voiceStart) this.voiceStart = t;
        this.lastVoice = t;
        if (!this.speaking && t - this.voiceStart >= this.o.minSpeechMs) { this.speaking = true; return 'start'; }
      } else if (!this.speaking) {
        this.voiceStart = 0;
      } else if (t - this.lastVoice > (t - this.voiceStart > this.o.longAfterMs ? Math.min(this.o.shortHangMs, this.o.hangMs) : this.o.hangMs)) {
        this.speaking = false; this.voiceStart = 0; return 'end';
      }
      return null;
    }
  }

  /* ---------------- 8) 인식 정확도 보강 ---------------- */
  // 약어의 한국어 읽기 (ASR 이 'RCS' 를 '알씨에스'로 받아쓰는 경우 교정)
  const KO_LETTER = { A:'에이', B:'비', C:'씨', D:'디', E:'이', F:'에프', G:'지', H:'에이치', I:'아이', J:'제이', K:'케이', L:'엘', M:'엠',
    N:'엔', O:'오', P:'피', Q:'큐', R:'알', S:'에스', T:'티', U:'유', V:'브이', W:'더블유', X:'엑스', Y:'와이', Z:'제트' };
  function acronymKo(code) {
    if (!/^[A-Z]{2,5}$/.test(code)) return null;
    return code.split('').map(c => KO_LETTER[c]);
  }
  // 교정 규칙 캐시 (용어집이 바뀔 때만 재생성)
  const corrCache = new WeakMap();
  function correctionRules(gl) {
    if (corrCache.has(gl)) return corrCache.get(gl);
    const ko = [], en = [];
    const seen = new Set();
    for (const g of gl) {
      // (a) 한국어 용어 사이 띄어쓰기 변이: '냉각 탑' → '냉각탑'
      if (g.ko && /^[가-힣]{3,}$/.test(g.ko) && !seen.has(g.ko)) {
        seen.add(g.ko);
        ko.push({ re: new RegExp(g.ko.split('').map(RE_ESC).join('\\s?'), 'g'), to: g.ko, need: /\s/ });
      }
      // (b) 약어: '알씨에스'·'알 씨 에스' → 'RCS' (2글자 약어는 오인 위험이 커서 3글자 이상만)
      const rd = g.code && g.code.length >= 3 && acronymKo(g.code);
      if (rd && !seen.has('@' + g.code)) {
        seen.add('@' + g.code);
        ko.push({ re: new RegExp('(^|[^\\uAC00-\\uD7A3])(' + rd.map(RE_ESC).join('\\s?') + ')(?=$|[^\\uAC00-\\uD7A3]|[은는이가을를의에도와과로])', 'g'), to: g.code, keepLead: true });
      }
      // (c) 영어 약어: 'r c s' / 'R.C.S.' / 'rcs' → 'RCS'
      if (g.code && /^[A-Z]{2,5}$/.test(g.code) && !seen.has('#' + g.code)) {
        seen.add('#' + g.code);
        en.push({ re: new RegExp('\\b' + g.code.split('').join('[\\s.]?') + '\\b\\.?', 'gi'), to: g.code });
      }
    }
    const r = { ko, en };
    corrCache.set(gl, r);
    return r;
  }
  function correctAsr(text, lang, gl) {
    let t = String(text || '');
    if (!t || !gl || !gl.length) return t;
    const R = correctionRules(gl);
    if (lang === 'ko') {
      for (const r of R.ko) {
        if (r.need && !r.need.test(t)) continue;
        t = r.keepLead ? t.replace(r.re, (_, lead) => lead + r.to) : t.replace(r.re, r.to);
      }
    }
    if (lang !== 'ko' || /[A-Za-z]/.test(t)) for (const r of R.en) t = t.replace(r.re, r.to);
    return t;
  }

  // Whisper 프롬프트에 넣을 용어: 사용자 용어 → 최근 등장 용어 → 약어·긴 용어 순, 글자 수 제한
  function promptTerms(gl, lang, o) {
    const opt = Object.assign({ recent: [], userCount: 0, max: 24 }, o || {});
    const out = [], seen = new Set();
    const add = t => { if (t && !seen.has(t) && out.length < opt.max) { seen.add(t); out.push(t); } };
    const term = g => (lang === 'ko' ? g.ko : lang === 'en' ? g.en : lang === 'cs' ? g.cs : '');
    // 사용자 용어는 목록 끝에 붙어 있음
    for (let i = gl.length - opt.userCount; i < gl.length; i++) if (i >= 0) add(term(gl[i]));
    for (const g of opt.recent) { add(term(g)); if (g.code && lang !== 'ko') add(g.code); }
    // 용어집 순서대로: 약어가 있는 용어 + 오인되기 쉬운 긴 전문용어 (짧은 일반어는 제외)
    for (const g of gl) {
      if (g.code && g.code.length >= 2 && /^[A-Z&]+$/.test(g.code)) add(lang === 'ko' ? term(g) : g.code);
      const t = term(g);
      if (t && (lang === 'ko' ? t.length >= 3 : (t.includes(' ') || t.length >= 7))) add(t);
    }
    return out.filter(Boolean);
  }
  // 프롬프트 = 용어 목록 + 직전 발화(문맥). Whisper 프롬프트 한도(223 토큰)보다 충분히 짧게
  function buildPrompt(o) {
    const terms = (o.terms || []).join(', ');
    const prev = String(o.prev || '').slice(-(o.prevChars || 120));
    const p = (terms ? terms + '.' : '') + (prev ? ' ' + prev : '');
    return p.trim().slice(-(o.maxChars || 320));
  }

  // 발화 길이에 맞춘 생성 토큰 상한 — 환각 루프·불필요한 디코딩 시간 차단
  function maxTokensFor(sec, lang) {
    const rate = lang === 'ko' || lang === 'ja' ? 14 : lang === 'zh' ? 12 : lang === 'ru' || lang === 'cs' || lang === 'pl' || lang === 'sk' ? 9 : 6;
    return Math.min(220, Math.ceil(sec * rate * 1.3) + 8);
  }

  // 후보 언어 판정: 1·2위 logit 차이가 작으면 '불확실' → 직전 화자 유지 (오판으로 방향이 뒤집히는 것 방지)
  function decideLang(scores, cands, prev, margin) {
    const m = margin == null ? 1.0 : margin;
    const r = cands.map(c => [c, scores[c] == null ? -1e9 : scores[c]]).sort((a, b) => b[1] - a[1]);
    const gap = r.length > 1 ? r[0][1] - r[1][1] : Infinity;
    if (gap < m && prev && cands.includes(prev)) return { lang: prev, gap, uncertain: true };
    return { lang: r[0][0], gap, uncertain: gap < m };
  }

  // Web Speech 대안(alternatives) 재순위: 신뢰도 + 용어집 적중 가산
  function rerank(alts, lang, gl) {
    if (!alts || !alts.length) return '';
    let best = alts[0], bs = -1e9;
    alts.forEach((a, i) => {
      const hits = gl ? findTerms(a.transcript || '', lang, gl).length : 0;
      const sc = (a.confidence || 0) + 0.15 * hits - i * 0.01;
      if (sc > bs) { bs = sc; best = a; }
    });
    return best.transcript || '';
  }

  // 기기 등급별 엔진 계획 — 폰·태블릿에서도 실시간이 되도록 모델 크기와 스레드를 고른다
  function devicePlan(f) {
    const mobile = !!f.mobile, gpu = !!f.webgpu, cores = f.cores || 4, mem = f.mem || 4;
    const tier = gpu && !mobile && mem >= 4 ? 'high' : (gpu || (!mobile && cores >= 8)) ? 'mid' : 'low';
    const W = n => 'onnx-community/whisper-' + n;
    const plan = {
      tier, mobile,
      // 2단계 인식: 빠른 초안(스트리밍) + 정확한 확정(발화 끝 1회) — Sainath et al., 2019 two-pass
      asrDraft: tier === 'low' ? null : W('tiny'),
      asrFinal: tier === 'high' ? W('small') : tier === 'mid' ? W('base') : (mobile ? W('tiny') : W('base')),
      mtMulti: mobile || mem < 4 ? 'Xenova/m2m100_418M' : 'Xenova/nllb-200-distilled-600M',
      numThreads: f.isolated ? Math.max(1, Math.min(4, cores - 1)) : 1,
      stepMs: tier === 'high' ? 450 : tier === 'mid' ? 650 : 900,
    };
    return plan;
  }
  // 확정 인식문이 초안과 많이 다르면 전체 문장을 다시 번역해 표시를 정정
  function needsRevision(draft, final) {
    const a = String(draft || '').trim(), b = String(final || '').trim();
    if (!a || !b) return false;
    return similarity(a, b) < 0.85;
  }

  /* ---------------- 내보내기 ---------------- */
  function pad(n) { return String(n).padStart(2, '0'); }
  function hms(ts) { const d = new Date(ts); return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds()); }
  function toTxt(hist) {
    let t = '실시간 통역 기록 (로컬 엔진)\n생성: ' + new Date().toLocaleString() + '\n' + '='.repeat(40) + '\n\n';
    hist.forEach(e => {
      t += '[' + hms(e.t) + '] ' + (e.role === 'A' ? '나' : '상대') + ' (' + (LANGS[e.src] ? LANGS[e.src].name : e.src) + ')\n  ' + e.text + '\n';
      if (e.tr) t += '  → ' + (LANGS[e.tgt] ? LANGS[e.tgt].name : e.tgt) + ': ' + e.tr + '\n';
      t += '\n';
    });
    return t;
  }
  function toCsv(hist) {
    const c = s => '"' + String(s == null ? '' : s).replace(/"/g, '""') + '"';
    let t = '﻿시간,화자,원문언어,원문,통역언어,통역,지연(ms)\n';
    hist.forEach(e => {
      t += [c(new Date(e.t).toLocaleString()), c(e.role === 'A' ? '나' : '상대'), c(e.src), c(e.text), c(e.tgt), c(e.tr), c(e.lat || '')].join(',') + '\n';
    });
    return t;
  }

  return {
    LANGS, toks, join, normTok, Agreement, segment, buildGlossary, parseUserGlossary, findTerms,
    applyGlossary, verifyTerms, isHallucination, collapseRepeats, similarity, isEcho, Stat,
    adaptiveRate, rms, Vad, toTxt, toCsv, hms,
    correctAsr, acronymKo, promptTerms, buildPrompt, maxTokensFor, decideLang, rerank, devicePlan, needsRevision,
  };
});
