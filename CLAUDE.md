# Voice Profiles: project notes for Claude

Demo for the Suisse Meets "voice profiles" job challenge (Challenge B, suisse-it.ch). It learns a team's voices once with explicit consent, then names speakers automatically in meeting transcripts instead of "Speaker 1 / Speaker 2". The audience is the hiring team: it must work from a link, alone, in a couple of minutes.

## Stack
- Static site, no build step, no backend. Plain ES modules. Hosted on GitHub Pages.
- Models via Transformers.js 4.3.2 from jsDelivr, run in two instances of `worker.js` ("speaker" and "asr") so each phrase is embedded and transcribed in parallel:
  - `onnx-community/wespeaker-voxceleb-resnet34-LM`: 256-d speaker embeddings (`AutoModel`, output `last_hidden_state`), L2-normalized. WASM, fp32, 27 MB, ~0.1 s of work per second of speech. License CC BY 4.0 (credited in README).
    - Replaced `Xenova/wavlm-base-plus-sv` (q8). The q8 WavLM gave some ordinary 3-4 s phrases a voiceprint ~0.6 from the same speaker (others ~0.97), and tiny changes to the audio (10 ms shift, 30% quieter) moved those phrases a lot: one person was split into several speakers. WeSpeaker: same phrases ~0.86, stable to 0.99.
    - Profiles store `model`. Profiles from another model are listed under Team voices as "Record again" (deletable, not used for matching); enrolling or remembering the same name replaces them.
  - `Xenova/whisper-base.en` (`tiny.en` on phones or with `?asr=tiny`): speech to text. WebGPU with encoder fp16 + decoder q4f16 when the adapter has `shader-f16`, else WASM q8; falls back to WASM if WebGPU fails to load or run. `?gpu=0` forces WASM.
  - The page is not cross-origin isolated (GitHub Pages can't set COOP/COEP), so WASM runs single-threaded per worker.
  - Measured on an M5 MacBook, 35 s clip: before (one worker, all WASM) 59 s to process; now ~1.5-2.3 s from end of a sentence to its line, live. `?gpu=0` file upload: 20 s.
- Storage: IndexedDB. DB `vp-meta` lists workspaces; each workspace has DB `vp-ws-<id>` with stores `profiles`, `audit`, `keys`. Embeddings are AES-GCM encrypted with a non-extractable per-workspace key. In-memory fallback if IndexedDB is blocked.

## Files
- `speakers.js`: `SpeakerTracker`, online clustering + profile matching. Pure logic, no DOM. Tested in `tests/`.
- `audio.js`: AudioWorklet mic tap, resample to 16 kHz (`createResampler`: windowed-sinc low-pass at 7.2 kHz, then interpolation; the old 3-sample average let 9-12 kHz through at -5 to -10 dB, folding it into the speech band), energy-based phrase splitter (VAD), file decoding. Only the live mic path resamples here; files are decoded to 16 kHz by `OfflineAudioContext`.
- `app.js`: UI, consent flows, enrollment, meeting pipeline, storage, rendering.
- `worker.js`: model loading and inference, one task at a time in order. Every meeting phrase goes through the speaker worker (even short ones with `embed: false`) so speakers are assigned in spoken order; a line shows "…" until its transcript arrives.
- `index.html`: markup and all CSS (light and dark tokens on `:root`).

## Key behavior
- Two ways in: (1) enroll before a meeting (consent, then ~20 s of speech); (2) unknown voices appear live as "Speaker N". Renaming is meeting-only unless "Remember this voice" is chosen, which requires that person's consent.
- Matching rule: name a speaker only if similarity >= `profileThreshold` AND it beats the second-best profile by `margin`. Better unnamed than wrongly named.
- New speakers get looser thresholds (`youngBonus`, fading over `youngSeconds`). `maxSpeakers` caps fragmentation. Renaming two speakers to the same name merges them.
- Default thresholds live in `DEFAULT_SETTINGS` (speakers.js) and are tunable with the in-page sliders. They are on the WeSpeaker scale (same person ~0.7-0.9, different people <0.6) and were chosen by a grid search on macOS text-to-speech voices (clean, 48 kHz, simulated room): one person, two people alternating, saved profiles, strangers. NOT yet checked on real voices. Calibrate from the scores on transcript lines; a line that started a new speaker shows how close the nearest existing speaker was.
- Short phrases: under `APP.minEmbedSeconds` (1 s) get no voiceprint and go to the previous speaker (WeSpeaker on <1 s is unreliable: right speaker 4/6). 1-2.5 s phrases get a lower bar to join a speaker (`shortBonus`), because their nearest speaker was right 6/6 but scores sat near `joinThreshold`.
- Known gap: the phrase splitter cuts on 0.66 s silences, so quick turn-taking (<0.66 s gaps) puts two people in one phrase. Splitting phrases at Whisper's sentence timestamps fixed this in a test (8/8 turns) but is not built yet.

- Look: styled after suisse-meets.ch (Inter/system font, slate text, indigo-to-violet gradient #6366f1 to #8b5cf6, 12-20 px radii, white page with a lavender glow, dark "app window" transcript with initials avatars). Own name and logo, labeled "Demo for the Suisse Meets challenge": never use their logo or name as ours. No Google Fonts (privacy). Keep copy minimal: short labels, details in tooltips (`title`).
- Icons: SVG sprite at the top of index.html (`#i-mic`, ...), `icon(name)` in app.js. Motion only via CSS under `prefers-reduced-motion: no-preference`; list items animate in once (`enterClass()`, `meeting.seen`).
- First visit: "3 steps" guide (`renderGuide()`; localStorage `vp-guide`, `vp-met`). On desktop the models (139 MB) start downloading right after load, shown in the first-visit popup (`#dl-dlg`, blurred backdrop), which opens immediately on the first visit (localStorage `vp-dl-seen`): "This is just a demo", two tiles (Demo: your computer, one-time download → Live version: rented Swiss GPUs, no download), the progress bar, "Got it". Keep it that short: reviewers don't read paragraphs. After closing, progress continues in the corner toast (`#model-box`) with a "Why?" link that reopens it. The percentage never goes backwards (file sizes arrive one by one). `modelsCached()` checks Cache Storage (`transformers-cache`) for the `.onnx` files to label it "Downloading" vs "Loading". Hero pills and the How it works table explain demo (browser, CPU/WebGPU) vs production (rented GPU servers in Switzerland); keep them in sync with CONCEPT.md.
- `[hidden] { display: none !important; }` in index.html: classes with `display` would otherwise override the hidden attribute.

## Commands
- `npm run dev`: serve locally at http://localhost:5173 (mic needs localhost or https). Port 5173 is often taken by another local project (ShikakuPuzzle); check the URL `serve` prints.
- `npm test`: unit tests for speakers.js (synthetic voices on the WeSpeaker scale) and the resampler in audio.js.

## Conventions
- Build DOM with `el()` and `textContent`. Never put user text into `innerHTML`.
- Keep it dependency-free and static so GitHub Pages works.
- UI copy in plain English, sentence case.
