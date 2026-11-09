# Voice Profiles

A working demo of the **Suisse Meets voice profiles** concept: learn a team's voices once, with explicit consent, then name speakers automatically in every later meeting, instead of "Speaker 1" and "Speaker 2".

**Try it:** https://andosinanyan.github.io/voice-profiles/ (Chrome, Edge or Safari, preferably on a computer). The page walks you through it in 3 steps.

**Concept and plan:** [CONCEPT.md](CONCEPT.md) answers the four challenge questions and lays out the build, test and launch plan.

## What it shows

| Challenge question | In the demo |
|---|---|
| How does a voice get in, and how is consent given? | **Team voices → Add a voice.** The person reads the consent text, agrees, and talks for about 20 seconds. Or, during a meeting, a new voice appears as *Speaker 2*. Renaming is just a label for that meeting, and *Remember this voice* requires that person's consent. Every consent is stored (text version, time, method) and listed in the audit log. |
| How does recognition work, and how do you know it's right? | Phrases → voiceprints (WeSpeaker) → grouped into speakers live → matched to saved profiles only when the match is above the threshold **and** clearly better than the next best. Otherwise the speaker stays unnamed. Mark lines ✓/✗ to measure accuracy; every line shows its similarity score. |
| How are customers kept separate? | Each workspace (= customer) has its own database and its own non-exportable AES-256 key. Voices are only compared within a workspace. |
| Deletion? | Delete removes the voiceprint and consent record at once. The audit log keeps only the fact that it happened. |

## Where it runs

**The demo runs on your computer.** Small models sized for a laptop run inside the browser: voiceprints on the CPU, speech-to-text on the graphics chip via WebGPU when the browser allows it, otherwise on the CPU. Nothing is uploaded, so it works from a link with no server.

**In production it moves to rented GPU servers in a Swiss data center.** The same pipeline then runs much larger models (more accurate names, Swiss German and 50+ languages, overlapping speech), at the same speed for everyone, whatever device records the meeting, and with nothing to download. The *How it works* tab in the app compares the two side by side.

## Try it in 2 minutes

1. **Team voices → Add a voice:** agree, read the text aloud until the bar is full.
2. **Meeting → Start recording:** talk. Lines appear with your name. Have someone else talk too: they appear as *Speaker 2*.
3. **Rename** Speaker 2, choose *Remember this voice*, and confirm consent. Start a new meeting and they're recognized.
4. Or **Upload recording**: any English meeting recording up to 20 minutes.

The first visit downloads two open-source models (about 140 MB, cached after that).

## How it's built

- Static site, no backend. Everything runs in the browser and nothing is uploaded.
- `speakers.js`: online speaker clustering and profile matching (pure logic, unit-tested).
- `audio.js`: microphone capture, 16 kHz resampling, phrase detection, file decoding.
- `worker.js`: runs the models off the main thread using [Transformers.js](https://github.com/huggingface/transformers.js), in two workers that work in parallel: `onnx-community/wespeaker-voxceleb-resnet34-LM` (speaker voiceprints, on the CPU; [WeSpeaker](https://github.com/wenet-e2e/wespeaker) ResNet34-LM, CC BY 4.0) and `Xenova/whisper-base.en` (speech to text, on the GPU via WebGPU when the browser supports it, otherwise on the CPU).
- `app.js`: UI, consent, IndexedDB storage, per-workspace encryption.

URL options: `?asr=tiny` (faster speech-to-text), `?enroll=30` (seconds recorded for a profile), `?gpu=0` (run speech-to-text on the CPU, to compare).

## Production version (differs from this demo)

- Recognition runs server-side on rented GPU servers in Swiss hosting, with a stronger model (e.g. ReDimNet / WeSpeaker ResNet-293) and proper diarization (pyannote) for overlapping speech.
- Consent is confirmed by the person from their own account, never by the meeting host.
- Per-tenant keys in a KMS, tenant ID enforced at the database layer, and no raw enrollment audio kept.
- Accuracy measured on a labeled test set per device type (phone, recorder, browser): correct name, wrong name (target near 0), unnamed.
