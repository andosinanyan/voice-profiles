# Voice profiles for Suisse Meets: concept and plan

Challenge B. Suisse Meets learns a team's voices once, with each person's explicit consent, and then names them automatically in every later meeting on any device. Transcripts no longer say "Speaker 1" and "Speaker 2", and nobody renames the same colleagues every week.

- **Working demo:** https://andosinanyan.github.io/voice-profiles/ (Chrome, Edge or Safari on a computer; the first visit downloads the models)
- **Source:** https://github.com/andosinanyan/voice-profiles

The demo runs entirely in the browser so it can be tried from a link. The production design below runs server-side in Swiss hosting. The last section lists the differences.

**One rule drives the whole design: a speaker who stays unnamed costs a click; a speaker with the wrong name destroys trust in the minutes.** The system names someone only when it is clearly sure. Otherwise it shows "Speaker N", as it does today.

---

## 1. How a voice gets in, and how consent is given

### What is stored
A voice profile is a **voiceprint**: 256 numbers calculated from a person's speech. It is not a recording. Enrollment audio is deleted as soon as the voiceprint is computed. The voiceprint is still biometric data that identifies a person. Under the revised Swiss data protection act it is sensitive personal data (revDSG Art. 5 lit. c no. 4), and under GDPR Art. 9 it is a special category. Both require **explicit** consent.

### Two ways in

**A. Self-enrollment, before any meeting.**
1. The workspace admin turns on voice profiles for the workspace. This alone stores nothing.
2. Each team member gets an invitation in their own Suisse Meets account (web, desktop or iOS app).
3. They read a short consent text in their language (DE, FR, IT, EN): what is stored, what it is used for, who can see it, and how to delete it. Then they tick "I agree".
4. They read a text aloud for about 20–30 seconds. The app checks the recording quality and asks them to repeat it if the result is too noisy.

**B. From a meeting.**
1. An unknown voice appears as "Speaker 2". The host renames it "Anna". **That name applies to this meeting only.** Nothing is stored.
2. If the host chooses "Remember this voice", Suisse Meets sends Anna a consent request in her own account. The host cannot agree on Anna's behalf.
3. Anna sees which meeting the voice came from, and can listen to a 10-second sample to check it is really her. She agrees or declines. If she agrees, her profile is created from that meeting's speech.

### What makes the consent valid
- **Freely given.** Data protection authorities question whether employees can freely consent to their employer. So declining has no consequence: everything still works, and the person simply stays "Speaker N". Admins can't see who declined, only who is enrolled.
- **Informed and recorded.** Every consent stores the version of the text, the time, the method (self-enrolled or confirmed from a meeting) and the person's account. A new version of the text asks for consent again.
- **Revocable at any time.** One click in the person's own settings deletes the voiceprint and the consent at once. The same happens when the person leaves the workspace, and after 12 months without use. The audit log keeps only the fact that a deletion happened, never the voiceprint.
- **Transparent in meetings.** The transcript header shows "Voice recognition on: 3 people identified by voice profile", and each auto-named line can be opened to see why it was named.

Before launch: a data protection impact assessment (revDSG Art. 22, GDPR Art. 35), consent texts reviewed by a Swiss data protection lawyer, and an update to the customer data processing agreement. The customer is the controller; Suisse Meets is the processor.

---

## 2. How recognition works, and how we measure that it is right

### Pipeline
1. **Split the meeting into turns.** Diarization (pyannote-style) finds who speaks when, including overlapping speech. This step works the same as today.
2. **Voiceprint per turn.** A speaker-embedding model turns each turn into 256 numbers. Two turns by the same person produce similar numbers. The demo uses WeSpeaker ResNet34-LM, the model inside pyannote. Production would test larger ones (WeSpeaker ResNet-293, ReDimNet) on our own data.
3. **Group turns into speakers.** Each speaker's voiceprint is the average of all their turns, weighted by how long they spoke. It gets more reliable as the meeting goes on.
4. **Match speakers against profiles in this workspace only.** A speaker is named only if both conditions hold:
   - the similarity to the best profile is at least the **threshold**, and
   - it beats the second-best profile by at least a **margin**, so two colleagues with similar voices don't get confused.

   Otherwise the speaker stays "Speaker N".
5. **Narrow the candidates when we know who is invited.** If the meeting comes from a calendar invite or a Teams, Zoom or Meet bot, compare first against the profiles of the invited people. Choosing among 5 people is far more reliable than choosing among 200. This fits the evidence-based naming Suisse Meets already does: a voice match becomes one more piece of evidence, next to the calendar and to "Hi, this is Anna".

### Why it works for Swiss German and on any device
- **Language:** the voiceprint describes the voice, not the words. The same profile works whether Anna speaks Swiss German, High German or French, including switching mid-sentence. Transcription stays with the current engine.
- **Device:** a voice recorded on a laptop sounds different on a phone, on Suisse Meets Pro or through a Teams bot. Three measures cover this:
  1. a profile can hold one voiceprint per device type, and matching uses the closest one;
  2. when Anna confirms an auto-named meeting, that meeting's voiceprint can be added to her profile, so it improves on the devices she actually uses (covered by her consent, and visible to her);
  3. thresholds are calibrated separately for each device type (below).

### How we measure that it's right
For every speaker in every meeting there are three outcomes:

| Outcome | Meaning | Target |
|---|---|---|
| Correct name | Named, and right | as high as possible |
| **Wrong name** | Named, and wrong | **under 1% of speaker-minutes** |
| Unnamed | Left as "Speaker N" | under 15% for enrolled people |

**Before launch:** a labeled test set of about 50 real meetings from consenting colleagues and pilot users. It covers each device (phone, Suisse Meets Pro, browser, Teams/Zoom bot), each language (Swiss German, German, French, English), 2–8 speakers, and rooms with background noise. Public meeting recordings (the AMI corpus) add volume. Thresholds are chosen per device type, as the point where wrong names stay under 1%. Any change of model or setting must pass this test set before it ships.

**After launch:** every rename of an auto-named speaker counts as a wrong name, and every manual naming of an enrolled person counts as a missed one. A dashboard shows both per device type and per week. A profile that gets corrected often asks its owner to record again.

**In the demo today:** every transcript line shows its similarity score, and lines can be marked ✓ or ✗ to compute accuracy live. The default thresholds came from a grid search on synthetic text-to-speech voices (clean, 48 kHz and simulated-room recordings). They still need calibration on real people and real meetings, which is Phase 2 of the plan.

---

## 3. How each customer's voice profiles are kept strictly separate

Separation is enforced in three layers, so a bug in one layer does not leak data:

1. **Matching scope.** A meeting can only be compared against profiles of its own workspace. There is no global index and no search across customers. A voice can never be named as another customer's employee, even if the person has a profile there too.
2. **Database.** Every profile row carries the tenant ID, and the database enforces it (row-level security in Postgres). The application cannot forget the filter.
3. **Encryption.** Each customer has its own key in a Swiss-hosted key management service. Voiceprints are encrypted with it. Deleting a customer destroys its key, which makes the voiceprints in backups unreadable as well (crypto-shredding).

In addition:
- Voiceprints are never used to train models, never leave Swiss hosting (including the AI sub-processors some other features may use), and can't be exported through the API.
- Every access is logged.
- An automated test runs on every release: a job for customer A tries to read customer B's profiles and must fail.

**In the demo:** each workspace has its own database and its own AES-256 key, created by the browser as non-extractable, so not even the page's own code can export it. Voices are only compared within a workspace. Deleting a workspace deletes its key.

---

## 4. Build, test and launch, step by step

About 8 weeks for a first public release. Each phase has an exit criterion; we only move on when it is met.

| Phase | Weeks | What happens | Exit criterion |
|---|---|---|---|
| **0. Privacy groundwork** | 1 | Impact assessment, consent texts in 4 languages reviewed by a lawyer, retention rules, processing agreement update | Legal sign-off |
| **1. Build** | 1–3 | Consent service and audit log; enrollment in web, desktop and iOS; voiceprint service in Swiss hosting; matching after diarization in the existing pipeline; switch per workspace, off by default | Works end to end on our own meetings |
| **2. Measure and calibrate** | 3–5 | Record and label the ~50-meeting test set; calibrate thresholds per device type; choose the model | Wrong name < 1%, unnamed < 15% on the test set |
| **3. Pilot** | 5–7 | Our own team, then 3 design-partner customers, opt-in; watch the correction dashboard; interview admins and team members | No wrong-name complaints, fewer renames per meeting, no privacy concerns raised |
| **4. Launch** | 8 | Available to every workspace, opt-in by the admin, then consent by each person; help article, announcement, in-app explanation | Monitoring live; enrollment rate and correction rate tracked weekly |

**After launch:**
- Improve the model with the test set as a gate.
- Plan model upgrades carefully: raw audio isn't kept, so a new model needs people to record again. The demo already flags profiles from an older model as "Record again" and stops using them.

---

## What the demo shows, and what production changes

| | Demo (this link) | Production |
|---|---|---|
| Where it runs | On the visitor's computer, in the browser: voiceprints on the CPU, speech-to-text on the graphics chip when available. Nothing is uploaded | Rented GPU servers in a Swiss data center, server-side, after the existing diarization |
| Speed | About 2 s from the end of a sentence to its line on a recent laptop; slower on older computers | The same for everyone, whatever device records the meeting |
| Voice model | WeSpeaker ResNet34 (27 MB), small enough for a laptop | Larger models (WeSpeaker ResNet-293, ReDimNet), chosen on the test set |
| Consent from a meeting | The host ticks "this person agrees" (demo shortcut) | Request sent to the person's own account |
| Storage | IndexedDB, one database and AES-256 key per workspace | Postgres with row-level security, per-tenant keys in a KMS |
| Transcription | Whisper English (in the browser) | Current Suisse Meets engine (Swiss German and 50+ languages) |
| Thresholds | Grid search on synthetic voices | Calibrated per device type on labeled meetings |

### Why GPU servers
The demo proves the pipeline on the smallest hardware there is: one browser tab on a laptop. In production the same steps run on rented GPUs in a Swiss data center:
- **More accurate:** larger voiceprint and diarization models that would not fit in a browser tab.
- **Fair to every device:** a phone, a Suisse Meets Pro recorder or a Teams bot gets the same result and the same speed. The user's own hardware no longer matters.
- **Efficient:** voiceprints are cheap next to transcription. One GPU computes them for many meetings in parallel, on the same servers that already transcribe. Phase 1 measures how many meeting-hours one GPU handles, to set the cost per meeting.
- **Swiss and private:** voiceprints and audio never leave the Swiss data center.

## Questions I would ask the team
1. How often do meetings come with a participant list (calendar invite or bot)? This decides how much narrowing the candidates to invited people (section 2, step 5) helps.
2. Do Suisse Meets Pro and the iOS app record in stereo or mono, and at what sample rate? This affects per-device calibration.
3. Which pipeline step does speaker labeling run in today, and is diarization in-house?
