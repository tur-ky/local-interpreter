# Local Interpreter

[![build](https://github.com/tur-ky/local-interpreter/actions/workflows/build.yml/badge.svg)](https://github.com/tur-ky/local-interpreter/actions/workflows/build.yml)

A Windows desktop app that listens to your system audio or microphone and shows
a live English translation of what is being said — entirely on your own machine.
No cloud APIs, no external LLMs, no account, no telemetry.

### Download

**[Get the installer from the latest release »](https://github.com/tur-ky/local-interpreter/releases/latest)**


Windows 10/11 64-bit. Nothing else to install — Python, Qt, the CUDA 12 runtime
and cuDNN 9 all ship inside it. It is not code-signed, so SmartScreen will show
"Windows protected your PC" the first time: *More info* → *Run anyway*.

---

* **UI** — PyQt6, dark theme
* **Audio** — [`soundcard`](https://github.com/bastibe/SoundCard) (WASAPI): system
  loopback, microphone, or both mixed together
* **Speech** — [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) on
  CUDA with `compute_type="float16"` and Silero VAD
* **Translation** — Whisper's own `task="translate"`, i.e. audio → English in one
  pass, no separate MT model

---

## Files

| File | What it is |
| --- | --- |
| `local_interpreter.py` | The entire application — one file, no local imports |
| `LocalInterpreter.spec` | PyInstaller recipe (bundles CUDA, cuDNN and the MSVC runtime) |
| `installer.iss` | Inno Setup script that produces the installer |
| `build.ps1` | icon → frozen app → installer, in one command |
| `make_icon.py` | Generates `assets/app.ico` |

---

## Using the app

**Mode** — `Live` shows the sentence *as it is being spoken*: the running
utterance is re-decoded about once a second and appears in italics under the
transcript, then it is replaced by a higher-quality final pass when the speaker
pauses. `Manual` records from Start to Stop and translates the whole take in one
pass (slightly more accurate, since Whisper sees the full context).

**Languages** — the left dropdown is the spoken language (`Auto-detect` is fine),
the right one is the target. Whisper translates *into English* only; that is a
property of the model, not of this app, so with any other target the panel shows
the transcript in the spoken language and says so.

**Audio source** — `System` captures whatever is coming out of your speakers
(the other side of a call), `Mic` captures you, `Both` mixes the two.

**Settings → Transcription**

* *Expected languages* restricts detection. Tick exactly one and the model is
  locked to it (`language="es"`), which is faster and noticeably more accurate.
  Tick several plus *Only these languages* and each utterance is detected, but
  only ever as one of the languages you picked.
* *Call topic*, *Background context* and *Important words* are combined into a
  single string and handed to faster-whisper's `initial_prompt`. This primes the
  decoder, so names, product codes and jargon come out spelled correctly.

**Settings → Translation**

* *Term mappings* are applied to the finished text — case-insensitive, whole
  word — right before it reaches the transcript.
* *Style instructions* are appended to the prompt as `Style: ...`.

**Settings → Engine & Audio** — model size, CUDA/CPU, compute type, explicit
capture devices, how aggressively live audio is split into utterances, and
whether to stream text while someone is still speaking (turn it off to halve the
GPU work).

**If you hear nothing on System** — Windows will not let a loopback be opened on
an endpoint another program holds in WASAPI *exclusive* mode, which is common
with Voicemeeter, ASIO tools and some DAC drivers. The app then falls back to
another loopback and says so in an amber banner; pick the right endpoint under
Settings → Engine & Audio. It also warns if 25 seconds pass with no speech.

---

## Requirements

* Windows 10/11, 64-bit
* NVIDIA GPU with a driver new enough for CUDA 12 (`large-v3` uses ~3 GB VRAM,
  `medium` ~1.5 GB). Without one the app falls back to the CPU automatically —
  usable with `small`/`medium`, slow with `large-v3`.
* ~2 GB of disk for the program, plus up to 4.6 GB if you install both models
  (`medium` 1.5 GB, `large-v3` 3.1 GB)

The installer itself is ~940 MB and carries every runtime dependency; the only
thing it pulls from the network is the model weights, once. Everything after
that is offline. Uninstalling asks before deleting the weights.

---

## Diagnostics

```bash
"C:\Program Files\Local Interpreter\LocalInterpreterFetch.exe" --self-test
```

Prints whether the CUDA runtime loaded, how many GPUs CTranslate2 sees, which
models are installed and which audio devices exist, then loads the largest
installed model and runs one inference through it.

To fetch weights by hand:

```bash
"C:\Program Files\Local Interpreter\LocalInterpreterFetch.exe" --fetch-models medium,large-v3 --dest="C:\ProgramData\LocalInterpreter\models"
```

---

## Building from source

```bash
pip install PyQt6 faster-whisper soundcard numpy pillow pyinstaller
winget install JRSoftware.InnoSetup
```

The spec file copies `cublas64_12.dll`, `cudnn*64_9.dll` and friends out of
whichever pip package provides them (`torch`, or the `nvidia-*` wheels), so at
least one of those has to be present on the build machine. It prints how many it
found; if any are missing the packaged app still runs, just on the CPU.

```powershell
.\build.ps1
```

Output: `dist\LocalInterpreter\` (portable) and
`installer\LocalInterpreter-Setup-1.0.1.exe`.

---

## Notes and known limits

* **Whisper only translates into English.** Any other target language shows the
  source transcript instead. Adding e.g. English → Spanish would mean shipping a
  second, separate NMT model.
* **Two passes per utterance.** When *Show the original speech* is on, each
  utterance is decoded twice (transcribe + translate). Turn it off to halve the
  work. English speech is only decoded once either way.
* **CTranslate2 must be imported before Qt.** Loading its CUDA libraries into a
  process that already has Qt in it crashes with an access violation, so
  `local_interpreter.py` imports the engine above the PyQt6 imports on purpose.
* **COM has to be initialised by `soundcard`'s own import.** Calling
  `CoInitializeEx` first and importing soundcard afterwards makes its import
  raise, because it treats `S_FALSE` ("already initialised on this thread") as
  fatal — and under PyQt an exception escaping a slot aborts the process. See
  `com_init()`.
* **`soundcard` and unusual endpoints.** A device held in WASAPI exclusive mode
  by another program, or one whose mix format is not `WAVEFORMATEXTENSIBLE`,
  cannot be opened. The app tries every candidate device in turn and reports what
  failed; pick a different one under Settings → Engine & Audio.
* **The installer is not code-signed**, so SmartScreen shows "Windows protected
  your PC" on first run — *More info* → *Run anyway*.
