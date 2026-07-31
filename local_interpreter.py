#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Interpreter - 100% offline live speech translation.

Single-file desktop app:
  * PyQt6 dark UI (Manual / Live modes, language bar, audio source selector)
  * `soundcard` WASAPI capture: system loopback, microphone, or both mixed
  * `faster-whisper` on CUDA (float16) with native task="translate"
  * Context priming via faster-whisper's `initial_prompt`
  * Post-translation term mapping via regex replacement

No cloud APIs, no external LLMs. The only network access this program can
ever make is downloading the Whisper weights from Hugging Face the first
time a model is used (the installer normally does that ahead of time).
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# CTranslate2 imports torch purely for its model converters, which this app
# never touches. Loading torch's DLLs after Qt is already in memory fails on
# Windows with "DLL initialization routine failed", so make the import a plain
# ImportError - ctranslate2 already handles that case and skips torch.
sys.modules.setdefault("torch", None)

APP_NAME = "Local Interpreter"
APP_VERSION = "1.0.0"
ORG_DIR = "LocalInterpreter"

SAMPLE_RATE = 16000
BLOCK_FRAMES = 1024               # ~64 ms per capture block
MAX_PROMPT_CHARS = 850            # Whisper's prompt window is ~224 tokens
MANUAL_LIMIT_S = 45 * 60          # cap on a single Manual-mode recording

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def app_dir() -> Path:
    """Directory the app is installed / running from."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def user_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    p = Path(base) / ORG_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_search_dirs() -> list[Path]:
    """Where bundled / downloaded models may live, in priority order."""
    dirs: list[Path] = []
    env = os.environ.get("LOCAL_INTERPRETER_MODELS")
    if env:
        dirs.append(Path(env))
    dirs.append(app_dir() / "models")
    program_data = os.environ.get("PROGRAMDATA")
    if program_data:
        dirs.append(Path(program_data) / ORG_DIR / "models")
    dirs.append(user_data_dir() / "models")
    out: list[Path] = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


def settings_path() -> Path:
    return user_data_dir() / "settings.json"


# ---------------------------------------------------------------------------
# CUDA runtime
#
# CTranslate2 links cuBLAS and cuDNN by name. In the packaged build those DLLs
# sit next to the executable; when running from source they usually come from
# a pip package (torch, or the nvidia-* wheels). Loading them explicitly, by
# full path, before ctranslate2 is imported is the only reliable way to get
# the right ones - a missing cuDNN makes ctranslate2 crash the process rather
# than raise.
# ---------------------------------------------------------------------------

CUDA_DLLS = [
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn_graph64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn64_9.dll",
]
ESSENTIAL_CUDA_DLLS = {"cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"}

_cuda_state: bool | None = None


def _cuda_dll_dirs() -> list[Path]:
    cands: list[Path] = []
    if is_frozen():
        cands += [app_dir(), app_dir() / "_internal"]
    else:
        import site
        roots: list[Path] = []
        try:
            roots.append(Path(site.getusersitepackages()))
        except Exception:
            pass
        try:
            roots += [Path(p) for p in site.getsitepackages()]
        except Exception:
            pass
        for r in roots:
            cands.append(r / "torch" / "lib")
            nvidia = r / "nvidia"
            if nvidia.is_dir():
                for sub in nvidia.iterdir():
                    cands += [sub / "bin", sub / "lib"]
    return [d for d in cands if d.is_dir()]


def prepare_cuda() -> bool:
    """Preload the CUDA libraries. False means: stay on the CPU."""
    global _cuda_state
    if _cuda_state is not None:
        return _cuda_state
    if sys.platform != "win32":
        _cuda_state = True
        return True

    dirs = _cuda_dll_dirs()
    for d in dirs:
        try:
            os.add_dll_directory(str(d))
        except Exception:
            pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

    loaded: set[str] = set()
    for _attempt in range(2):                     # second pass fixes ordering
        for name in CUDA_DLLS:
            if name in loaded:
                continue
            for d in dirs:
                p = d / name
                if not p.exists():
                    continue
                try:
                    ctypes.WinDLL(str(p))
                    loaded.add(name)
                    break
                except OSError:
                    pass
            else:
                try:                              # already on the system?
                    ctypes.WinDLL(name)
                    loaded.add(name)
                except OSError:
                    pass

    _cuda_state = ESSENTIAL_CUDA_DLLS <= loaded
    return _cuda_state


def preload_engine() -> None:
    """Load CTranslate2 *before* Qt.

    Importing PyQt6 first and only then bringing CTranslate2's CUDA libraries
    into the process kills it with an access violation the moment a CUDA model
    is created. Reversing the order is stable, so the engine is imported here,
    above the Qt imports, and the ordering must not be "tidied up".
    """
    prepare_cuda()
    try:
        import ctranslate2  # noqa: F401
    except Exception:
        traceback.print_exc()


preload_engine()

from PyQt6.QtCore import (      # noqa: E402  (must come after preload_engine)
    Qt,
    QSize,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (       # noqa: E402
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPixmap,
    QTextBlockFormat,
    QTextCursor,
)
from PyQt6.QtWidgets import (   # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

LANGUAGES: dict[str, str] = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
    "fo": "Faroese", "fr": "French", "gl": "Galician", "gu": "Gujarati",
    "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew", "hi": "Hindi",
    "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer",
    "kn": "Kannada", "ko": "Korean", "la": "Latin", "lb": "Luxembourgish",
    "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian",
    "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam",
    "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Burmese", "ne": "Nepali", "nl": "Dutch", "nn": "Nynorsk",
    "no": "Norwegian", "oc": "Occitan", "pa": "Punjabi", "pl": "Polish",
    "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak",
    "sl": "Slovenian", "sn": "Shona", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "tg": "Tajik", "th": "Thai",
    "tk": "Turkmen", "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar",
    "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese",
    "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese", "zh": "Chinese",
}

COMMON_LANGS = [
    "en", "es", "ru", "uk", "de", "fr", "it", "pt", "pl", "nl", "tr", "ar",
    "zh", "ja", "ko", "hi", "he", "vi", "th", "cs", "ro", "sv", "el", "fa",
]


def lang_name(code: str) -> str:
    return LANGUAGES.get(code, code)


def sorted_lang_items() -> list[tuple[str, str]]:
    """Common languages first, then everything else alphabetically."""
    rest = sorted(
        (c for c in LANGUAGES if c not in COMMON_LANGS),
        key=lambda c: LANGUAGES[c],
    )
    return [(c, LANGUAGES[c]) for c in COMMON_LANGS] + [(c, LANGUAGES[c]) for c in rest]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    # main view
    mode: str = "live"                        # "manual" | "live"
    source_lang: str = "auto"                 # "auto" or a whisper code
    target_lang: str = "en"
    audio_source: str = "system"              # "system" | "mic" | "both"

    # transcription
    model_size: str = "large-v3"
    device: str = "auto"                      # "auto" | "cuda" | "cpu"
    compute_type: str = "float16"
    expected_languages: list[str] = field(default_factory=lambda: ["en", "es"])
    only_expected_languages: bool = False
    topic: str = ""
    background: str = ""
    important_words: str = ""
    show_source_text: bool = True

    # translation
    term_mappings: list[list[str]] = field(default_factory=list)
    style_instructions: str = ""

    # devices
    system_device: str = ""                   # "" == system default
    mic_device: str = ""

    # tuning
    silence_ms: int = 700
    max_segment_s: float = 14.0
    vad_sensitivity: int = 2                  # 1 (strict) .. 3 (loose)

    @staticmethod
    def load() -> "Settings":
        s = Settings()
        p = settings_path()
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                known = {f for f in asdict(s)}
                for k, v in raw.items():
                    if k in known:
                        setattr(s, k, v)
            except Exception:
                traceback.print_exc()
        return s

    def save(self) -> None:
        try:
            settings_path().write_text(
                json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            traceback.print_exc()

    # -- derived -----------------------------------------------------------

    def initial_prompt(self) -> str | None:
        """Fold every context field into faster-whisper's `initial_prompt`."""
        parts: list[str] = []
        if self.topic.strip():
            parts.append(f"Call topic: {self.topic.strip().rstrip('.')}.")
        if self.background.strip():
            bg = " ".join(self.background.split())
            parts.append(bg if bg.endswith((".", "!", "?")) else bg + ".")
        words = [w.strip() for w in re.split(r"[,\n;]+", self.important_words) if w.strip()]
        if words:
            parts.append("Important words: " + ", ".join(words) + ".")
        if self.style_instructions.strip():
            parts.append(f"Style: {self.style_instructions.strip().rstrip('.')}.")
        prompt = " ".join(parts).strip()
        if not prompt:
            return None
        return prompt[:MAX_PROMPT_CHARS]

    def mappings(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for row in self.term_mappings:
            if len(row) >= 2 and str(row[0]).strip():
                out.append((str(row[0]).strip(), str(row[1])))
        return out


def apply_term_mappings(text: str, mappings: list[tuple[str, str]]) -> str:
    """Swap preferred translations in, longest match first, case-insensitive."""
    if not text or not mappings:
        return text
    for src, dst in sorted(mappings, key=lambda kv: -len(kv[0])):
        pattern = re.escape(src)
        if re.match(r"\w", src) and re.search(r"\w$", src):
            pattern = r"\b" + pattern + r"\b"
        try:
            text = re.sub(pattern, lambda _m, d=dst: d, text, flags=re.IGNORECASE)
        except re.error:
            continue
    return text


# ---------------------------------------------------------------------------
# Model discovery / download
# ---------------------------------------------------------------------------

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
HF_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}
MODEL_FILES = [
    "config.json", "model.bin", "tokenizer.json",
    "vocabulary.json", "vocabulary.txt", "preprocessor_config.json",
]
MODEL_DOWNLOAD_SIZE = {
    "tiny": "about 75 MB",
    "base": "about 145 MB",
    "small": "about 490 MB",
    "medium": "about 1.5 GB",
    "large-v3": "about 3.1 GB",
}
MODEL_BYTES = {
    "tiny": 76_000_000,
    "base": 146_000_000,
    "small": 489_000_000,
    "medium": 1_530_000_000,
    "large-v3": 3_090_000_000,
}


REQUIRED_MODEL_FILES = ["model.bin", "config.json", "tokenizer.json"]


def model_is_complete(d: Path) -> bool:
    """A half-finished download must not look like an installed model."""
    if not all((d / f).exists() for f in REQUIRED_MODEL_FILES):
        return False
    if not ((d / "vocabulary.json").exists() or (d / "vocabulary.txt").exists()):
        return False
    return (d / "model.bin").stat().st_size > 10_000_000


def local_model_dir(size: str) -> Path | None:
    for base in model_search_dirs():
        cand = base / f"faster-whisper-{size}"
        if model_is_complete(cand):
            return cand
    return None


def resolve_model(size: str) -> str:
    """Local directory if we have it, otherwise the HF repo id."""
    local = local_model_dir(size)
    return str(local) if local else HF_REPOS.get(size, size)


def download_model(size: str, dest_root: Path, progress=None) -> Path:
    """Fetch one model into dest_root/faster-whisper-<size>."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import tqdm as hf_tqdm

    target = dest_root / f"faster-whisper-{size}"
    target.mkdir(parents=True, exist_ok=True)

    # Files are fetched concurrently, each with its own progress bar, so the
    # per-file counters are summed into one figure for the whole model.
    tally: dict[int, list[int]] = {}

    class _Tqdm(hf_tqdm):
        """Silent tqdm that forwards byte counts to `progress`.

        `disable=True` makes tqdm skip most of its own bookkeeping, so the
        counters reported from here are kept locally.
        """

        def __init__(self, *a, **kw):
            self._seen = 0
            self._total = kw.get("total") or 0
            self._bytes = kw.get("unit") == "B"            # not the "N files" bar
            kw["disable"] = True
            super().__init__(*a, **kw)

        def update(self, n=1):
            self._seen += n or 0
            if progress and self._bytes:
                tally[id(self)] = [self._seen, self._total]
                done = sum(v[0] for v in tally.values())
                # the Hub does not always send a content length, so fall back
                # to the published size of the model
                total = max(sum(v[1] for v in tally.values()),
                            MODEL_BYTES.get(size, 0))
                progress(size, done, total)
            return super().update(n)

    snapshot_download(
        repo_id=HF_REPOS[size],
        local_dir=str(target),
        allow_patterns=MODEL_FILES,
        max_workers=4,
        tqdm_class=_Tqdm,
    )
    return target


# ---------------------------------------------------------------------------
# Audio capture (soundcard / WASAPI)
# ---------------------------------------------------------------------------


def com_init() -> None:
    """soundcard only initialises COM on the importing thread."""
    if sys.platform == "win32":
        try:
            ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        except Exception:
            pass


def list_output_devices() -> list[str]:
    import soundcard as sc
    try:
        return [s.name for s in sc.all_speakers()]
    except Exception:
        return []


def list_input_devices() -> list[str]:
    import soundcard as sc
    try:
        return [m.name for m in sc.all_microphones(include_loopback=False)]
    except Exception:
        return []


def _loopback_candidates(preferred: str) -> list:
    import soundcard as sc
    cands = []
    try:
        if preferred:
            for m in sc.all_microphones(include_loopback=True):
                if m.isloopback and m.name == preferred:
                    cands.append(m)
        default_spk = sc.default_speaker()
        try:
            cands.append(sc.get_microphone(id=str(default_spk.name), include_loopback=True))
        except Exception:
            pass
        for m in sc.all_microphones(include_loopback=True):
            if m.isloopback and all(m.name != c.name for c in cands):
                cands.append(m)
    except Exception:
        traceback.print_exc()
    return cands


def _mic_candidates(preferred: str) -> list:
    import soundcard as sc
    cands = []
    try:
        if preferred:
            for m in sc.all_microphones(include_loopback=False):
                if m.name == preferred:
                    cands.append(m)
        try:
            d = sc.default_microphone()
            if all(d.name != c.name for c in cands):
                cands.append(d)
        except Exception:
            pass
        for m in sc.all_microphones(include_loopback=False):
            if all(m.name != c.name for c in cands):
                cands.append(m)
    except Exception:
        traceback.print_exc()
    return cands


class CaptureThread(threading.Thread):
    """Pushes mono float32 blocks at SAMPLE_RATE into a queue.

    Some WASAPI endpoints refuse `soundcard`'s recorder (devices held in
    exclusive mode, or endpoints whose mix format is not
    WAVEFORMATEXTENSIBLE), so every candidate device is tried in turn.
    """

    def __init__(self, kind: str, preferred: str, out_q: queue.Queue, log):
        super().__init__(daemon=True, name=f"capture-{kind}")
        self.kind = kind                      # "system" | "mic"
        self.preferred = preferred
        self.out_q = out_q
        self.log = log
        self._stop = threading.Event()
        self.device_name: str | None = None
        self.error: str | None = None
        self.ready = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        com_init()
        cands = (_loopback_candidates if self.kind == "system" else _mic_candidates)(self.preferred)
        if not cands:
            self.error = f"No {'system audio (loopback)' if self.kind == 'system' else 'microphone'} device found."
            self.ready.set()
            return

        problems: list[str] = []
        for dev in cands:
            if self._stop.is_set():
                break
            try:
                with dev.recorder(samplerate=SAMPLE_RATE, channels=None,
                                  blocksize=BLOCK_FRAMES) as rec:
                    self.device_name = dev.name
                    self.error = None
                    self.ready.set()
                    self.log(f"{self.kind}: capturing from “{dev.name}”")
                    while not self._stop.is_set():
                        data = rec.record(numframes=BLOCK_FRAMES)
                        if data is None or len(data) == 0:
                            continue
                        mono = data.mean(axis=1) if data.ndim > 1 else data
                        try:
                            self.out_q.put_nowait(mono.astype(np.float32, copy=False))
                        except queue.Full:
                            pass
                return
            except Exception as exc:
                reason = type(exc).__name__ if not str(exc) else str(exc)
                if "0x8889000a" in reason:
                    reason = "device is in exclusive use by another app"
                elif isinstance(exc, AssertionError):
                    reason = "unsupported WASAPI mix format"
                problems.append(f"“{dev.name}”: {reason}")
                continue

        if not self._stop.is_set():
            self.error = "Could not open " + (
                "system audio. " if self.kind == "system" else "microphone. "
            ) + " / ".join(problems[:3])
        self.ready.set()


# ---------------------------------------------------------------------------
# Audio engine: mixes sources, splits speech into segments
# ---------------------------------------------------------------------------


class AudioEngine(QThread):
    segment_ready = pyqtSignal(object, str)   # np.ndarray, source label
    level = pyqtSignal(float)
    failed = pyqtSignal(str)
    info = pyqtSignal(str)

    def __init__(self, settings: Settings, live: bool, parent=None):
        super().__init__(parent)
        self.s = settings
        self.live = live
        self._stop = threading.Event()
        self._captures: list[CaptureThread] = []
        self.manual_buffer: list[np.ndarray] = []
        self.manual_recorded = 0

    def stop(self) -> None:
        self._stop.set()

    # -- helpers -----------------------------------------------------------

    def _sources(self) -> list[str]:
        if self.s.audio_source == "both":
            return ["system", "mic"]
        return [self.s.audio_source]

    def run(self) -> None:
        com_init()
        wanted = self._sources()
        queues: dict[str, queue.Queue] = {k: queue.Queue(maxsize=400) for k in wanted}
        for kind in wanted:
            pref = self.s.system_device if kind == "system" else self.s.mic_device
            t = CaptureThread(kind, pref, queues[kind], self.info.emit)
            t.start()
            self._captures.append(t)

        for t in self._captures:
            t.ready.wait(timeout=8.0)

        alive = [t for t in self._captures if t.error is None]
        if not alive:
            msgs = [t.error for t in self._captures if t.error]
            self.failed.emit("\n".join(msgs) or "Audio capture failed to start.")
            return
        if len(alive) < len(self._captures):
            for t in self._captures:
                if t.error:
                    self.info.emit(t.error)

        active = [t.kind for t in alive]
        label = "+".join(active)
        buffers: dict[str, np.ndarray] = {k: np.zeros(0, np.float32) for k in active}
        by_kind = {t.kind: t for t in alive}
        last_data = {k: time.time() for k in active}

        # --- energy segmenter state
        sens = {1: 0.010, 2: 0.006, 3: 0.0035}.get(self.s.vad_sensitivity, 0.006)
        noise = 0.002
        speech: list[np.ndarray] = []
        preroll: list[np.ndarray] = []
        preroll_max = int(0.4 * SAMPLE_RATE)
        in_speech = False
        silence_run = 0.0
        voiced_run = 0.0
        silence_limit = max(0.25, self.s.silence_ms / 1000.0)
        max_len = max(4.0, float(self.s.max_segment_s))
        last_level_emit = 0.0
        aborted = False
        limit_warned = False

        while not self._stop.is_set():
            for k in active:
                q = queues[k]
                chunks = []
                while True:
                    try:
                        chunks.append(q.get_nowait())
                    except queue.Empty:
                        break
                if chunks:
                    buffers[k] = np.concatenate([buffers[k]] + chunks)
                    last_data[k] = time.time()

            n = min(len(buffers[k]) for k in active)
            if n < BLOCK_FRAMES:
                # a source that died would stall the mix forever - drop it
                now = time.time()
                stalled = [k for k in active if now - last_data[k] > 3.0]
                for k in stalled:
                    if len(active) > 1:
                        active.remove(k)
                        label = "+".join(active)
                        self.info.emit(f"{k}: no audio, dropped from the mix")
                    elif not by_kind[k].is_alive():
                        self.failed.emit(
                            by_kind[k].error or f"{k} capture stopped unexpectedly."
                        )
                        aborted = True
                        break
                    else:
                        last_data[k] = now   # silent but alive: keep waiting
                if aborted:
                    break
                self.msleep(8)
                continue

            block = np.zeros(n, np.float32)
            for k in active:
                block += buffers[k][:n]
                buffers[k] = buffers[k][n:]
            if len(active) > 1:
                np.clip(block, -1.0, 1.0, out=block)

            dur = n / SAMPLE_RATE
            rms = float(np.sqrt(np.mean(np.square(block))) + 1e-9)

            now = time.time()
            if now - last_level_emit > 0.05:
                self.level.emit(min(1.0, rms * 12.0))
                last_level_emit = now

            if not self.live:
                if self.manual_recorded < MANUAL_LIMIT_S * SAMPLE_RATE:
                    self.manual_buffer.append(block)
                    self.manual_recorded += n
                elif not limit_warned:
                    limit_warned = True
                    self.info.emit(
                        f"Recording limit of {MANUAL_LIMIT_S // 60} minutes reached "
                        "— press Stop to translate what was captured."
                    )
                continue

            # adaptive noise floor
            if rms < noise * 1.6:
                noise = 0.995 * noise + 0.005 * rms
            threshold = max(sens, noise * 3.2)

            if rms > threshold:
                voiced_run += dur
                silence_run = 0.0
                if not in_speech and voiced_run >= 0.10:
                    in_speech = True
                    speech = list(preroll)
                    preroll = []
                if in_speech:
                    speech.append(block)
            else:
                voiced_run = 0.0
                if in_speech:
                    speech.append(block)
                    silence_run += dur
                else:
                    preroll.append(block)
                    total = sum(len(b) for b in preroll)
                    while total > preroll_max and preroll:
                        total -= len(preroll.pop(0))

            seg_len = sum(len(b) for b in speech) / SAMPLE_RATE if speech else 0.0
            if in_speech and (silence_run >= silence_limit or seg_len >= max_len):
                audio = np.concatenate(speech) if speech else np.zeros(0, np.float32)
                speech = []
                in_speech = False
                silence_run = 0.0
                if len(audio) / SAMPLE_RATE >= 0.35:
                    self.segment_ready.emit(audio, label)

        for t in self._captures:
            t.stop()

        if not aborted and not self.live and self.manual_buffer:
            audio = np.concatenate(self.manual_buffer)
            self.manual_buffer = []
            if len(audio) / SAMPLE_RATE >= 0.3:
                self.segment_ready.emit(audio, label)


# ---------------------------------------------------------------------------
# Transcription worker
# ---------------------------------------------------------------------------


@dataclass
class Job:
    audio: np.ndarray
    source: str
    live: bool


class Transcriber(QThread):
    result = pyqtSignal(dict)
    status = pyqtSignal(str)
    failed = pyqtSignal(str)
    busy = pyqtSignal(bool)

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.s = settings
        self.q: queue.Queue[Job | None] = queue.Queue()
        self.model = None
        self._loaded_key: tuple | None = None
        self._stop = threading.Event()

    def submit(self, job: Job) -> None:
        self.q.put(job)

    def stop(self) -> None:
        self._stop.set()
        self.q.put(None)

    # -- model -------------------------------------------------------------

    def ensure_model(self) -> bool:
        key = (self.s.model_size, self.s.device, self.s.compute_type)
        if self.model is not None and key == self._loaded_key:
            return True
        try:
            cuda_libs = prepare_cuda()
            from faster_whisper import WhisperModel
            import ctranslate2

            device = self.s.device
            if device in ("auto", "cuda"):
                gpus = 0
                try:
                    gpus = ctranslate2.get_cuda_device_count()
                except Exception:
                    gpus = 0
                if gpus == 0 or not cuda_libs:
                    if device == "cuda":
                        self.status.emit(
                            "No usable CUDA GPU found — running on the CPU instead."
                        )
                    device = "cpu"
                else:
                    device = "cuda"

            compute = self.s.compute_type
            if device == "cpu" and compute in ("float16", "int8_float16"):
                compute = "int8"

            path = resolve_model(self.s.model_size)
            if not os.path.isdir(path):
                self.status.emit(f"Downloading the {self.s.model_size} model…")
                path = str(download_model(self.s.model_size, user_data_dir() / "models"))

            self.status.emit(f"Loading {self.s.model_size} on {device}…")
            try:
                self.model = self._open(path, device, compute)
            except Exception as exc:
                if device != "cuda":
                    raise
                traceback.print_exc()
                self.status.emit(f"CUDA failed ({exc}) — falling back to the CPU…")
                device, compute = "cpu", "int8"
                self.model = self._open(path, device, compute)

            self._loaded_key = key
            self.status.emit(f"Ready · {self.s.model_size} · {device}/{compute}")
            return True
        except Exception as exc:
            self.model = None
            self._loaded_key = None
            self.failed.emit(f"Could not load the speech model:\n{exc}")
            return False

    def _open(self, path: str, device: str, compute: str):
        from faster_whisper import WhisperModel
        return WhisperModel(
            path,
            device=device,
            compute_type=compute,
            cpu_threads=max(4, (os.cpu_count() or 8) // 2),
            num_workers=1,
        )

    # -- inference ---------------------------------------------------------

    def _pick_language(self, audio: np.ndarray) -> str | None:
        """Honour the source dropdown and the 'only these languages' lock."""
        if self.s.source_lang != "auto":
            return self.s.source_lang
        allowed = [c for c in self.s.expected_languages if c in LANGUAGES]
        if not allowed:
            return None
        if len(allowed) == 1:
            return allowed[0]
        if not self.s.only_expected_languages:
            return None
        try:
            _, _, probs = self.model.detect_language(audio=audio, vad_filter=False)
            allowed_set = set(allowed)
            best = max(
                ((c, p) for c, p in probs if c in allowed_set),
                key=lambda cp: cp[1],
                default=None,
            )
            return best[0] if best else allowed[0]
        except Exception:
            return allowed[0]

    def _run_whisper(self, audio, task, language, live) -> tuple[str, str | None]:
        segments, info = self.model.transcribe(
            audio,
            task=task,
            language=language,
            initial_prompt=self.s.initial_prompt(),
            beam_size=1 if live else 5,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250, speech_pad_ms=200),
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return re.sub(r"\s+", " ", text), getattr(info, "language", None)

    def run(self) -> None:
        while not self._stop.is_set():
            job = self.q.get()
            if job is None or self._stop.is_set():
                break
            # drop stale live audio if we fell behind
            if job.live:
                while self.q.qsize() > 3:
                    nxt = self.q.get()
                    if nxt is None:
                        return
                    job = nxt
            if not self.ensure_model():
                continue
            self.busy.emit(True)
            try:
                audio = np.ascontiguousarray(job.audio, dtype=np.float32)
                peak = float(np.abs(audio).max()) if len(audio) else 0.0
                if peak > 0:
                    audio = audio * min(3.0, 0.85 / peak) if peak < 0.3 else audio

                language = self._pick_language(audio)
                want_english = self.s.target_lang == "en"

                source_text = ""
                detected = language
                if self.s.show_source_text or not want_english:
                    source_text, detected = self._run_whisper(
                        audio, "transcribe", language, job.live
                    )

                if want_english:
                    if detected == "en" and source_text:
                        translation = source_text
                    else:
                        translation, det2 = self._run_whisper(
                            audio, "translate", language or detected, job.live
                        )
                        detected = detected or det2
                else:
                    # Whisper translates into English only; for any other
                    # target the transcript is what a local model can give.
                    translation = ""

                translation = apply_term_mappings(translation, self.s.mappings())
                source_text = apply_term_mappings(source_text, self.s.mappings())

                if translation or source_text:
                    self.result.emit({
                        "source_text": source_text,
                        "translation": translation,
                        "language": detected or "?",
                        "input": job.source,
                        "duration": len(audio) / SAMPLE_RATE,
                        "time": time.strftime("%H:%M:%S"),
                    })
            except Exception as exc:
                traceback.print_exc()
                self.failed.emit(f"Transcription failed: {exc}")
            finally:
                self.busy.emit(False)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class SegmentedControl(QWidget):
    """A row of mutually-exclusive pill buttons."""

    changed = pyqtSignal(str)

    def __init__(self, options: list[tuple[str, str]], value: str, parent=None):
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(3)
        self.setObjectName("segmented")
        for key, text in options:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setObjectName("segbtn")
            b.clicked.connect(lambda _c=False, k=key: self.set_value(k, emit=True))
            lay.addWidget(b)
            self._buttons[key] = b
        self.set_value(value, emit=False)

    def value(self) -> str:
        for k, b in self._buttons.items():
            if b.isChecked():
                return k
        return ""

    def set_value(self, key: str, emit: bool = True) -> None:
        for k, b in self._buttons.items():
            b.setChecked(k == key)
        if emit:
            self.changed.emit(key)


class LevelMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(6)
        self.setMinimumWidth(120)
        self._level = 0.0
        self._peak = 0.0
        t = QTimer(self)
        t.timeout.connect(self._decay)
        t.start(60)

    def set_level(self, v: float) -> None:
        self._level = max(0.0, min(1.0, v))
        self._peak = max(self._peak, self._level)

    def _decay(self) -> None:
        self._level *= 0.75
        self._peak *= 0.94
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#20262e"))
        p.drawRoundedRect(r, 3, 3)
        w = int(r.width() * self._level)
        if w > 0:
            color = QColor("#22c55e") if self._level < 0.85 else QColor("#f59e0b")
            p.setBrush(color)
            rr = r.adjusted(0, 0, w - r.width(), 0)
            p.drawRoundedRect(rr, 3, 3)


class TermMappingTable(QTableWidget):
    def __init__(self, rows: list[list[str]], parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Word or phrase", "Preferred translation", ""])
        self.verticalHeader().setVisible(False)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(2, 34)
        self.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        for row in rows:
            self.add_row(row[0] if row else "", row[1] if len(row) > 1 else "")

    def add_row(self, a: str = "", b: str = "") -> None:
        r = self.rowCount()
        self.insertRow(r)
        self.setItem(r, 0, QTableWidgetItem(a))
        self.setItem(r, 1, QTableWidgetItem(b))
        btn = QPushButton("✕")
        btn.setObjectName("rowdel")
        btn.setToolTip("Remove this mapping")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda: self._remove(btn))
        self.setCellWidget(r, 2, btn)

    def _remove(self, btn) -> None:
        for r in range(self.rowCount()):
            if self.cellWidget(r, 2) is btn:
                self.removeRow(r)
                return

    def values(self) -> list[list[str]]:
        out = []
        for r in range(self.rowCount()):
            a = self.item(r, 0).text().strip() if self.item(r, 0) else ""
            b = self.item(r, 1).text().strip() if self.item(r, 1) else ""
            if a:
                out.append([a, b])
        return out


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------


class SettingsDialog(QDialog):
    def __init__(self, s: Settings, parent=None):
        super().__init__(parent)
        self.s = s
        self.setWindowTitle("Settings")
        self.setMinimumSize(QSize(640, 620))

        tabs = QTabWidget()
        tabs.addTab(self._transcription_tab(), "Transcription")
        tabs.addTab(self._translation_tab(), "Translation")
        tabs.addTab(self._engine_tab(), "Engine && Audio")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(14)
        lay.addWidget(tabs)
        lay.addWidget(buttons)

    # -- tabs --------------------------------------------------------------

    def _section(self, title: str, subtitle: str = "") -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(6)
        lbl = QLabel(title)
        lbl.setObjectName("sectionTitle")
        box.addWidget(lbl)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("hint")
            sub.setWordWrap(True)
            box.addWidget(sub)
        return box

    def _transcription_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 12, 4, 4)
        lay.setSpacing(16)

        block = self._section(
            "Expected languages",
            "Restricts Whisper's language detection. With exactly one language "
            "selected the model is locked to it (language=\"xx\"), which is both "
            "faster and more accurate.",
        )
        self.lang_list = QListWidget()
        self.lang_list.setFixedHeight(150)
        for code, name in sorted_lang_items():
            item = QListWidgetItem(f"{name}  ({code})")
            item.setData(Qt.ItemDataRole.UserRole, code)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if code in self.s.expected_languages
                else Qt.CheckState.Unchecked
            )
            self.lang_list.addItem(item)
        block.addWidget(self.lang_list)
        self.only_expected = QCheckBox("Only these languages")
        self.only_expected.setChecked(self.s.only_expected_languages)
        block.addWidget(self.only_expected)
        lay.addLayout(block)

        ctx = self._section(
            "Context",
            "Everything below is merged into one string and passed to "
            "faster-whisper's initial_prompt, priming the decoder for your "
            "jargon, names and topic.",
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, 150)

        grid.addWidget(QLabel("Call topic"), 0, 0)
        self.topic = QLineEdit(self.s.topic)
        self.topic.setPlaceholderText("e.g. quarterly logistics review with a Spanish supplier")
        grid.addWidget(self.topic, 0, 1)

        grid.addWidget(QLabel("Background context"), 1, 0, Qt.AlignmentFlag.AlignTop)
        self.background = QTextEdit(self.s.background)
        self.background.setPlaceholderText("Who is on the call, what was agreed last time, product names…")
        self.background.setFixedHeight(80)
        grid.addWidget(self.background, 1, 1)

        grid.addWidget(QLabel("Important words"), 2, 0, Qt.AlignmentFlag.AlignTop)
        self.keywords = QTextEdit(self.s.important_words)
        self.keywords.setPlaceholderText("Comma separated: Kiril, Rustam, Kaspi, ERP, SKU-4471")
        self.keywords.setFixedHeight(60)
        grid.addWidget(self.keywords, 2, 1)
        ctx.addLayout(grid)
        lay.addLayout(ctx)

        self.show_source = QCheckBox("Show the original speech above each translation")
        self.show_source.setChecked(self.s.show_source_text)
        lay.addWidget(self.show_source)
        lay.addStretch(1)
        return w

    def _translation_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 12, 4, 4)
        lay.setSpacing(16)

        block = self._section(
            "Term mappings",
            "Applied to the finished text with a case-insensitive, whole-word "
            "replacement before it reaches the transcript.",
        )
        self.terms = TermMappingTable(self.s.term_mappings)
        block.addWidget(self.terms)
        add = QPushButton("+  Add")
        add.setObjectName("ghost")
        add.setCursor(Qt.CursorShape.PointingHandCursor)
        add.clicked.connect(lambda: self.terms.add_row())
        row = QHBoxLayout()
        row.addWidget(add)
        row.addStretch(1)
        block.addLayout(row)
        lay.addLayout(block)

        style = self._section(
            "Style instructions",
            "Optional. Appended to the initial_prompt as “Style: …”, which nudges "
            "register and punctuation.",
        )
        self.style = QLineEdit(self.s.style_instructions)
        self.style.setPlaceholderText("e.g. Formal tone, full sentences, keep numbers as digits")
        style.addWidget(self.style)
        lay.addLayout(style)
        lay.addStretch(1)
        return w

    def _engine_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 12, 4, 4)
        lay.setSpacing(16)

        block = self._section("Model", "")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, 150)

        grid.addWidget(QLabel("Model size"), 0, 0)
        self.model_size = QComboBox()
        for size in MODEL_SIZES:
            here = local_model_dir(size) is not None
            self.model_size.addItem(f"{size}{'' if here else '  (will download)'}", size)
        idx = self.model_size.findData(self.s.model_size)
        self.model_size.setCurrentIndex(max(0, idx))
        grid.addWidget(self.model_size, 0, 1)

        grid.addWidget(QLabel("Device"), 1, 0)
        self.device = QComboBox()
        for key, text in [("auto", "Auto (CUDA if available)"), ("cuda", "CUDA"), ("cpu", "CPU")]:
            self.device.addItem(text, key)
        self.device.setCurrentIndex(max(0, self.device.findData(self.s.device)))
        grid.addWidget(self.device, 1, 1)

        grid.addWidget(QLabel("Compute type"), 2, 0)
        self.compute = QComboBox()
        for key in ["float16", "int8_float16", "int8", "float32"]:
            self.compute.addItem(key, key)
        self.compute.setCurrentIndex(max(0, self.compute.findData(self.s.compute_type)))
        grid.addWidget(self.compute, 2, 1)
        block.addLayout(grid)
        lay.addLayout(block)

        dev = self._section("Audio devices", "Leave on Default unless capture fails.")
        grid2 = QGridLayout()
        grid2.setHorizontalSpacing(12)
        grid2.setVerticalSpacing(8)
        grid2.setColumnMinimumWidth(0, 150)
        grid2.addWidget(QLabel("System audio"), 0, 0)
        self.sys_dev = QComboBox()
        self.sys_dev.addItem("Default output (loopback)", "")
        for name in list_output_devices():
            self.sys_dev.addItem(name, name)
        self.sys_dev.setCurrentIndex(max(0, self.sys_dev.findData(self.s.system_device)))
        grid2.addWidget(self.sys_dev, 0, 1)

        grid2.addWidget(QLabel("Microphone"), 1, 0)
        self.mic_dev = QComboBox()
        self.mic_dev.addItem("Default microphone", "")
        for name in list_input_devices():
            self.mic_dev.addItem(name, name)
        self.mic_dev.setCurrentIndex(max(0, self.mic_dev.findData(self.s.mic_device)))
        grid2.addWidget(self.mic_dev, 1, 1)
        dev.addLayout(grid2)
        lay.addLayout(dev)

        seg = self._section("Segmentation", "How live audio is cut into utterances.")
        grid3 = QGridLayout()
        grid3.setHorizontalSpacing(12)
        grid3.setVerticalSpacing(8)
        grid3.setColumnMinimumWidth(0, 150)
        grid3.addWidget(QLabel("Silence before flush"), 0, 0)
        self.silence = QComboBox()
        for ms, text in [(450, "450 ms (snappy)"), (700, "700 ms (balanced)"), (1000, "1.0 s (patient)")]:
            self.silence.addItem(text, ms)
        self.silence.setCurrentIndex(max(0, self.silence.findData(self.s.silence_ms)))
        grid3.addWidget(self.silence, 0, 1)

        grid3.addWidget(QLabel("Max utterance"), 1, 0)
        self.maxseg = QComboBox()
        for sec, text in [(8.0, "8 s"), (14.0, "14 s"), (22.0, "22 s")]:
            self.maxseg.addItem(text, sec)
        self.maxseg.setCurrentIndex(max(0, self.maxseg.findData(self.s.max_segment_s)))
        grid3.addWidget(self.maxseg, 1, 1)

        grid3.addWidget(QLabel("Mic sensitivity"), 2, 0)
        self.sens = QComboBox()
        for key, text in [(1, "Low (noisy room)"), (2, "Normal"), (3, "High (quiet room)")]:
            self.sens.addItem(text, key)
        self.sens.setCurrentIndex(max(0, self.sens.findData(self.s.vad_sensitivity)))
        grid3.addWidget(self.sens, 2, 1)
        seg.addLayout(grid3)
        lay.addLayout(seg)
        lay.addStretch(1)
        return w

    # -- result ------------------------------------------------------------

    def apply_to(self, s: Settings) -> None:
        codes = []
        for i in range(self.lang_list.count()):
            it = self.lang_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                codes.append(it.data(Qt.ItemDataRole.UserRole))
        s.expected_languages = codes
        s.only_expected_languages = self.only_expected.isChecked()
        s.topic = self.topic.text()
        s.background = self.background.toPlainText()
        s.important_words = self.keywords.toPlainText()
        s.show_source_text = self.show_source.isChecked()
        s.term_mappings = self.terms.values()
        s.style_instructions = self.style.text()
        s.model_size = self.model_size.currentData()
        s.device = self.device.currentData()
        s.compute_type = self.compute.currentData()
        s.system_device = self.sys_dev.currentData()
        s.mic_device = self.mic_dev.currentData()
        s.silence_ms = int(self.silence.currentData())
        s.max_segment_s = float(self.maxseg.currentData())
        s.vad_sensitivity = int(self.sens.currentData())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

STYLESHEET = """
QWidget { background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
QDialog, QTabWidget::pane { background: #0d1117; border: none; }
QLabel#title { font-size: 15px; font-weight: 600; letter-spacing: 0.3px; }
QLabel#sectionTitle { font-size: 13px; font-weight: 600; color: #e6edf3; }
QLabel#hint { color: #7d8590; font-size: 11px; }
QLabel#status { color: #7d8590; font-size: 11px; }
QLabel#warn { color: #d29922; font-size: 11px; }

#card { background: #161b22; border: 1px solid #21262d; border-radius: 12px; }

#segmented { background: #161b22; border: 1px solid #21262d; border-radius: 10px; }
QPushButton#segbtn {
    background: transparent; border: none; border-radius: 7px;
    padding: 6px 16px; color: #8b949e; font-weight: 500;
}
QPushButton#segbtn:hover { color: #e6edf3; }
QPushButton#segbtn:checked { background: #2f81f7; color: #ffffff; }

QPushButton {
    background: #21262d; border: 1px solid #30363d; border-radius: 8px;
    padding: 7px 14px; color: #e6edf3;
}
QPushButton:hover { background: #30363d; }
QPushButton:pressed { background: #282e36; }
QPushButton#ghost { background: transparent; border: 1px dashed #30363d; color: #8b949e; }
QPushButton#ghost:hover { color: #e6edf3; border-color: #8b949e; }
QPushButton#icon { background: transparent; border: none; font-size: 17px; padding: 4px 8px; }
QPushButton#icon:hover { background: #21262d; border-radius: 8px; }
QPushButton#rowdel { background: transparent; border: none; color: #8b949e;
    font-size: 14px; padding: 0px; }
QPushButton#rowdel:hover { color: #f85149; }

QPushButton#primary {
    background: #238636; border: 1px solid #2ea043; border-radius: 26px;
    font-size: 16px; font-weight: 600; color: #ffffff; padding: 0px;
}
QPushButton#primary:hover { background: #2ea043; }
QPushButton#primary[recording="true"] { background: #da3633; border-color: #f85149; }
QPushButton#primary[recording="true"]:hover { background: #f85149; }
QPushButton#primary:disabled { background: #21262d; border-color: #30363d; color: #6e7681; }

QComboBox, QLineEdit, QTextEdit, QListWidget, QTableWidget {
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
    padding: 6px 9px; selection-background-color: #2f81f7;
}
QComboBox:focus, QLineEdit:focus, QTextEdit:focus { border-color: #2f81f7; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow {
    image: none; width: 0px; height: 0px; margin-right: 9px;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid #8b949e;
}
QComboBox::down-arrow:on { border-top-color: #e6edf3; }
QComboBox QAbstractItemView {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    selection-background-color: #2f81f7; padding: 4px; outline: none;
}
QComboBox#lang { font-size: 14px; font-weight: 600; padding: 9px 12px; background: #161b22; }

QTextEdit#transcript { background: #0d1117; border: none; padding: 4px 10px; font-size: 14px; }

QTableWidget { gridline-color: #21262d; }
QHeaderView::section { background: #161b22; color: #8b949e; border: none;
    border-bottom: 1px solid #21262d; padding: 6px; }
QTableWidget::item { padding: 4px; }

QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid #30363d; background: #0d1117; }
QCheckBox::indicator:checked { background: #2f81f7; border-color: #2f81f7; }

QTabBar::tab { background: transparent; color: #8b949e; padding: 8px 16px;
    border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #e6edf3; border-bottom: 2px solid #2f81f7; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #30363d; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #484f58; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

QDialogButtonBox QPushButton { min-width: 88px; padding: 7px 16px; }
QProgressBar { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    text-align: center; height: 16px; }
QProgressBar::chunk { background: #2f81f7; border-radius: 5px; }
"""


def make_icon() -> QIcon:
    """App icon: a blue rounded square with a speech-wave glyph."""
    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#2f81f7"))
    p.drawRoundedRect(2, 2, 60, 60, 14, 14)
    p.setBrush(QColor("#ffffff"))
    for i, h in enumerate((16, 30, 42, 30, 16)):
        x = 12 + i * 9
        p.drawRoundedRect(x, 32 - h // 2, 5, h, 2, 2)
    p.end()
    return QIcon(pm)


class MainWindow(QWidget):
    def __init__(self, s: Settings):
        super().__init__()
        self.s = s
        self.engine: AudioEngine | None = None
        self.running = False
        self.entries: list[dict] = []

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(make_icon())
        self.resize(880, 760)
        self.setMinimumSize(560, 560)

        self.transcriber = Transcriber(self.s)
        self.transcriber.result.connect(self.on_result)
        self.transcriber.status.connect(self.set_status)
        self.transcriber.failed.connect(self.on_error)
        self.transcriber.busy.connect(self.on_busy)
        self.transcriber.start()

        self._build_ui()
        self._sync_language_note()
        QTimer.singleShot(150, self._startup_check)

    # -- UI ----------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 18)
        root.setSpacing(14)

        # header ------------------------------------------------------------
        header = QHBoxLayout()
        header.setSpacing(10)
        dot = QLabel("●")
        dot.setStyleSheet("color:#2f81f7;font-size:15px;")
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        header.addWidget(dot)
        header.addWidget(title)
        header.addStretch(1)
        self.mode_ctrl = SegmentedControl(
            [("manual", "Manual"), ("live", "Live")], self.s.mode
        )
        self.mode_ctrl.changed.connect(self.on_mode_changed)
        header.addWidget(self.mode_ctrl)
        gear = QPushButton("⚙")
        gear.setObjectName("icon")
        gear.setToolTip("Settings")
        gear.setCursor(Qt.CursorShape.PointingHandCursor)
        gear.clicked.connect(self.open_settings)
        header.addWidget(gear)
        root.addLayout(header)

        # language bar -------------------------------------------------------
        langcard = QFrame()
        langcard.setObjectName("card")
        lb = QHBoxLayout(langcard)
        lb.setContentsMargins(12, 10, 12, 10)
        lb.setSpacing(10)

        self.src_combo = QComboBox()
        self.src_combo.setObjectName("lang")
        self.src_combo.addItem("Auto-detect", "auto")
        for code, name in sorted_lang_items():
            self.src_combo.addItem(name, code)
        self.src_combo.setCurrentIndex(max(0, self.src_combo.findData(self.s.source_lang)))
        self.src_combo.currentIndexChanged.connect(self.on_lang_changed)

        swap = QPushButton("⇄")
        swap.setObjectName("icon")
        swap.setToolTip("Swap languages")
        swap.setCursor(Qt.CursorShape.PointingHandCursor)
        swap.clicked.connect(self.swap_languages)

        self.tgt_combo = QComboBox()
        self.tgt_combo.setObjectName("lang")
        for code, name in sorted_lang_items():
            self.tgt_combo.addItem(name, code)
        self.tgt_combo.setCurrentIndex(max(0, self.tgt_combo.findData(self.s.target_lang)))
        self.tgt_combo.currentIndexChanged.connect(self.on_lang_changed)

        lb.addWidget(self.src_combo, 1)
        lb.addWidget(swap)
        lb.addWidget(self.tgt_combo, 1)
        root.addWidget(langcard)

        self.lang_note = QLabel("")
        self.lang_note.setObjectName("warn")
        self.lang_note.setWordWrap(True)
        root.addWidget(self.lang_note)

        # transcript ---------------------------------------------------------
        tcard = QFrame()
        tcard.setObjectName("card")
        tl = QVBoxLayout(tcard)
        tl.setContentsMargins(8, 8, 8, 8)
        tl.setSpacing(6)

        bar = QHBoxLayout()
        bar.setContentsMargins(6, 2, 6, 0)
        self.busy_label = QLabel("transcribing…")
        self.busy_label.setObjectName("status")
        self.busy_label.hide()
        bar.addWidget(self.busy_label)
        bar.addStretch(1)
        for text, slot, tip in (
            ("Copy", self.copy_transcript, "Copy the whole transcript"),
            ("Save…", self.save_transcript, "Save the transcript to a text file"),
            ("Clear", self.clear_transcript, "Clear the transcript"),
        ):
            b = QPushButton(text)
            b.setObjectName("ghost")
            b.setToolTip(tip)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(slot)
            bar.addWidget(b)
        tl.addLayout(bar)

        self.transcript = QTextEdit()
        self.transcript.setObjectName("transcript")
        self.transcript.setReadOnly(True)
        self.transcript.setCursorWidth(0)
        self.transcript.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.transcript.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        tl.addWidget(self.transcript)
        root.addWidget(tcard, 1)
        self._render_placeholder()

        # status + control ---------------------------------------------------
        statusrow = QHBoxLayout()
        statusrow.setSpacing(10)
        self.meter = LevelMeter()
        statusrow.addWidget(self.meter)
        self.status = QLabel("Idle")
        self.status.setObjectName("status")
        statusrow.addWidget(self.status)
        statusrow.addStretch(1)
        root.addLayout(statusrow)

        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.setFixedHeight(52)
        self.start_btn.setMinimumWidth(220)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self.toggle)
        brow = QHBoxLayout()
        brow.addStretch(1)
        brow.addWidget(self.start_btn, 2)
        brow.addStretch(1)
        root.addLayout(brow)

        # audio source -------------------------------------------------------
        srow = QHBoxLayout()
        srow.addStretch(1)
        self.source_ctrl = SegmentedControl(
            [("system", "🔊  System"), ("mic", "🎙  Mic"), ("both", "⇋  Both")],
            self.s.audio_source,
        )
        self.source_ctrl.changed.connect(self.on_source_changed)
        srow.addWidget(self.source_ctrl)
        srow.addStretch(1)
        root.addLayout(srow)

    # -- transcript rendering ---------------------------------------------

    def _render_placeholder(self) -> None:
        self.transcript.setHtml(
            "<div style='color:#484f58;padding:26px 8px;font-size:13px;'>"
            "Pick an audio source, then press <b>Start</b>.<br><br>"
            "<b>Live</b> streams each utterance as it is spoken.<br>"
            "<b>Manual</b> records until you press Stop, then translates the "
            "whole take in one pass."
            "</div>"
        )

    def append_entry(self, e: dict) -> None:
        first = not self.entries
        if first:
            self.transcript.clear()
        self.entries.append(e)
        lang = lang_name(e.get("language", "?"))
        meta = f"{e['time']} · {lang} · {e['input']} · {e['duration']:.1f}s"
        html = [
            "<div style='margin:0 0 4px 0;'>",
            f"<span style='color:#6e7681;font-size:11px;'>{esc(meta)}</span>",
        ]
        if e.get("source_text") and (self.s.show_source_text or not e.get("translation")):
            html.append(
                f"<div style='color:#8b949e;font-size:13px;margin-top:2px;'>{esc(e['source_text'])}</div>"
            )
        if e.get("translation"):
            html.append(
                f"<div style='color:#e6edf3;font-size:15px;margin-top:2px;'>{esc(e['translation'])}</div>"
            )
        html.append("</div>")

        cur = self.transcript.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        if not first:
            gap = QTextBlockFormat()         # separate entries, no inherited margins
            gap.setTopMargin(16)
            cur.insertBlock(gap)
        cur.insertHtml("".join(html))
        self.transcript.setTextCursor(cur)
        sb = self.transcript.verticalScrollBar()
        sb.setValue(sb.maximum())

    # -- events ------------------------------------------------------------

    def _startup_check(self) -> None:
        if local_model_dir(self.s.model_size) is None:
            self.set_status(
                f"{self.s.model_size} weights are not installed yet — they will "
                "download on first start."
            )

    def _sync_language_note(self) -> None:
        if self.s.target_lang != "en":
            self.lang_note.setText(
                f"Whisper can only translate into English locally, so with "
                f"{lang_name(self.s.target_lang)} as the target the panel shows the "
                f"transcript in the spoken language. Set the target to English for translation."
            )
            self.lang_note.show()
        else:
            self.lang_note.hide()

    def on_lang_changed(self) -> None:
        self.s.source_lang = self.src_combo.currentData()
        self.s.target_lang = self.tgt_combo.currentData()
        self.s.save()
        self._sync_language_note()

    def swap_languages(self) -> None:
        src, tgt = self.s.source_lang, self.s.target_lang
        if src == "auto":
            src = "en"
        self.src_combo.setCurrentIndex(max(0, self.src_combo.findData(tgt)))
        self.tgt_combo.setCurrentIndex(max(0, self.tgt_combo.findData(src)))

    def on_mode_changed(self, mode: str) -> None:
        if self.running:
            self.stop()
        self.s.mode = mode
        self.s.save()
        self.start_btn.setText("Start")

    def on_source_changed(self, source: str) -> None:
        if self.running:
            self.stop()
        self.s.audio_source = source
        self.s.save()

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.s, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            was_running = self.running
            if was_running:
                self.stop()
            dlg.apply_to(self.s)
            self.s.save()
            self.set_status("Settings saved")
            if was_running:
                self.start()

    def toggle(self) -> None:
        self.stop() if self.running else self.start()

    def ensure_model_present(self) -> bool:
        """Offer to fetch the weights before the first run."""
        size = self.s.model_size
        if local_model_dir(size) is not None:
            return True
        answer = QMessageBox.question(
            self,
            APP_NAME,
            f"The {size} speech model is not installed yet "
            f"({MODEL_DOWNLOAD_SIZE.get(size, 'about 1 GB')}).\n\n"
            "Download it now? This is the one and only time the app uses the "
            "network — everything after that runs entirely on this machine.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        dlg = DownloadDialog([size], user_data_dir() / "models", self)
        return dlg.exec() == QDialog.DialogCode.Accepted

    def start(self) -> None:
        if self.running:
            return
        if not self.ensure_model_present():
            self.set_status("Idle — model not installed")
            return
        self.running = True
        self.start_btn.setText("Stop")
        self.start_btn.setProperty("recording", "true")
        self.start_btn.style().polish(self.start_btn)
        live = self.s.mode == "live"
        self.set_status("Listening…" if live else "Recording — press Stop when done")

        self.engine = AudioEngine(self.s, live=live)
        self.engine.segment_ready.connect(self.on_segment)
        self.engine.level.connect(self.meter.set_level)
        self.engine.failed.connect(self.on_audio_failed)
        self.engine.info.connect(self.set_status)
        self.engine.start()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.start_btn.setText("Start")
        self.start_btn.setProperty("recording", "false")
        self.start_btn.style().polish(self.start_btn)
        if self.engine:
            self.engine.stop()
            self.engine.wait(4000)
            self.engine = None
        self.set_status("Idle")

    def on_segment(self, audio: np.ndarray, source: str) -> None:
        self.transcriber.submit(Job(audio=audio, source=source, live=self.s.mode == "live"))

    def on_busy(self, busy: bool) -> None:
        self.busy_label.setVisible(busy)

    def on_result(self, e: dict) -> None:
        self.append_entry(e)

    def on_audio_failed(self, msg: str) -> None:
        self.stop()
        QMessageBox.warning(self, "Audio capture", msg)

    def on_error(self, msg: str) -> None:
        self.set_status("Error")
        QMessageBox.critical(self, APP_NAME, msg)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    # -- transcript actions -------------------------------------------------

    def transcript_text(self) -> str:
        lines = []
        for e in self.entries:
            lines.append(f"[{e['time']}] ({lang_name(e.get('language', '?'))}, {e['input']})")
            if e.get("source_text"):
                lines.append(f"  {e['source_text']}")
            if e.get("translation"):
                lines.append(f"> {e['translation']}")
            lines.append("")
        return "\n".join(lines)

    def copy_transcript(self) -> None:
        QApplication.clipboard().setText(self.transcript_text())
        self.set_status("Transcript copied")

    def save_transcript(self) -> None:
        if not self.entries:
            return
        default = str(Path.home() / f"interpreter-{time.strftime('%Y%m%d-%H%M')}.txt")
        path, _ = QFileDialog.getSaveFileName(self, "Save transcript", default, "Text (*.txt)")
        if path:
            Path(path).write_text(self.transcript_text(), encoding="utf-8")
            self.set_status(f"Saved to {path}")

    def clear_transcript(self) -> None:
        self.entries.clear()
        self._render_placeholder()

    def closeEvent(self, event) -> None:
        self.stop()
        self.transcriber.stop()
        self.transcriber.wait(3000)
        self.s.save()
        event.accept()


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# First-run model download dialog
# ---------------------------------------------------------------------------


class DownloadDialog(QDialog):
    progress = pyqtSignal(str, int, int)
    finished_ok = pyqtSignal(bool, str)

    def __init__(self, sizes: list[str], dest: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downloading speech models")
        self.setFixedWidth(460)
        self.sizes = sizes
        self.dest = dest

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 20)
        lay.setSpacing(12)
        head = QLabel("Getting the offline speech models")
        head.setObjectName("sectionTitle")
        lay.addWidget(head)
        self.label = QLabel("Starting…")
        self.label.setObjectName("hint")
        self.label.setWordWrap(True)
        lay.addWidget(self.label)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        lay.addWidget(self.bar)

        self.progress.connect(self._on_progress)
        self.finished_ok.connect(self._on_done)
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        try:
            for size in self.sizes:
                if local_model_dir(size) is not None:
                    continue
                self.progress.emit(size, 0, 0)
                download_model(size, self.dest, progress=self.progress.emit)
            self.finished_ok.emit(True, "")
        except Exception as exc:
            self.finished_ok.emit(False, str(exc))

    def _on_progress(self, size: str, n: int, total: int) -> None:
        total = max(total, n)                 # the fallback size is approximate
        if total:
            self.label.setText(
                f"{size} — {n / 1e6:,.0f} MB of {total / 1e6:,.0f} MB"
            )
            self.bar.setRange(0, 100)
            self.bar.setValue(int(100 * n / total))
        else:
            self.label.setText(f"{size} — {n / 1e6:,.0f} MB")
            self.bar.setRange(0, 0)

    def _on_done(self, ok: bool, err: str) -> None:
        if ok:
            self.accept()
        else:
            QMessageBox.warning(self, "Download failed", err)
            self.reject()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def fetch_models_cli(sizes: list[str], dest: Path) -> int:
    """Used by the installer: download weights with plain console output."""
    dest.mkdir(parents=True, exist_ok=True)
    state = {"last": 0.0}

    def report(size: str, n: int, total: int) -> None:
        now = time.time()
        total = max(total, n)                 # the fallback size is approximate
        if total and now - state["last"] > 1.0:
            state["last"] = now
            print(f"       {n / 1e6:7,.0f} MB of {total / 1e6:7,.0f} MB "
                  f"({100 * n / total:5.1f}%)", flush=True)

    for size in sizes:
        if local_model_dir(size) is not None:
            print(f"[skip] {size} already installed", flush=True)
            continue
        print(f"[get ] {size} ({HF_REPOS[size]})", flush=True)
        try:
            download_model(size, dest, progress=report)
            print(f"[ok  ] {size}", flush=True)
        except Exception as exc:
            print(f"[fail] {size}: {exc}", flush=True)
            return 1
    return 0


def self_test_cli() -> int:
    """Console diagnostics: engine, GPU, models, audio devices."""
    print(f"{APP_NAME} {APP_VERSION} — self test")
    print(f"  frozen        : {is_frozen()}")
    print(f"  program dir   : {app_dir()}")
    print(f"  settings      : {settings_path()}")

    cuda = prepare_cuda()
    print(f"  CUDA runtime  : {'loaded' if cuda else 'NOT FOUND (CPU only)'}")
    try:
        import ctranslate2
        print(f"  ctranslate2   : {ctranslate2.__version__}, "
              f"{ctranslate2.get_cuda_device_count()} CUDA device(s)")
    except Exception as exc:
        print(f"  ctranslate2   : FAILED - {exc}")
        return 1

    print("  models        :")
    installed = []
    for size in MODEL_SIZES:
        d = local_model_dir(size)
        print(f"      {size:9s} {d if d else '- not installed'}")
        if d:
            installed.append(size)

    print("  audio         :")
    try:
        for name in list_output_devices():
            print(f"      out  {name}")
        for name in list_input_devices():
            print(f"      in   {name}")
    except Exception as exc:
        print(f"      device enumeration failed: {exc}")

    if not installed:
        print("\nNo models installed - run with --fetch-models medium,large-v3")
        return 1

    size = installed[-1]
    s = Settings.load()
    s.model_size = size
    tr = Transcriber(s)
    tr.status.connect(lambda m: print(f"  {m}"))
    tr.failed.connect(lambda m: print(f"  ERROR: {m}"))
    print(f"\n  loading {size}…")
    t0 = time.time()
    if not tr.ensure_model():
        return 1
    print(f"  loaded in {time.time() - t0:.1f}s")

    rng = np.random.default_rng(0)
    tone = (0.05 * np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE)
            + 0.01 * rng.standard_normal(SAMPLE_RATE * 3)).astype(np.float32)
    t0 = time.time()
    text, lang = tr._run_whisper(tone, "translate", None, live=True)
    print(f"  inference ran in {time.time() - t0:.2f}s (lang={lang}, text={text!r})")
    print("\nSelf test passed.")
    return 0


def main() -> int:
    if "--self-test" in sys.argv or "--fetch-models" in sys.argv:
        try:                                   # the installer console is cp1252
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if "--self-test" in sys.argv:
        return self_test_cli()

    if "--fetch-models" in sys.argv:
        i = sys.argv.index("--fetch-models")
        rest = [a for a in sys.argv[i + 1:] if not a.startswith("-")]
        sizes = rest[0].split(",") if rest else ["medium", "large-v3"]
        dest_arg = [a for a in sys.argv if a.startswith("--dest=")]
        dest = Path(dest_arg[0][7:]) if dest_arg else (user_data_dir() / "models")
        return fetch_models_cli([s for s in sizes if s in HF_REPOS], dest)

    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "LocalInterpreter.App"
            )
        except Exception:
            pass

    com_init()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyleSheet(STYLESHEET)
    app.setWindowIcon(make_icon())
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    s = Settings.load()
    win = MainWindow(s)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
