"""Cancellable voice synthesis, sentence boundaries and ordered delivery."""
from __future__ import annotations
from . import protocol
import asyncio
import logging
import re
from .config import Settings

log = logging.getLogger("lari")
TTS_ATTEMPTS = 4
TTS_RETRY_BACKOFF_S = (0.3, 0.8, 2.0)
MAX_CONSECUTIVE_TTS_FAILURES = 2


class TTSUnavailable(RuntimeError):
    """Voice output failed; the public message contains no provider details."""


async def synthesize_once(text: str, voice: str) -> bytes:
    """Use edge-tts in the caller's task so cancellation closes its stream."""
    import edge_tts
    buf = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk.get("type") == "audio":
            buf.extend(chunk["data"])
    return bytes(buf)


async def tts(text: str, settings: Settings) -> bytes:
    """Retry transient synthesis failures with cancellable backoff."""
    for attempt in range(TTS_ATTEMPTS):
        if attempt:
            await asyncio.sleep(TTS_RETRY_BACKOFF_S[attempt - 1])
        try:
            audio = await synthesize_once(text, settings.tts_voice)
            if not audio:
                raise TTSUnavailable("tts non disponibile")
            return audio
        except Exception as exc:
            log.warning("TTS attempt %d/%d failed (%s)",
                        attempt + 1, TTS_ATTEMPTS, type(exc).__name__)
    raise TTSUnavailable("tts non disponibile") from None


class SpeakableSentenceBuffer:
    """Collect SSE deltas without ever cutting a word in half."""

    _END_RE = re.compile(r"[.!?]+(?:[\"'’”»\)\]]+)?(?=\s|$)|[;:]+(?=\s|$)|\n+")

    def __init__(self, max_chars: int,
                 total_limit: int):
        self.max_chars = max(1, max_chars)
        self.total_limit = max(1, total_limit)
        self.buffer = ""
        self.total_chars = 0

    def _take(self, end: int) -> str:
        sentence = self.buffer[:end].strip()
        self.buffer = self.buffer[end:].lstrip()
        return sentence

    def _split_long_prefix(self) -> str | None:
        if len(self.buffer) <= self.max_chars:
            return None
        # Only split at whitespace.  If one token is unusually long, retain it
        # until punctuation/final flush rather than producing partial-word audio.
        cut = self.buffer.rfind(" ", 0, self.max_chars + 1)
        if cut <= 0:
            return None
        return self._take(cut)

    def feed(self, delta: str) -> list[str]:
        if not isinstance(delta, str) or not delta:
            return []
        self.total_chars += len(delta)
        if self.total_chars > self.total_limit:
            raise RuntimeError("risposta Hermes troppo lunga")
        self.buffer += delta
        sentences: list[str] = []
        while self.buffer:
            match = self._END_RE.search(self.buffer)
            if match:
                sentence = self._take(match.end())
                if sentence:
                    sentences.append(sentence)
                continue
            sentence = self._split_long_prefix()
            if sentence:
                sentences.append(sentence)
                continue
            break
        return sentences

    def flush(self) -> list[str]:
        sentences: list[str] = []
        while self.buffer:
            sentence = self._split_long_prefix()
            if sentence:
                sentences.append(sentence)
                continue
            sentence = self.buffer.strip()
            self.buffer = ""
            if sentence:
                sentences.append(sentence)
        return sentences



class Reading:
    """One response's bounded voice queue, owned by exactly one Session.

    A single worker synthesizes and sends every segment in order. The first
    failed segment marks the turn's error without skipping later segments.
    A success resets the failure count; two consecutive failures drain the
    queue, letting Hermes finish without resubmission.
    Playback completion remains the satellite's ACK, not queue completion.
    """

    def __init__(self, turn, send_json, send_audio, started, audio_sent, valid, *, settings: Settings):
        self._settings = settings
        self.turn = turn
        self.send_json = send_json
        self.send_audio = send_audio
        self.on_started = started
        self.on_audio_sent = audio_sent
        self.valid = valid
        self.buffer = SpeakableSentenceBuffer(settings.stream_sentence_max_chars, settings.stream_text_max_chars)
        self.queue = asyncio.Queue(maxsize=max(1, settings.stream_tts_queue_max))
        self.started = False
        self.error: TTSUnavailable | None = None
        self.sequence = 0
        self.worker: asyncio.Task | None = None

    async def _synthesize(self, text):
        try:
            return await tts(text, self._settings)
        except Exception:
            raise TTSUnavailable("tts non disponibile") from None

    async def _deliver(self, audio, segmented=True):
        if not audio or not self.valid():
            return
        if not self.started:
            await self.on_started()
            if not self.valid():
                return
            if segmented:
                await self.send_json(protocol.audio_start(turn=self.turn))
            self.started = True
        if not self.valid():
            return
        if segmented:
            await self.send_json(protocol.audio_chunk(turn=self.turn, seq=self.sequence))
        else:
            await self.send_json(protocol.audio(fmt="mp3", bytes=len(audio), turn=self.turn))
        if not self.valid():
            return
        if self.send_audio is None:
            raise TTSUnavailable("websocket audio non disponibile")
        await self.send_audio(audio)
        self.on_audio_sent()
        self.sequence += 1

    def start(self):
        self.worker = asyncio.create_task(self._run(), name=f"tts-{self.turn}")

    async def _run(self):
        consecutive_failures = 0
        segment = 0
        while True:
            sentence = await self.queue.get()
            try:
                if sentence is None:
                    return
                segment += 1
                # The turn error persists; only consecutive failures stop synthesis.
                if consecutive_failures < MAX_CONSECUTIVE_TTS_FAILURES and self.valid():
                    try:
                        await self._deliver(await self._synthesize(sentence))
                        consecutive_failures = 0
                    except Exception:
                        if self.error is None:
                            self.error = TTSUnavailable("tts non disponibile")
                        consecutive_failures += 1
                        log.warning("TTS segment %d failed (%d consecutive failures)",
                                    segment, consecutive_failures)
            finally:
                self.queue.task_done()

    async def feed(self, delta):
        if self.valid():
            for sentence in self.buffer.feed(delta):
                await self.queue.put(sentence)

    async def finish(self, flush=True):
        if flush:
            for sentence in self.buffer.flush():
                await self.queue.put(sentence)
        await self.queue.join()
        await self.queue.put(None)
        await self.worker
        if self.started and self.valid():
            await self.send_json(protocol.audio_end(turn=self.turn))

    async def speak(self, text):
        if len(text) > self._settings.stream_text_max_chars:
            raise TTSUnavailable("risposta troppo lunga")
        try:
            await self._deliver(await self._synthesize(text), segmented=False)
        except Exception:
            raise TTSUnavailable("tts non disponibile") from None

    async def cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        self.buffer.buffer = ""
