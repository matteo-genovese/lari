"""Single source of truth for the configurable wake phrase.

Everything wake-related derives from one configured phrase (plus optional
aliases): the start-anchored command regex, the Vosk constrained grammar and
its loose detection, the junk-prefix cleanup for free-form transcripts, the
ASR bias keyterms and the style prompt.  Changing the phrase rebuilds all of
them coherently; nothing is hardcoded to any particular phrase here beyond
observed ASR mis-renderings of the default, kept as calibration data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

# Common spoken interjections accepted before the wake phrase itself.
INTERJECTIONS = ("hey", "ehi", "eh", "e", "hi", "ok", "ciao", "yo")
_SEP = r"[\s,.!?:;\-]*"

# Observed free-transcript mis-renderings of known wake cores, used only by the
# post-wake junk cleanup (never as wake triggers).  Extend via real-voice
# calibration, not guesses.
OBSERVED_JUNK: dict[str, tuple[str, ...]] = {
    "nic": ("inc", "heinrich", "di"),
}

# Per-letter ASR confusion classes (accents included).  c/k interchange and
# i/y interchange are the confusions observed on real phone audio; other
# letters match strictly.
_LETTER_CLASS: dict[str, str] = {
    "a": "aà", "e": "eéè", "i": "iíy", "o": "oóò", "u": "uù",
    "c": "ck", "k": "ck",
}
# A word's final vowel may render as one of its confusion set (lari -> lare).
_FINAL_VOWEL_SWAP: dict[str, str] = {
    "a": "aà", "e": "eéèi", "i": "iíye", "o": "oóò", "u": "uù",
    "y": "iíye",
}
# A word ending in a consonant may pick up at most one spurious vowel (nic ->
# nica/nici/nick are real renderings; nicola is not, \b rejects it).
_TRAILING_VOWELS = "aeio"

_INTERJ_RE = "(?:%s)" % "|".join(INTERJECTIONS)


def normalize(text: str) -> str:
    """Lowercase, strip edge punctuation, collapse spaces."""
    cleaned = re.sub(r"^[\s,.!?:;\"'’”»(\[]+|[\s,.!?:;\"'’”»)\]]+$", "", text.strip())
    return re.sub(r"\s+", " ", cleaned.lower())


def default_phrase(language: str) -> str:
    """Italian installations greet with \"ehi lari\", others with \"hey lari\"."""
    return "ehi lari" if (language or "").strip().lower().startswith("it") else "hey lari"


def _fold(letter: str) -> str:
    for base, group in _LETTER_CLASS.items():
        if letter in group:
            return base
    return letter


def _token_pattern(word: str) -> str:
    """Tolerant regex fragment for one wake word (accents and known confusions)."""
    parts: list[str] = []
    letters = [c for c in word if c.isalpha()]
    for index, letter in enumerate(letters):
        base = _fold(letter)
        group = _LETTER_CLASS.get(base)
        last = index == len(letters) - 1
        if base in "aeiouy":
            if last:
                parts.append("[%s]" % _FINAL_VOWEL_SWAP.get(base, group or base))
            else:
                parts.append("[%s]" % (group or base))
        else:
            if group and len(group) > 1:
                parts.append("(?:[%s][%s]?)" % (group, group))
            else:
                parts.append(re.escape(letter))
    pattern = "".join(parts)
    if letters and _fold(letters[-1]) not in "aeiouy":
        pattern += "[%s]?" % _TRAILING_VOWELS
    return pattern


def _phrase_pattern(phrase: str) -> str:
    return r"\s+".join(_token_pattern(word) for word in normalize(phrase).split())


def _display(phrase: str) -> str:
    return " ".join(word.capitalize() for word in normalize(phrase).split())


def _ordered_unique(items: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for item in items:
        if item and item not in seen:
            seen[item] = None
    return tuple(seen)


def _twin(phrase: str) -> str:
    """The hey/ehi twin of a phrase, so both renderings bias and detect alike."""
    words = normalize(phrase).split()
    if words and words[0] in ("hey", "ehi"):
        swapped = "ehi" if words[0] == "hey" else "hey"
        return " ".join([swapped, *words[1:]])
    return " ".join(["ehi", *words])


@dataclass(frozen=True)
class WakeConfig:
    phrase: str
    display: str
    twin_display: str
    core: str
    core_display: str
    aliases: tuple[str, ...]
    grammar: tuple[str, ...]
    keyterms: tuple[str, ...]
    batch_keyterms: tuple[str, ...]
    realtime_keyterms: tuple[str, ...]
    style_prompt: str
    prompt: str
    command_re: re.Pattern
    loose_re: re.Pattern
    cleanup_re: re.Pattern
    junk_words: tuple[str, ...] = field(default=())

    def command(self, text: str) -> str | None:
        """Return the request after the wake; None when not addressed to us."""
        match = self.command_re.search(text)
        if not match:
            return None
        return text[match.end():].lstrip(" ,.!?;:-\u2019'\u201c\u201d")

    def strip_junk(self, text: str) -> str:
        """Remove up to two leading junk renderings of the wake from a transcript."""
        return self.cleanup_re.sub("", text, count=1)


def build_wake_config(phrase: str, aliases: Iterable[str] = (),
                      vocab: Iterable[str] = (),
                      command_override: str | None = None) -> WakeConfig:
    phrase = normalize(phrase)
    words = phrase.split()
    if not words or not any(c.isalpha() for c in phrase):
        raise ValueError("wake phrase must contain letters")
    aliases = _ordered_unique(normalize(a) for a in aliases)
    vocab = _ordered_unique(t.strip() for t in vocab if t.strip())
    core = words[-1]
    twin = _twin(phrase)
    display = _display(phrase)
    twin_display = _display(twin)
    core_display = core.capitalize()

    # Literals for the constrained Vosk grammar and its detection: proven
    # shape is [twin, phrase, core, aliases..., "[unk]"].
    literals = _ordered_unique([twin, phrase, core, *aliases])
    grammar = (*literals, "[unk]")

    command_re = re.compile(
        r"^\s*(?:%s%s)?(?:%s|%s|%s)\b" % (
            _INTERJ_RE, _SEP,
            "|".join(_phrase_pattern(a) for a in aliases) or _phrase_pattern(phrase),
            _phrase_pattern(phrase),
            _token_pattern(core),
        ),
        re.IGNORECASE,
    )
    if command_override:
        command_re = re.compile(command_override, re.IGNORECASE)
    loose_re = re.compile(
        r"\b(?:%s)\b" % "|".join(re.escape(lit) for lit in literals),
        re.IGNORECASE,
    )

    # Junk cleanup: short interjections, observed mis-renderings of this core,
    # and every word of a configured alias.  Deliberately narrower than the
    # wake interjection set so real request words survive ("E che ne dici?").
    junk_words = _ordered_unique([
        "ehi", "hey",
        *OBSERVED_JUNK.get(core, ()),
        *(word for alias in aliases for word in alias.split()),
    ])
    cleanup_re = re.compile(
        r"^(?:(?:%s)\b%s){1,2}" % ("|".join(re.escape(w) for w in junk_words), _SEP),
        re.IGNORECASE,
    )

    keyterms = _ordered_unique([display, twin_display, core_display])
    batch_keyterms = _ordered_unique([*keyterms, *vocab])
    realtime_keyterms = tuple(t for t in batch_keyterms if len(t) <= 20)
    if vocab:
        style_prompt = "Italiano. Frase di sveglia: %s / %s. Nomi propri: %s." % (
            display, twin_display, ", ".join([core_display, *vocab]),
        )
    else:
        style_prompt = "Italiano. Frase di sveglia: %s / %s. Nome proprio: %s." % (
            display, twin_display, core_display,
        )
    prompt = "%s. %s." % (display, twin_display)

    return WakeConfig(
        phrase=phrase,
        display=display,
        twin_display=twin_display,
        core=core,
        core_display=core_display,
        aliases=aliases,
        grammar=grammar,
        keyterms=keyterms,
        batch_keyterms=batch_keyterms,
        realtime_keyterms=realtime_keyterms,
        style_prompt=style_prompt,
        prompt=prompt,
        command_re=command_re,
        loose_re=loose_re,
        cleanup_re=cleanup_re,
        junk_words=junk_words,
    )


def from_env(environ: Mapping[str, str] | None = None) -> WakeConfig:
    """Build the config from BUDDY_* settings (legacy names are canonical)."""
    import os
    env = os.environ if environ is None else environ
    phrase = env.get("BUDDY_WAKE_PHRASE", "").strip() or default_phrase(
        env.get("BUDDY_STT_LANG", "")
    )
    aliases = (a for a in env.get("BUDDY_WAKE_ALIASES", "").split(","))
    vocab = (t for t in env.get("BUDDY_STT_KEYTERMS", "").split(","))
    return build_wake_config(
        phrase,
        aliases=aliases,
        vocab=vocab,
        command_override=env.get("BUDDY_WAKE_RE", "").strip() or None,
    )
