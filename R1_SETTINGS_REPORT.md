# R1 — Iniezione di Settings

Worktree `/home/matt/lari-cleanup`, branch `refactor/di-cleanup`, HEAD `4dc942b`.
Nessun commit, push o dipendenza aggiunta. Nessuna modifica al servizio live.

`config.py` mantiene tutti i nomi env e i default. `server.py` chiama
`get_settings()` una sola volta e passa lo stesso oggetto alle sessioni.
`Session` richiede `settings`, conserva un riferimento `_settings` e lo passa
alla pipeline; nessun componente runtime legge più la config del processo.

| Modulo | Istantanee / costanti rimosse | Iniezione attuale |
| --- | --- | --- |
| `session.py` | `_SETTINGS`, `WAKE_PROVIDER`, `AMBIENT_PAUSE_S`, `ECHO_MUTE_S`, `FOLLOWUP_S`, `PLAYBACK_ACK_TIMEOUT_S`, `MIN_SPEECH_S`; ledger globale | `Session(..., settings=...)`, letture da `self._settings`; ledger passato dal server o costruito con quelle Settings |
| `audio.py` | `_SETTINGS`, `VAD_MIN_RMS`, `VAD_NOISE_MULT`, `SILENCE_END_S`, `MAX_UTTERANCE_S`, `MIN_SPEECH_S`, `IDLE_ABORT_S` | Il mixin legge direttamente `self._settings` della sessione |
| `tts.py` | `_SETTINGS`, `TTS_VOICE`, `STREAM_TTS_QUEUE_MAX`, `STREAM_TEXT_MAX_CHARS`, `STREAM_SENTENCE_MAX_CHARS` | `Reading(..., settings=...)`, `tts(text, settings)`; limiti espliciti a `SpeakableSentenceBuffer` |
| `wake/detector.py` | `_SETTINGS`, `WAKE_CONFIG`, `VOSK_MODEL_DIR`, `WAKE_PROVIDER`, `WAKE_PHRASE`, `WAKE_SENSITIVITY`, `CONFIRM_FRAMES` | `WakeConfig` alle funzioni di gate/risoluzione; `Settings` a `make_engine`; percorso a `get_vosk` |
| `wake/confirm.py` | `_SETTINGS`, `WAKE_CONFIG` | `confirm_candidate(pcm, cfg, model=..., language=...)` |
| `wake/runtime.py` | `_SETTINGS`, `WAKE_PROVIDER`, `ECHO_MUTE_S`, `WAKE_CONFIRM`, `AMBIENT_PAUSE_S` | `self._settings`, passata ai gate e all'engine |
| `stt/local.py` | `_SETTINGS`, `WAKE_CONFIG`, `STT_MODEL`, `STT_LANG` | `transcribe(pcm, cfg, model, language)`; `get_stt(model)` |
| `stt/vosk.py` | `_SETTINGS`, `WAKE_CONFIG` (quest'ultima era inutilizzata) | `transcribe_vosk(pcm, cfg)` |
| `stt/dispatch.py` | `_SETTINGS`, `WAKE_CONFIG` (inutilizzata), `STT_BACKEND` | `Settings` obbligatoria alle funzioni di dispatch, anche al fallback locale |
| `stt/providers.py` | `_WAKE`, `KEYTERMS`, `STYLE_PROMPT`; getter nella funzione | `transcribe(..., settings=...)`, bias da `settings.wake_config` |
| `stt/realtime.py` | `_WAKE`, `REALTIME_KEYTERMS`, `REALTIME_URL`, budget creato all'import; getter in budget/connect; letterale inutilizzato `REALTIME_DAILY_SECONDS` | URL da `realtime_url(cfg)`; Settings a `connect`, `DailyAudioBudget` e `realtime_daily_budget` |
| `usage.py` | Getter nel costruttore, non all'import | `UsageLedger(..., settings=...)`; server condivide il ledger tra le sessioni |

## Letture e proprietà preservate

- Nessun `AudioConfig` derivato: il VAD legge direttamente le Settings immutabili.
  Soglie, endpointing, trim e segmentazione TTS mantengono gli stessi algoritmi.
- Per mantenere `vosk_wake(pcm, cfg)`, `WakeConfig` include `model_dir`.
  `Settings.__post_init__` lo deriva da `vosk_model_dir`, anche quando
  `dataclasses.replace` cambia quel percorso. Un WakeConfig costruito da solo
  per operazioni testuali non richiede un modello; per Vosk serve il percorso.
- Whisper mantiene la cache condivisa per nome modello. Vosk passa dal singleton
  alla cache condivisa per percorso: stesso percorso = stesso modello, percorsi
  diversi = modelli diversi. Ogni chiamata crea il proprio recognizer.
- Settings e WakeConfig sono frozen e condivisibili. Turno, playback, waiter,
  realtime, follow-up, stato UI, buffer audio e coda TTS restano per-sessione.
- Il budget realtime crea handle sul ledger configurato; i lock su file già
  esistenti coordinano le riserve tra handle/sessioni/processi. Il server passa
  un unico UsageLedger alle sessioni, conservando la sincronizzazione originaria.
- L'import dell'engine opzionale aggiunge `hermes_root` al percorso solo quando
  `make_engine(settings)` lo richiede, non all'import del detector.
- Restano invariati i payload, le lingue fisse dei provider cloud, i timeout
  realtime fissi e i tempi tecnici del protocollo (inclusi 700 ms di mute client).
  Le funzioni runtime richiedono ora i parametri espliciti; script e test sono
  stati adattati. I benchmark caricano Settings una volta e le passano ai backend.
- Aggiornata la fixture AST del wake per le modifiche intenzionali alle firme,
  alla cache e al campo model_dir; algoritmi lessicali e grammatica invariati.

## Guardia AST

`tests/test_architecture.py` visita le assegnazioni a livello di modulo,
anche nei blocchi condizionali, e analizza il loro valore tramite AST.
Copre assegnazioni normali, annotate, attributi, getter rinominati tramite import
ed espressioni composte. Non cerca stringhe di codice o nomi di variabili.
Le sole località approvate, in lista esplicita e commentata, sono
`lari/config.py` (proprietario) e `lari/server.py` (composition root).

Testo del fallimento:

```text
{file}:{line}: module settings snapshot; inject Settings via a constructor/function parameter from lari.server or Session
```

## Test e prove negative

`tests/test_settings_injection.py` aggiunge 9 test:

1. Payload followup e finestra/compensazione eco con valori diversi dal processo.
2. Rifiuto osservabile di un turno breve con min_speech_s personalizzato, prima STT.
3. Soglia VAD e idle_abort_s personalizzati nel registratore reale.
4. Mute dell'eco personalizzato nel wake worker: nessun turno schedulato.
5. Settings condivisa/immutabile, code TTS e stato di connessione indipendenti.
6. Voce TTS, backpressure e limiti di segmentazione/testo personalizzati.
7. Backend/prompt STT, URL realtime, ledger e cap personalizzati e cap condiviso.
8. Cache pesanti condivise per modello/percorso.
9. Propagazione di vosk_model_dir dopo dataclasses.replace.

La guardia aggiunge 2 test: scansione del runtime e verifica delle forme AST.
`test_p4_session.py` mantiene i test di indipendenza con iniezione esplicita.

Mutazioni eseguite e ripristinate con `finally`:

- Reintrodotta `FOLLOWUP_S = get_settings().followup_s` in session.py e usata nel
  payload: `test_custom_followup_and_echo_are_observable` fallisce con
  `{'type': 'followup', 'seconds': 30.0} != {'type': 'followup', 'seconds': 43.0}`.
- Aggiunta `_S = get_settings()` in audio.py: la scansione fallisce con
  `lari/audio.py:241: module settings snapshot; inject Settings via a constructor/function parameter from lari.server or Session`.

## Verifica

Interprete: `/home/matt/lari/.venv/bin/python` (solo esecuzione dell'interprete
esistente, senza modifiche al servizio live).

| Comando / verifica | Esito |
| --- | --- |
| `python -m py_compile lari/*.py lari/*/*.py` | PASS |
| `python -m unittest discover -q` | Blocco nel test HTTP degli asset; esecuzione limitata a 15 s, uscita 124 |
| Suite completa con risveglio periodico del selector | PASS: 191 test, 5.000 s |
| Guardia separata e test_p4_session separato, senza workaround | PASS: rispettivamente 7 e 12 test; i 9 nuovi test sono verdi nella suite completa |
| `grep` get_settings escludendo config.py e server.py | Output vuoto, uscita 1 attesa |
| `grep` senza esclusioni | Solo definizione in config.py e singola chiamata in server.py |
| `python scripts/bench_wake_gate.py` | PASS: intestazione, candidates=0, vetoed_seconds=0.0s |
| `python scripts/bench_stt_abc.py --help` | PASS |
| `bench_stt_abc.py --local` con WAV temporaneo e backend simulati | PASS: entrambi i trascrittori locali eseguiti |
| `git diff --check` | PASS |
| `graphify update .` | PASS, aggiornamento AST locale |

Il blocco della suite diretta è riproducibile anche sul test branding della
copia originale di HEAD in `/tmp/r1-baseline` (timeout 12 s, uscita 124).
Lo stack mostra il loop in `selectors.select` e il worker AnyIO in `queue.get`.
Il workaround di verifica tiene il loop attivo ogni 10 ms, senza cambiare codice,
dipendenze o test del progetto. Il launcher usato è `/tmp/r1_suite_poll.py`:

```python
import asyncio
import sys
import unittest
sys.path.insert(0, '/home/matt/lari-cleanup')
class PollingPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        loop = super().new_event_loop()
        def tick():
            loop.call_later(.01, tick)
        loop.call_soon(tick)
        return loop
asyncio.set_event_loop_policy(PollingPolicy())
suite = unittest.defaultTestLoader.discover(
    '/home/matt/lari-cleanup/tests', top_level_dir='/home/matt/lari-cleanup')
result = unittest.TextTestRunner(verbosity=1).run(suite)
sys.exit(not result.wasSuccessful())
```

Il PASS con questo launcher è distinto dal comando diretto, che resta da
rieseguire in un ambiente dove il risveglio del selector dai thread funzioni.
