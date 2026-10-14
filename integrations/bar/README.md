# Stato Lari nella barra

`lari-bar-status` usa Python 3 e la sola libreria standard. Interroga il bridge
con timeout di 2 secondi e stampa una riga JSON con `text`, `tooltip` e `class`.
Non richiede plugin QML né modifiche al bridge.

## Installazione

Dal repository:

```sh
mkdir -p ~/.local/bin
cp integrations/bar/lari-bar-status ~/.local/bin/lari-bar-status
chmod +x ~/.local/bin/lari-bar-status
```

Verificare che `~/.local/bin` sia nel `PATH`. La configurazione viene scelta
in questo ordine:

1. `--url <url>`: URL completo dell'endpoint di stato.
2. `--env <file>`: file dotenv da cui leggere `LARI_PORT` e `LARI_TOKEN`.
3. `LARI_ENV`: variabile d'ambiente con il percorso del file dotenv.
4. `LARI_STATUS_URL`: variabile d'ambiente con l'URL completo dell'endpoint,
   validato con le stesse regole di `--url`.

Non c'è un percorso predefinito. Usare ad esempio `--env '<repo>/.env'`,
sostituendo `<repo>` con il percorso del repository, oppure impostare
`LARI_ENV=~/.config/lari/env` nell'ambiente della barra.
Il file viene letto come dati, senza eseguire comandi shell.
Il token non viene mostrato nell'output o negli errori.

Per un bridge remoto, `--url` sostituisce la lettura dell'`.env`: deve contenere
l'URL completo dell'endpoint, nella forma `https://bridge.example/<token>/status`.
Non salvare URL autenticati nella configurazione pubblica del repository.
`--glyph 'mic'` permette di sostituire la glifa microfono Nerd Font, anche con
testo normale.

## Omarchy 4 / Quickshell

Aggiungere questo modulo all'array `bar.layout.right` di
`~/.config/omarchy/shell.json`, mantenendo gli altri moduli:

```json
{
  "id": "lari",
  "type": "command",
  "interval": 3,
  "exec": "~/.local/bin/lari-bar-status --env '<repo>/.env'",
  "onClick": "lari-desktop"
}
```

Sostituire `<repo>` prima dell'uso. `lari-desktop` deve essere installato e
disponibile nel `PATH`: il clic apre il popup del satellite.
Dopo la prima personalizzazione `shell.json` è canonico: non viene effettuato
un deep merge con i valori predefiniti. Conservare quindi le sezioni e i moduli
che si vogliono mantenere. Il formato `command` e questa regola sono documentati
nella [documentazione ufficiale Omarchy](https://github.com/omacom/omarchy/blob/quattro/docs/omarchy-shell.md).

## Waybar

Aggiungere `custom/lari` a `modules-right` nella configurazione Waybar e
inserire questo blocco tra le proprietà della configurazione:

```json
"custom/lari": {
  "exec": "~/.local/bin/lari-bar-status --env '<repo>/.env'",
  "interval": 3,
  "return-type": "json",
  "on-click": "lari-desktop"
}
```

Sostituire `<repo>` e ricaricare la barra dopo aver modificato la configurazione.

## Verifica e stati

```sh
lari-bar-status --plain --env '<repo>/.env'
```

Con `LARI_ENV` o `LARI_STATUS_URL` nell'ambiente basta `lari-bar-status --plain`.
Senza configurazione lo script restituisce la classe `offline`, il tooltip
«configurazione assente: usa --env, --url o LARI_ENV» e codice di uscita 0.
Esempio di output con il bridge disponibile:

```text
 2 [busy] Frase di wake: ciao Lari STT: vosk phone: speaking desktop: listening
```

Le classi sono `idle` (nessun satellite connesso, oppure satelliti inattivi),
`listening` (in ascolto), `busy` (waking, recording, transcribing, thinking o
speaking), `error` (errore o stato sconosciuto) e `offline` (bridge non
disponibile, timeout, configurazione o risposta non valida). Con più satelliti
connessi, la precedenza è `busy`, `error`, `listening`, `idle`; il testo mostra
glifa e numero di connessi. Il tooltip elenca ogni satellite connesso e il suo
stato su una riga, insieme alla frase di wake e al backend STT.

Gli errori operativi producono comunque output e codice di uscita 0.
Usare un intervallo di **2–5 secondi**: l'aggiornamento avviene al successivo
polling, quindi gli stati molto brevi possono non apparire.

## Rimozione

Rimuovere il modulo `lari` da `bar.layout.right`, oppure `custom/lari` dal blocco
Waybar e da `modules-right`. Ricaricare la barra, poi eliminare lo script:

```sh
rm ~/.local/bin/lari-bar-status
```
