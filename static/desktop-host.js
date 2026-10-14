// Native-host adapter for reference satellites without the desktop API.
// Injected by Qt only: browsers do not auto-start microphone capture.
(() => {
  let connected = false, state = "idle", phrase = "", error = null;
  let hideRequested = false, starting = false, latestSocket = null;
  const NativeWebSocket = window.WebSocket;
  window.WebSocket = class extends NativeWebSocket {
    constructor(...args) {
      super(...args);
      latestSocket = this;
      this.addEventListener("open", () => {
        if (latestSocket !== this) return;
        connected = true;
        state = "listening";
      });
      this.addEventListener("message", event => {
        if (latestSocket !== this || typeof event.data !== "string") return;
        let message;
        try { message = JSON.parse(event.data); } catch { return; }
        if (message.type === "state") {
          state = message.state;
          if (message.phrase) phrase = message.phrase;
          error = message.error || null;
        } else if (message.type === "fatal") {
          state = "error";
          error = message.error || "Errore vocale";
        }
      });
      this.addEventListener("close", () => {
        if (latestSocket !== this) return;
        connected = false;
        // Older reference clients do not reconnect their socket. Restart their
        // capture only if listening is still armed when the retry fires.
        setTimeout(() => {
          if (window.lariDesktop?.remoteReference
              && latestSocket === this && isRunning()) {
            stop();
            start();
          }
        }, 3000);
      });
    }
  };
  const get = id => document.getElementById(id);
  const isRunning = () => !!get("stopBtn") && !get("stopBtn").classList.contains("hidden");
  function start() {
    if (isRunning() || starting) return;
    starting = true;
    get("startBtn")?.click();
    // Reference start() is asynchronous; avoid duplicate getUserMedia calls.
    const timer = setInterval(() => {
      if (isRunning() || get("orb")?.classList.contains("error")) {
        starting = false;
        clearInterval(timer);
      }
    }, 100);
  }
  function stop() {
    get("stopBtn")?.click();
    state = "idle";
    connected = false;
  }
  function install() {
    if (window.lariDesktop) {
      // The bundled client already has explicit lifecycle and reconnect logic.
      return;
    }
    window.lariDesktop = {
      remoteReference:true,
      start, stop,
      compact() {
        if (get("desktopHostStyle")) return;
        const css = document.createElement("style");
        css.id = "desktopHostStyle";
        css.textContent = `
          body{padding:14px!important;border:1px solid #1e2330;border-radius:18px;overflow:hidden}
          .stage{gap:16px!important;min-height:230px}
          .orb{width:150px!important;height:180px!important}
          .usage,.month,#stopBtn,#pttBtn{display:none!important}
          .log{max-height:130px!important;min-height:42px;flex-shrink:0}
          button{padding:10px!important;font-size:14px!important}
          .hint{max-height:36px;overflow:hidden;font-size:11px!important}
          #desktopClose{width:26px;padding:0!important;border:0;background:transparent;color:#7b8496;font-size:22px!important}
        `;
        document.head.appendChild(css);
        const close = document.createElement("button");
        close.id = "desktopClose";
        close.textContent = "×";
        close.setAttribute("aria-label", "Nascondi popup");
        close.onclick = () => { hideRequested = true; };
        document.querySelector("header")?.appendChild(close);
      },
      status() {
        const hide = hideRequested;
        hideRequested = false;
        return {running:isRunning(), starting, connected, state, phrase,
          frames:window.__frames || 0, error, hideRequested:hide};
      }
    };
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", install, {once:true});
  else install();
})();
