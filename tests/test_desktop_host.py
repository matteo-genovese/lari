"""Exercise the native adapter against an older satellite's DOM/socket API."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(shutil.which("node"), "Node is required for the host adapter test")
class DesktopHostTests(unittest.TestCase):
    def test_older_client_capture_states_pause_and_reconnect(self):
        source = (ROOT / "static/desktop-host.js").read_text()
        harness = r'''
const assert = require("assert/strict"), vm = require("vm");
const elements = new Map(), timers = [];
let starts = 0, stops = 0, interval;
function element(id) {
  if (!elements.has(id)) {
    const classes = new Set();
    elements.set(id, {classList:{contains:c=>classes.has(c), add:c=>classes.add(c), remove:c=>classes.delete(c)},
      setAttribute(){}, appendChild(){}, click(){this.onclick?.()}});
  }
  return elements.get(id);
}
element("stopBtn").classList.add("hidden");
element("startBtn").onclick = () => { starts++; element("stopBtn").classList.remove("hidden"); };
element("stopBtn").onclick = () => { stops++; element("stopBtn").classList.add("hidden"); };
class Socket {
  constructor(){this.handlers={};}
  addEventListener(name, fn){this.handlers[name]=fn;}
  fire(name,data){this.handlers[name]?.({data});}
}
const context = {
  window:{WebSocket:Socket, __frames:42},
  document:{readyState:"complete", getElementById:id=>elements.get(id),
    createElement(){return {setAttribute(){}}},
    head:{appendChild(node){elements.set(node.id,node)}}, querySelector:()=>element("header")},
  setTimeout(fn){timers.push(fn)}, setInterval(fn){interval=fn; return 1}, clearInterval(){},
};
vm.createContext(context);
vm.runInContext(SOURCE, context);
const api = context.window.lariDesktop;
assert.equal(elements.has("desktopHostStyle"), false);
api.compact();
const css = elements.get("desktopHostStyle").textContent;
const hiddenSelectors = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
  .filter(([, , declarations]) => /display\s*:\s*none\b/.test(declarations))
  .flatMap(([, selectors]) => selectors.split(",").map(s => s.trim()));
for (const selector of ["#pttBtn", "#tapBtn"]) assert.equal(hiddenSelectors.includes(selector), false, selector);
for (const selector of [".usage", ".month", "#stopBtn"]) assert.equal(hiddenSelectors.includes(selector), true, selector);
api.start(); api.start(); interval();
assert.equal(starts,1);
const ws = new context.window.WebSocket("wss://private.invalid/test/ws");
ws.fire("open");
ws.fire("message",JSON.stringify({type:"state",state:"waking",phrase:"Hey Nic"}));
assert.equal(api.status().phrase,"Hey Nic");
assert.equal(api.status().state,"waking");
assert.equal(api.status().connected,true);
assert.equal(api.status().frames,42);
ws.fire("close"); timers.pop()(); interval();
assert.equal(starts,2); assert.equal(stops,1);
api.stop(); ws.fire("close"); timers.pop()();
assert.equal(starts,2); // An intentional pause never reopens the microphone.
assert.equal(api.status().running,false);
assert.equal(api.status().state,"idle");
context.window.lariDesktop={nativeReference:true};
vm.runInContext(SOURCE, context);
assert.equal(context.window.lariDesktop.nativeReference,true);
'''
        result = subprocess.run(["node", "-e", harness.replace("SOURCE", json.dumps(source))],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
