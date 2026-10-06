"""Exercise the real inline browser state renderer without a microphone or paid API."""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


@unittest.skipUnless(shutil.which("node"), "Node is needed for the inline-browser test")
class UsageInterfaceTests(unittest.TestCase):
    def test_tap_to_toggle_controls_and_server_resets(self):
        script = HTML.read_text().split("<script>", 1)[1].split("</script>", 1)[0]
        script = script.rsplit("})();", 1)[0] + (
            "globalThis.__testLari = {armTap, disarmTap, handleJsonFrame, setConn, "
            "setState, renderUsage, markSpeaking, stop, "
            "get tapArmed(){return tapArmed}, "
            "configure(options){if ('running' in options) running = options.running; "
            "if ('playing' in options) playing = options.playing; "
            "if ('ws' in options) ws = options.ws; "
            "if ('followupUntil' in options) followupUntil = options.followupUntil;}};})();"
        )
        harness = r'''
const assert = require("assert/strict");
const vm = require("vm");
const widgets = new Map();
function widget(id){
  if (!widgets.has(id)) {
    const classes = new Set();
    widgets.set(id, {
      textContent:"", style:{}, events:{},
      classList:{add(c){classes.add(c)}, remove(c){classes.delete(c)},
        contains(c){return classes.has(c)}, toggle(c,on){on ? classes.add(c) : classes.delete(c)}},
      addEventListener(name,fn){this.events[name] = fn}, setPointerCapture(){},
    });
  }
  return widgets.get(id);
}
const context = {
  location:{pathname:"/test/", protocol:"https:", host:"example.invalid", search:""},
  document:{getElementById:widget, addEventListener(){}}, window:{},
  setInterval(){return 1}, clearInterval(){}, setTimeout(){return 1}, clearTimeout(){},
  URLSearchParams,
};
vm.createContext(context);
vm.runInContext(SOURCE, context);
const api = context.__testLari;
const sent = [];
const socket = {readyState:1, send(data){sent.push(JSON.parse(data))}, close(){}};
const down = {type:"ptt", phase:"down"}, up = {type:"ptt", phase:"up"};
const tap = widget("tapBtn"), hold = widget("pttBtn"), hint = widget("hint");
function state(value){api.handleJsonFrame(JSON.stringify({type:"state", state:value}));}
function ready(){
  api.configure({running:true, playing:false, ws:socket});
  api.setConn(true);
  api.handleJsonFrame(JSON.stringify({type:"state", state:"listening", phrase:"test phrase"}));
  sent.length = 0;
}
function armed(){
  assert.equal(api.tapArmed, true);
  assert.equal(tap.textContent, "Tap to stop");
  assert.equal(tap.classList.contains("held"), true);
}
function disarmed(){
  assert.equal(api.tapArmed, false);
  assert.equal(tap.textContent, "Tap to talk");
  assert.equal(tap.classList.contains("held"), false);
}
ready();
api.armTap();
armed();
assert.deepEqual(sent, [down]);
api.armTap();
assert.deepEqual(sent, [down]);
assert.equal(hint.textContent, "Tap again to send.");
api.configure({followupUntil:1});
api.renderUsage();
api.setState("listening");
assert.equal(hint.textContent, "Tap again to send.");
api.disarmTap(true);
disarmed();
assert.deepEqual(sent, [down, up]);
api.disarmTap(true);
assert.deepEqual(sent, [down, up]);
// The recording acknowledgement must preserve the second-tap gesture.
ready();
tap.onclick();
state("waking");
state("recording");
armed();
assert.deepEqual(sent, [down]);
tap.onclick();
disarmed();
assert.deepEqual(sent, [down, up]);
for (const value of ["transcribing", "thinking", "speaking", "listening", "error"]){
  ready();
  api.armTap();
  sent.length = 0;
  state(value);
  disarmed();
  assert.deepEqual(sent, []);
  assert.equal(hint.textContent, "Say “test phrase” to talk.");
}
for (const reset of [() => api.setConn(false), () => api.markSpeaking(), () => api.stop()]){
  ready();
  api.armTap();
  sent.length = 0;
  reset();
  disarmed();
  assert.deepEqual(sent, []);
}
for (const options of [{playing:true}, {running:false}, {ws:null},
    {ws:{readyState:0, send(){assert.fail("closed socket sent data")}}}]){
  ready();
  api.configure(options);
  api.armTap();
  disarmed();
  assert.deepEqual(sent, []);
}
for (const value of ["waking", "recording", "transcribing", "thinking", "speaking", "error", "idle"]){
  ready();
  api.setState(value);
  api.armTap();
  disarmed();
  assert.deepEqual(sent, []);
}
ready();
api.armTap();
hold.events.pointerdown({preventDefault(){assert.fail("hold armed during toggle")}});
armed();
assert.deepEqual(sent, [down]);
api.disarmTap(true);
sent.length = 0;
hold.events.pointerdown({preventDefault(){}, pointerId:1});
assert.equal(hold.textContent, "Release to send");
api.armTap();
disarmed();
assert.deepEqual(sent, [down]);
hold.events.pointerup();
assert.deepEqual(sent, [down, up]);
'''
        result = subprocess.run(
            [shutil.which("node"), "-e", harness.replace("SOURCE", json.dumps(script))],
            text=True, capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_paid_followup_and_expired_states(self):
        html = HTML.read_text()
        self.assertIn('id="usageStatus"', html)
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        script = script.rsplit("})();", 1)[0] + (
            "globalThis.__testLari = {handleJsonFrame, setConn, setState, renderUsage};})();"
        )
        harness = r'''
const vm = require("vm");
const widgets = new Map();
function widget(id){
  if (!widgets.has(id)) widgets.set(id, {
    textContent: "", className: "", style: {},
    classList: {toggle(){}, add(){}, remove(){}},
    addEventListener(){}, appendChild(){},
  });
  return widgets.get(id);
}
let now = 100000;
class Clock extends Date { static now(){ return now; } }
const context = {
  location: {pathname:"/unit-test/", protocol:"https:", host:"example.invalid", search:""},
  document: {getElementById: widget, addEventListener(){}}, window: {},
  Date: Clock, setInterval(){return 1}, clearInterval(){}, setTimeout(){return 1}, clearTimeout(){},
  URLSearchParams,
};
vm.createContext(context);
vm.runInContext(SOURCE, context);
const api = context.__testLari;
function status(){return widget("usageStatus").textContent;}
function send(data){api.handleJsonFrame(JSON.stringify(data));return status();}
api.setConn(true);
const preWake = send({type:"state",state:"listening",phrase:"test phrase"});
send({type:"state",state:"waking",followup:false,turn:1});
const pending = send({type:"state",state:"recording",followup:false,turn:1});
const paid = send({type:"stt_status",mode:"realtime",turn:1});
const fallbackAfterPaid = send({type:"stt_status",mode:"local",turn:1});
const thinking = send({type:"state",state:"thinking",turn:1});
const ready = send({type:"state",state:"listening",turn:1});
const followup = send({type:"followup",seconds:30});
now += 31000;
api.renderUsage();
const expired = status();
const expiredHint = widget("hint").textContent;
send({type:"followup",seconds:30});
send({type:"state",state:"waking",followup:true,turn:2});
const followupPending = send({type:"state",state:"recording",followup:true,turn:2});
const fallback = send({type:"stt_status",mode:"local",turn:2});
send({type:"state",state:"listening",turn:2});
send({type:"state",state:"waking",followup:true,manual:true,turn:3});
const manualPending = send({type:"state",state:"recording",followup:true,manual:true,turn:3});
const manualLocal = send({type:"stt_status",mode:"local",turn:3});
send({type:"state",state:"waking",followup:false,turn:4});
const nextWake = status();
api.setConn(false);
const disconnected = status();
console.log(JSON.stringify({preWake,pending,paid,fallbackAfterPaid,thinking,ready,followup,followupPending,fallback,expired,expiredHint,manualPending,manualLocal,nextWake,disconnected}));
'''
        harness = harness.replace("SOURCE", json.dumps(script))
        binary = shutil.which("node")
        assert binary is not None
        result = subprocess.run(
            [binary, "-e", harness], text=True,
            capture_output=True, check=True, timeout=15,
        )
        states = json.loads(result.stdout)
        self.assertIn("LOCAL", states["preWake"])
        self.assertIn("POST-WAKE", states["pending"])
        self.assertIn("NOT CONFIRMED", states["pending"])
        self.assertIn("ELEVENLABS", states["paid"])
        self.assertIn("AUDIO SENT", states["paid"])
        self.assertIn("AUDIO ALREADY SENT", states["fallbackAfterPaid"])
        self.assertIn("LOCAL", states["fallbackAfterPaid"])
        self.assertIn("NO AUDIO", states["thinking"])
        self.assertIn("LOCAL", states["ready"])
        self.assertIn("FOLLOW-UP", states["followup"])
        self.assertIn("FOLLOW-UP", states["followupPending"])
        self.assertIn("LOCAL", states["fallback"])
        self.assertIn("LOCAL", states["expired"])
        self.assertNotIn("FOLLOW-UP", states["expired"])
        self.assertIn("test phrase", states["expiredHint"])
        self.assertIn("MANUAL", states["manualPending"])
        self.assertIn("NOT CONFIRMED", states["manualPending"])
        self.assertNotIn("FOLLOW-UP", states["manualPending"])
        self.assertIn("MANUAL", states["manualLocal"])
        self.assertIn("LOCAL", states["manualLocal"])
        self.assertIn("POST-WAKE", states["nextWake"])
        self.assertNotIn("MANUAL", states["nextWake"])
        self.assertIn("DISCONNECTED", states["disconnected"])


if __name__ == "__main__":
    unittest.main()
