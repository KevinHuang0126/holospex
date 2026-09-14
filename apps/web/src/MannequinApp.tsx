import { MannequinDemo } from "./camera/MannequinDemo";
import { LiveFeedDemo } from "./camera/LiveFeedDemo";
import { useState } from "react";
import { useHudControls } from "./overlays/useHudControls";
import type { HudMode } from "./overlays/hudControls";

export function MannequinApp() {
  const hud = useHudControls();
  const [input, setInput] = useState<"live" | "upload" | "url" | "reference">("live");
  return <main>
    <header className="topbar"><a className="wordmark" href="/mannequin"><span className="brand-icon">H</span>Holospex</a><span className="prototype-tag">Live video / anatomy HUD</span></header>
    <section className="intro"><p className="eyebrow">Video anatomy identification</p><h1>Your video.<br /><span>Anatomy in context.</span></h1><p>Identify anatomy from your camera, an uploaded video, or a stream link using the trained model. Use Training image AR to explore labeled training imagery on the mannequin.</p></section>
    <section className="device-setup" aria-label="Live anatomy demo">
      <div className="setup-row" role="group" aria-label="Input mode">
        <button aria-pressed={input === "live"} onClick={() => setInput("live")}>Camera identification</button>
        <button aria-pressed={input === "upload"} onClick={() => setInput("upload")}>Upload video</button>
        <button aria-pressed={input === "url"} onClick={() => setInput("url")}>Stream link</button>
        <button aria-pressed={input === "reference"} onClick={() => setInput("reference")}>Training image AR</button>
      </div>
      <div className="setup-row">
        {input === "reference" && <label>Learning mode <select value={hud.state.mode} onChange={event => hud.setMode(event.target.value as HudMode)}><option value="learn">Learn</option><option value="identify">Identify</option><option value="assess">Assess</option><option value="feedback">Feedback</option></select></label>}
        <label className="setup-check"><input type="checkbox" checked={hud.state.overlaysRequested} onChange={event => hud.showOverlays(event.target.checked)} />Show overlays</label>
      </div>
      {input !== "reference" ? <LiveFeedDemo key={input} source={input} visible={hud.state.overlaysRequested} /> : <MannequinDemo mode={hud.state.mode} visible={hud.state.overlaysRequested} />}
    </section>
    <footer><a href="/samples">Dataset image overlays</a> · <a href="/prototype">Full prototype</a></footer>
  </main>;
}
