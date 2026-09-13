import { MannequinDemo } from "./camera/MannequinDemo";
import { useHudControls } from "./overlays/useHudControls";
import type { HudMode } from "./overlays/hudControls";

export function MannequinApp() {
  const hud = useHudControls();
  return <main>
    <header className="topbar"><a className="wordmark" href="/mannequin"><span className="brand-icon">H</span>Holospex</a><span className="prototype-tag">Live camera overlay</span></header>
    <section className="intro"><p className="eyebrow">Labeled surgical imagery / AR</p><h1>The surgical image.<br /><span>In your camera view.</span></h1><p>Place a detailed surgical image and its anatomy labels over your live camera. Add the mannequin, fit the sample to its anatomical region, then anchor the combined scene to a table marker.</p></section>
    <section className="device-setup" aria-label="Mannequin camera demo">
      <div className="setup-row">
        <label>Learning mode <select value={hud.state.mode} onChange={event => hud.setMode(event.target.value as HudMode)}><option value="learn">Learn</option><option value="identify">Identify</option><option value="assess">Assess</option><option value="feedback">Feedback</option></select></label>
        <label className="setup-check"><input type="checkbox" checked={hud.state.overlaysRequested} onChange={event => hud.showOverlays(event.target.checked)} />Show overlays</label>
      </div>
      <MannequinDemo mode={hud.state.mode} visible={hud.state.overlaysRequested} />
    </section>
    <footer><a href="/samples">Dataset image overlays</a> · <a href="/prototype">Full prototype</a></footer>
  </main>;
}
