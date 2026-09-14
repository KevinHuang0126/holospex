import { DatasetSamplesDemo } from "./overlays/DatasetSamplesDemo";

/** Standalone React host for testing the HUD before the learning app is connected. */
export function SampleTestApp() {
  return <main className="sample-test-app">
    <header className="topbar"><a className="wordmark" href="/samples"><span className="brand-icon">H</span>Holospex</a><span className="prototype-tag">Training image overlays</span></header>
    <DatasetSamplesDemo autoLoadLocal={import.meta.env.DEV && import.meta.env.MODE === "samples"} />
    <footer><a href="/mannequin">Open the live mannequin overlay</a> · <a href="/prototype">Open the full prototype</a></footer>
  </main>;
}
