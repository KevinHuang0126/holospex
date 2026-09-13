import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { SampleTestApp } from "./SampleTestApp";
import { MannequinApp } from "./MannequinApp";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing application root");
const path = window.location.pathname.replace(/\/$/, "");
const Entry = path === "/mannequin" ? MannequinApp : path === "/samples" || (!path && import.meta.env.MODE === "samples") ? SampleTestApp : App;
createRoot(root).render(<StrictMode><Entry /></StrictMode>);
