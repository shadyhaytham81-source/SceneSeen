"use strict";
// SceneSeen Phase 1 UI — plain JS, no build step.

const $ = (s) => document.querySelector(s);
const state = { vid: null, result: null, file: null, editing: false, defaults: null, previewEnd: null, params: null };
const COLORS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"].map(
  (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim());
const colorFor = (i) => COLORS[i % COLORS.length];

// ------------------------------------------------------------------ utils
function fmt(t) {
  t = Math.max(0, t);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
function fmtDur(t) {
  if (t < 60) return `${Math.round(t)} sec`;
  const m = Math.floor(t / 60), s = Math.round(t % 60);
  return s ? `${m}m ${s}s` : `${m}m`;
}
const pad2 = (n) => String(n).padStart(2, "0");
async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}
function show(view) {
  for (const v of ["home", "processing", "results"]) $("#" + v).hidden = v !== view;
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------ developer mode
const devOn = () => document.body.classList.contains("devmode");
function setDev(on) {
  document.body.classList.toggle("devmode", on);
  $("#devToggle").checked = on;
  try { localStorage.setItem("sceneseen.dev", on ? "1" : "0"); } catch (_) {}
  if (on && state.vid && !$("#results").hidden) loadResult(state.vid);
}
$("#devToggle").addEventListener("change", (e) => setDev(e.target.checked));

// ------------------------------------------------------------------ home
const drop = $("#drop"), fileInput = $("#fileInput");
$("#chooseBtn").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
drop.addEventListener("click", (e) => { if (!state.file && e.target === drop) fileInput.click(); });
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
fileInput.addEventListener("change", () => fileInput.files[0] && pickFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && pickFile(e.dataTransfer.files[0]));

function pickFile(f) {
  state.file = f;
  $("#fileName").textContent = f.name;
  $("#fileSize").textContent = ` · ${(f.size / 1e6).toFixed(1)} MB`;
  $("#filePicked").hidden = false;
  $("#homeError").hidden = true;
}
$("#analyzeBtn").addEventListener("click", () => state.file && runAnalysis(state.file));

async function loadRecent() {
  try {
    const list = await api("/api/videos");
    $("#recent").hidden = !list.length;
    $("#recentList").replaceChildren(...list.map((v) => el("li", {},
      el("a", { href: `#/v/${v.video_id}` }, el("span", {}, v.name),
        el("span", { class: "muted" }, `${v.scene_count} scenes · ${fmt(v.duration)}`)))));
  } catch (_) {}
}

// ------------------------------------------------------------------ processing
function setStep(stage, done = false) {
  const order = ["upload", "reading", "shots", "scenes", "results"];
  const idx = order.indexOf(stage);
  document.querySelectorAll("#steps li").forEach((li, i) => {
    li.classList.toggle("done", i < idx || (done && i === idx));
    li.classList.toggle("active", i === idx && !done);
  });
}
function setBar(frac) { $("#procBar").style.width = `${Math.round(frac * 100)}%`; }

function upload(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const fd = new FormData();
    fd.append("file", file);
    xhr.open("POST", "/api/upload");
    xhr.upload.onprogress = (e) => e.lengthComputable && setBar(0.1 * e.loaded / e.total);
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) {}
      xhr.status < 300 ? resolve(body) : reject(new Error(body.detail || "Upload failed"));
    };
    xhr.onerror = () => reject(new Error("Upload failed — is the SceneSeen server running?"));
    xhr.send(fd);
  });
}

async function runAnalysis(file) {
  show("processing");
  $("#procError").hidden = true;
  $("#procName").textContent = file.name;
  setStep("upload"); setBar(0);
  try {
    const up = await upload(file);
    state.vid = up.video_id;
    await analyzeAndWait(up.video_id);
    location.hash = `#/v/${up.video_id}`;
  } catch (e) {
    $("#procError").textContent = e.message;
    $("#procError").hidden = false;
    $("#procHint").innerHTML = '<a href="/">Try another video</a>';
  }
}

async function analyzeAndWait(vid) {
  let job = await post(`/api/videos/${vid}/analyze`);
  while (job.status === "queued" || job.status === "running") {
    if (job.stage) setStep(job.stage);
    setBar(0.1 + 0.9 * job.progress);
    await sleep(600);
    job = await api(`/api/jobs/${job.id}`);
  }
  if (job.status === "error") throw new Error(job.error);
  setStep("results", true); setBar(1);
}

// ------------------------------------------------------------------ results
async function loadResult(vid) {
  state.vid = vid;
  const r = await api(`/api/videos/${vid}/result${devOn() ? "?debug=1" : ""}`);
  state.result = r;
  state.params = null;
  renderResult(r);
  show("results");
}

function renderResult(r) {
  const player = $("#player");
  if (player.getAttribute("src") !== r.video_url) player.src = r.video_url;
  $("#resVideo").textContent = r.video;
  $("#resCount").textContent = `${r.scene_count} ${r.scene_count === 1 ? "Scene" : "Scenes"} Detected`;
  const meta = [`${fmtDur(r.duration)} video`];
  if (r.corrected) meta.push("edited by you");
  if (r.preview_only) meta.push("preview with new parameters — not saved");
  if (r.has_ground_truth) meta.push("ground truth saved");
  $("#resMeta").textContent = meta.join(" · ");
  $("#tlEnd").textContent = fmt(r.duration);

  // timeline
  const tl = $("#timeline");
  tl.querySelectorAll(".seg").forEach((n) => n.remove());
  r.scenes.forEach((s, i) => {
    const seg = el("div", { class: "seg", style: `width:${(100 * s.duration_seconds) / r.duration}%;background:${colorFor(i)}`,
      title: `Scene ${pad2(s.scene_id)} · ${fmt(s.start_seconds)} → ${fmt(s.end_seconds)}`, "data-i": i },
      s.duration_seconds / r.duration > 0.045 ? pad2(s.scene_id) : "");
    tl.insertBefore(seg, $("#playhead"));
  });

  // cards
  $("#cards").replaceChildren(...r.scenes.map((s, i) => {
    const c = colorFor(i);
    const last = i === r.scenes.length - 1;
    return el("article", { class: "card", style: `--c:${c}`, "data-i": i },
      el("div", { class: "thumb", style: s.thumb ? `background-image:url(${s.thumb})` : "", onclick: () => preview(i) }),
      el("div", { class: "body" },
        el("span", { class: "label" }, `SCENE ${pad2(s.scene_id)}`),
        el("span", { class: "range" }, `${fmt(s.start_seconds)} → ${fmt(s.end_seconds)}`),
        el("span", { class: "dur" }, `Duration: ${fmtDur(s.duration_seconds)}${devOn() && s.shot_count ? ` · ${s.shot_count} shots` : ""}`),
        el("div", { class: "row" },
          el("button", { class: "btn small", type: "button", onclick: () => preview(i) }, "Preview"),
          state.editing && !last && !r.preview_only
            ? el("button", { class: "btn small ghost", type: "button", onclick: () => correction({ op: "merge", scene_index: i }) }, "Merge with next")
            : null)));
  }));
  $("#editTools").hidden = !state.editing;
  $("#editToggle").disabled = !!r.preview_only;
  $("#exportClips").disabled = $("#exportJson").disabled = !!r.preview_only;
  if (devOn() && r.debug) renderDebug(r);
}

$("#timeline").addEventListener("click", (e) => {
  const r = state.result;
  const rect = e.currentTarget.getBoundingClientRect();
  const t = ((e.clientX - rect.left) / rect.width) * r.duration;
  const i = r.scenes.findIndex((s) => t >= s.start_seconds && t < s.end_seconds);
  if (e.target.classList.contains("seg")) preview(i);
  else { $("#player").currentTime = t; }
});

function preview(i) {
  const s = state.result.scenes[i];
  const p = $("#player");
  state.previewEnd = s.end_seconds;
  p.currentTime = s.start_seconds + 0.01;
  p.play().catch(() => {});
  // rAF is throttled in background tabs; a short timer keeps the stop point tight
  clearInterval(state.previewTimer);
  state.previewTimer = setInterval(() => {
    if (state.previewEnd == null) return clearInterval(state.previewTimer);
    if (!p.paused && p.currentTime >= state.previewEnd - 0.03) {
      p.pause(); p.currentTime = state.previewEnd - 0.04; state.previewEnd = null; clearInterval(state.previewTimer);
    }
  }, 30);
  $(".player-wrap").scrollIntoView({ behavior: "smooth", block: "center" });
}

function currentScene(t) {
  const sc = state.result?.scenes || [];
  return sc.findIndex((s) => t >= s.start_seconds && t < s.end_seconds);
}
function tick() {
  const p = $("#player"), r = state.result;
  if (r && !$("#results").hidden) {
    const t = p.currentTime;
    if (state.previewEnd != null && !p.paused && t >= state.previewEnd - 0.03) {
      p.pause(); p.currentTime = state.previewEnd - 0.04; state.previewEnd = null;
    }
    $("#playhead").style.left = `${(100 * t) / r.duration}%`;
    const i = currentScene(t);
    document.querySelectorAll(".seg").forEach((n) => n.classList.toggle("active", +n.dataset.i === i));
    document.querySelectorAll(".card").forEach((n) => n.classList.toggle("active", +n.dataset.i === i));
    const now = $("#nowPlaying");
    if (i >= 0 && (t > 0 || !p.paused)) {
      const s = r.scenes[i];
      now.hidden = false;
      now.textContent = `Scene ${pad2(s.scene_id)} · ${fmt(s.start_seconds)} → ${fmt(s.end_seconds)}`;
    } else now.hidden = true;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
$("#player").addEventListener("seeking", () => {
  const p = $("#player");
  if (state.previewEnd != null && (p.currentTime > state.previewEnd || p.currentTime < state.previewEnd - 3600)) state.previewEnd = null;
});
$("#player").addEventListener("error", () => {
  $("#nowPlaying").hidden = false;
  $("#nowPlaying").textContent = "This browser cannot play this video format.";
});

// ------------------------------------------------------------------ exports
$("#exportJson").addEventListener("click", () => { location.href = `/api/videos/${state.vid}/export.json`; });
$("#exportClips").addEventListener("click", async () => {
  const btn = $("#exportClips"), st = $("#clipStatus");
  btn.disabled = true; st.hidden = false; st.textContent = "Preparing scene clips…";
  try {
    let job = await post(`/api/videos/${state.vid}/export-clips`);
    while (job.status === "queued" || job.status === "running") {
      st.textContent = `${job.stage_label || "Preparing scene clips"}… ${Math.round(job.progress * 100)}%`;
      await sleep(700);
      job = await api(`/api/jobs/${job.id}`);
    }
    if (job.status === "error") throw new Error(job.error);
    st.textContent = "Scene clips ready — downloading.";
    location.href = `/api/videos/${state.vid}/clips.zip`;
  } catch (e) { st.textContent = `Export failed: ${e.message}`; }
  btn.disabled = false;
});

// ------------------------------------------------------------------ corrections
$("#editToggle").addEventListener("change", (e) => { state.editing = e.target.checked; renderResult(state.result); });
async function correction(body) {
  $("#editState").textContent = "";
  try {
    const r = await post(`/api/videos/${state.vid}/corrections`, body);
    if (devOn()) return loadResult(state.vid);
    state.result = r; renderResult(r);
  } catch (e) { $("#editState").textContent = e.message; }
}
$("#splitBtn").addEventListener("click", () => correction({ op: "split", at: $("#player").currentTime }));
$("#resetBtn").addEventListener("click", () => correction({ op: "reset" }));
$("#clearBtn").addEventListener("click", () => {
  if (confirm("Start labelling from a single scene? You will add every boundary by hand with “Split at playhead”.")) correction({ op: "clear" });
});
$("#confirmBtn").addEventListener("click", async () => {
  try { const r = await post(`/api/videos/${state.vid}/confirm`); state.result = r; devOn() ? loadResult(state.vid) : renderResult(r); $("#editState").textContent = "Marked as reviewed."; }
  catch (e) { $("#editState").textContent = e.message; }
});
$("#gtBtn").addEventListener("click", async () => {
  const who = prompt("Save the current scene boundaries as ground truth for evaluation.\nOnly do this after watching the video.\n\nAnnotator name:", "");
  if (who === null) return;
  try {
    const res = await post(`/api/videos/${state.vid}/ground-truth`, { annotator: who });
    $("#editState").textContent = `Saved ${res.boundaries.length} boundaries → ${res.saved.split("/").slice(-2).join("/")}`;
  } catch (e) { $("#editState").textContent = e.message; }
});

// ------------------------------------------------------------------ developer view
const PARAM_HELP = {
  clip_weight: "semantic (CLIP) weight", color_weight: "colour weight", clip_floor: "CLIP similarity floor",
  window_shots: "context window (shots)", window_seconds: "context window (s)", distance_penalty: "penalty per shot gap",
  coherence_topk: "top-k links", abs_threshold: "min boundary score", depth_threshold: "min peak depth",
  strong_threshold: "score that skips depth test", min_scene_seconds: "min scene length (s)",
};
function renderDebug(r) {
  const d = r.debug;
  const t = d.timings;
  const rows = [
    ["Shot detector", d.shot_detector], ["Shots", `${r.shot_count} (raw ${d.raw_shot_count})`], ["Scenes", r.scene_count],
    ["Shot detection", sec(t.shot_detection ?? t.shot_detection_cached, t.shot_detection == null)],
    ["Frames + embeddings", sec(t.features ?? t.features_cached, t.features == null)],
    ["Grouping", `${(t.grouping * 1000).toFixed(1)} ms`], ["Total (this run)", t.total != null ? `${t.total}s` : "—"],
  ];
  $("#timings").replaceChildren(...rows.map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {}, String(v)))));

  if (!state.defaults) state.defaults = { ...d.grouping_config };
  const cur = state.params || d.grouping_config;
  $("#params").replaceChildren(...Object.entries(cur).map(([k, v]) => el("label", {}, PARAM_HELP[k] || k,
    el("input", { name: k, type: "number", step: Number.isInteger(v) && !k.includes("weight") ? "1" : "0.01", value: v }))));

  renderChart(r);
  const cuts = d.cuts;
  const sceneStart = new Set(r.scenes.map((s) => s.shot_start));
  const sceneOf = (i) => r.scenes.findIndex((s) => i >= s.shot_start && i <= s.shot_end) + 1;
  $("#shotSummary").textContent = `— ${d.shots.length} shots grouped into ${r.scene_count} scenes`;
  const head = el("tr", {}, ...["", "#", "start", "end", "dur", "scene", "cut score", "depth", "decision", "best link"].map((h) => el("th", {}, h)));
  $("#shotTable").replaceChildren(head, ...d.shots.map((s, i) => {
    const c = i > 0 ? cuts[i - 1] : null;
    return el("tr", { class: sceneStart.has(i) && i > 0 ? "newscene" : "" },
      el("td", {}, el("img", { loading: "lazy", src: d.thumb_base + i, alt: "", onclick: () => { $("#player").currentTime = s.start; } })),
      el("td", {}, i), el("td", {}, fmt(s.start)), el("td", {}, fmt(s.end)), el("td", {}, `${(s.end - s.start).toFixed(1)}s`),
      el("td", {}, sceneOf(i)),
      el("td", {}, c ? c.score.toFixed(3) : ""), el("td", {}, c ? c.depth.toFixed(3) : ""),
      el("td", {}, c ? el("span", { class: `pill ${c.decision}` }, c.decision.replace("_", " ")) : ""),
      el("td", { class: "muted" }, c && c.best_pair ? `${c.best_pair[0]} ↔ ${c.best_pair[1]} (${c.coherence.toFixed(2)})` : ""));
  }));
}
const sec = (v, cached) => (v == null ? "—" : `${Number(v).toFixed(1)}s${cached ? " (cached)" : ""}`);

function renderChart(r) {
  const W = Math.max(320, $("#chart").clientWidth || 1000), H = 220, P = 24;
  const cuts = r.debug.cuts, g = r.debug.grouping_config;
  const x = (t) => P + ((W - 2 * P) * t) / r.duration, y = (v) => H - P - (H - 2 * P) * Math.min(1, Math.max(0, v));
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const add = (tag, attrs, text) => {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text) e.textContent = text;
    svg.append(e); return e;
  };
  r.scenes.forEach((s, i) => add("rect", { x: x(s.start_seconds), y: H - P + 4, width: Math.max(1, x(s.end_seconds) - x(s.start_seconds) - 1), height: 6, fill: colorFor(i) }));
  const bw = Math.max(2, Math.min(8, (W - 2 * P) / Math.max(cuts.length, 1) * 0.6));
  for (const c of cuts) {
    const b = add("rect", { x: x(c.time) - bw / 2, y: y(c.score), width: bw, height: H - P - y(c.score), class: `bar-${c.decision}`, style: "cursor:pointer" });
    const title = document.createElementNS(ns, "title");
    title.textContent = `${fmt(c.time)}  score ${c.score.toFixed(3)}  depth ${c.depth.toFixed(3)}  ${c.decision}`;
    b.append(title);
    b.addEventListener("click", () => { $("#player").currentTime = c.time; });
  }
  for (const [v, label] of [[g.abs_threshold, "min score"], [g.strong_threshold, "strong"]]) {
    if (v > 1) continue;
    add("line", { x1: P, x2: W - P, y1: y(v), y2: y(v), stroke: "#8f8b85", "stroke-dasharray": "4 4", "stroke-width": 1 });
    add("text", { x: W - P, y: y(v) - 4, fill: "#8f8b85", "font-size": 11, "text-anchor": "end" }, `${label} ${v}`);
  }
  add("line", { x1: P, x2: W - P, y1: H - P, y2: H - P, stroke: "#2c2c33" });
  $("#chart").replaceChildren(svg);
}

function readParams() {
  const out = {};
  new FormData($("#params")).forEach((v, k) => { out[k] = Number(v); });
  return out;
}
$("#applyParams").addEventListener("click", async () => {
  $("#paramState").textContent = "…";
  try {
    const params = readParams();
    const t0 = performance.now();
    const r = await post(`/api/videos/${state.vid}/regroup`, { params });
    state.result = r; state.params = params;
    renderResult(r);
    $("#paramState").textContent = `${r.scene_count} scenes · ${Math.round(performance.now() - t0)} ms round-trip. Preview only — copy values into config/*.toml to keep them.`;
  } catch (e) { $("#paramState").textContent = e.message; }
});
$("#resetParams").addEventListener("click", () => { state.params = null; $("#paramState").textContent = ""; loadResult(state.vid); });

// ------------------------------------------------------------------ routing
async function route() {
  const m = location.hash.match(/^#\/v\/([0-9a-f]+)/);
  if (m) {
    try { await loadResult(m[1]); return; }
    catch (e) { $("#homeError").textContent = e.message; $("#homeError").hidden = false; }
  }
  state.file = null; $("#filePicked").hidden = true; fileInput.value = "";
  show("home"); loadRecent();
}
window.addEventListener("hashchange", route);
try { setDev(localStorage.getItem("sceneseen.dev") === "1" || new URLSearchParams(location.search).has("dev")); } catch (_) {}
route();
