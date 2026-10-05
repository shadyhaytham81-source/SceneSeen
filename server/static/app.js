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
  for (const v of ["home", "processing", "results", "catalog"]) $("#" + v).hidden = v !== view;
  $("#navCatalog").classList.toggle("on", view === "catalog");
  $("#navVideos").classList.toggle("on", view !== "catalog");
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------ developer mode
const devOn = () => document.body.classList.contains("devmode");
function setDev(on) {
  document.body.classList.toggle("devmode", on);
  $("#devToggle").checked = on;
  try { localStorage.setItem("sceneseen.dev", on ? "1" : "0"); } catch (_) {}
  if (on && state.vid && !$("#results").hidden) loadResult(state.vid);
  else if (state.commercial && !$("#results").hidden) renderCommercial();   // drop developer details at once
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
  state.uniqueOpen = null;
  state.commOpen = null; state.commFilter = "all";
  loadUnique(vid).then(() => loadCommercial(vid));
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
  if (r.preview_only) { $("#uniqPanel").hidden = true; $("#commPanel").hidden = true; }   // both follow the saved scenes
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

// ------------------------------------------------------------------ commercial objects (phase 2A)
const pct = (v) => `${Math.round(v * 100)}%`;
async function loadCommercial(vid) {
  const panel = $("#commPanel");
  panel.hidden = false;
  try { state.commercial = await api(`/api/videos/${vid}/commercial`); }
  catch (e) { state.commercial = { status: "unavailable", reason: e.message, scenes: [], summary: {} }; }
  if (devOn()) {
    try {
      const p = (await api("/api/commercial/report")).displayed;
      state.commPrecision = p.reviewed ? `${pct(p.precision)} (${p.correct} correct of ${p.reviewed} judged, all reviewed videos)` : null;
    } catch (_) {}
  }
  renderCommercial();
  loadProducts(vid);
}
async function runCommercial() {
  const btn = $("#commRun"), st = $("#commState"), bar = $("#commBarWrap");
  btn.disabled = true; bar.hidden = false; $("#commBar").style.width = "2%";
  st.textContent = "Starting… the first run downloads/loads the detection model.";
  try {
    let job = await post(`/api/videos/${state.vid}/commercial/analyze`);
    while (job.status === "queued" || job.status === "running") {
      st.textContent = `${job.stage_label || "Loading the detection model"}… ${Math.round(job.progress * 100)}%`;
      $("#commBar").style.width = `${Math.max(2, Math.round(job.progress * 100))}%`;
      await sleep(700);
      job = await api(`/api/jobs/${job.id}`);
    }
    if (job.status === "error") throw new Error(job.error);
    st.textContent = "";
  } catch (e) {
    state.commercial = { status: "unavailable", reason: e.message, scenes: [], summary: {} };
    bar.hidden = true; btn.disabled = false;
    return renderCommercial();
  }
  bar.hidden = true; btn.disabled = false;
  await loadCommercial(state.vid);
}
$("#commRun").addEventListener("click", runCommercial);
$("#reviewToggle").addEventListener("change", (e) => { state.reviewMode = e.target.checked; renderCommercial(); });

function renderCommercial() {
  const c = state.commercial;
  if (!c) return;
  const dev = devOn(), s = c.summary || {}, btn = $("#commRun"), st = $("#commState");
  const reviewing = dev || state.reviewMode, pfilter = state.prodFilter || "all";
  renderProductBar();
  const filter = state.commFilter || "all";
  btn.hidden = true; st.textContent = "";
  $("#commFilters").replaceChildren(); $("#commScenes").replaceChildren(); $("#commDev").hidden = true;
  if (c.status === "unavailable") {
    $("#commSummary").textContent = "Commercial analysis unavailable. Scenes and Unique Shots are not affected.";
    st.textContent = dev && c.reason ? c.reason : "";
    btn.hidden = false; btn.textContent = "Try again";
    return;
  }
  if (c.status === "not_run") {
    $("#commSummary").textContent = `Finds commercially relevant objects (fashion, electronics, cars, food & drink…) in each scene. Only ${s.unique_shots} unique shots are analysed, not every frame.`;
    btn.hidden = false; btn.textContent = "Run Commercial Analysis";
    return;
  }
  $("#commSummary").textContent = `${s.candidates_shown} commercial object${s.candidates_shown === 1 ? "" : "s"} in ${s.scenes} scene${s.scenes === 1 ? "" : "s"} · analysed ${s.frames_analysed} of ${s.frames_needed} unique shots`
    + (c.status === "partial" ? " · incomplete" : "");
  if (c.status === "partial") { btn.hidden = false; btn.textContent = "Analyse the rest"; st.textContent = dev && c.reason ? c.reason : ""; }

  // filters: category counts over displayed candidates + scene contexts
  const counts = {};
  c.scenes.forEach((sc) => {
    sc.candidates.filter((x) => x.displayed).forEach((x) => { counts[x.category] = (counts[x.category] || 0) + 1; });
    const v = sc.context && sc.context.venue;
    if (v && v.category) counts[v.category] = (counts[v.category] || 0) + 1;
  });
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const chip = (id, name, n) => el("button", { class: "chip" + (filter === id ? " on" : ""), type: "button", role: "tab",
    onclick: () => { state.commFilter = id; renderCommercial(); } }, name, el("span", { class: "n" }, String(n)));
  $("#commFilters").replaceChildren(chip("all", "All", total),
    ...c.categories.filter((k) => counts[k.id]).map((k) => chip(k.id, k.name, counts[k.id])));

  const seek = (t) => { const p = $("#player"); state.previewEnd = null; p.currentTime = t + 0.01; $(".player-wrap").scrollIntoView({ behavior: "smooth", block: "center" }); };
  const frameUrl = (b, extra) => `${c.frame_base}${b.shot_id}/${Math.round(b.position * 100)}${extra || ""}`;
  $("#commScenes").replaceChildren(...c.scenes.map((sc, i) => {
    const ctx = sc.context || {}, venue = ctx.venue, env = ctx.environment;
    const all = sc.candidates.filter((x) => x.displayed || dev);
    const cands = all.filter((x) => (filter === "all" || x.category === filter) && (pfilter === "all" || productStatus(x) === pfilter));
    if (pfilter !== "all" && !cands.length) return null;
    const venueMatch = venue && venue.category && (filter === "all" || venue.category === filter);
    if (filter !== "all" && !cands.length && !venueMatch) return null;
    const grid = el("div", { class: "cgrid" });
    cands.forEach((x) => {
      const b = x.best;
      const card = el("button", { class: "ccard" + (x.displayed ? "" : " hiddenc"), type: "button", "data-key": x.key,
        title: x.displayed ? "Click to see where it appears" : `Hidden from normal view: ${x.hidden_reason}` },
        el("div", { class: "crop", style: `background-image:url(${frameUrl(b, `?box=${b.box.join(",")}&size=360`)})` },
          el("span", { class: "seen" }, `Seen ×${x.seen_count}`)),
        el("div", { class: "cbody" },
          el("span", { class: "clabel" }, `${x.icon} ${x.label}`),
          el("span", { class: "ccat" }, x.category_name),
          el("span", { class: "cmeta" }, `Confidence: ${pct(x.detection_confidence)}`),
          el("span", { class: "cmeta" }, `${fmt(x.first_seen)}–${fmt(x.last_seen)}`),
          productBadge(x),
          reviewing && !dev ? reviewButtons(x) : null,
          dev ? el("div", { class: "cdev" },
            `raw "${x.debug.raw_label}" · det ${x.detection_confidence.toFixed(3)} (min ${x.debug.threshold})`, el("br"),
            `relevance ${x.commercial_relevance.toFixed(2)} = base ${x.debug.relevance_factors.base} · size ${x.debug.relevance_factors.size} · centre ${x.debug.relevance_factors.centrality} · persist ${x.debug.relevance_factors.persistence}`, el("br"),
            `box [${b.box.map((v) => v.toFixed(2)).join(", ")}] · shot #${b.shot_id}`, el("br"),
            `unique: ${x.debug.unique_shot_ids.join(", ")}`,
            x.displayed ? null : el("div", {}, `hidden: ${x.hidden_reason}`),
            reviewButtons(x)) : null));
      card.addEventListener("click", (ev) => {
        if (ev.target.closest(".rev")) return;
        const open = card.classList.contains("open");
        grid.querySelectorAll(".cdetail").forEach((n) => n.remove());
        grid.querySelectorAll(".ccard.open").forEach((n) => n.classList.remove("open"));
        state.openCard = open ? null : x.key;
        if (open) return;
        card.classList.add("open");
        const bx = b.box;
        const tb = state.unique && state.unique.thumb_base;
        const det = el("div", { class: "cdetail" },
          el("div", {}, el("h5", {}, `${x.icon} ${x.label} · ${x.category_name}`),
            el("div", { class: "cframe" }, el("img", { src: frameUrl(b, "?size=960"), alt: "", loading: "lazy" }),
              el("div", { class: "bbox", style: `left:${bx[0] * 100}%;top:${bx[1] * 100}%;width:${(bx[2] - bx[0]) * 100}%;height:${(bx[3] - bx[1]) * 100}%` })),
            el("p", { class: "muted small", style: "margin:8px 0 0" },
              `Detected in ${x.detected_in_unique_shots} unique shot${x.detected_in_unique_shots > 1 ? "s" : ""}; those camera set-ups occur ${x.seen_count} time${x.seen_count > 1 ? "s" : ""} in this scene (${fmtDur(x.screen_time)} on screen).`)),
          el("div", {}, el("h5", {}, `Where it appears (${x.seen_count})`),
            el("div", { class: "cocc" }, ...x.occurrences.map((o) => tb
              ? shotTile({ thumb_base: tb }, o.shot_id, { cap: `${fmt(o.start)} → ${fmt(o.end)}`, badge: o.detected ? "analysed" : "",
                  title: o.detected ? "Detected in this frame" : "Same camera set-up as an analysed shot", onclick: () => seek(o.start) })
              : el("button", { class: "btn small", type: "button", onclick: () => seek(o.start) }, `${fmt(o.start)} → ${fmt(o.end)}`)))),
          x.displayed ? productBlock(x) : null);
        card.after(det);
      });
      grid.append(card);
    });
    const d = el("details", { class: "cscene", style: `--c:${colorFor(i)}` },
      el("summary", {}, el("span", { class: "sname" }, `SCENE ${pad2(sc.scene_id)}`),
        venue ? el("span", { class: "tag", title: `Scene context (confidence ${pct(venue.confidence)}). A property of the whole scene, not an object.` }, `${venue.icon || "📍"} ${venue.label}`) : null,
        env ? el("span", { class: "tag soft" }, env.label) : null,
        el("span", { class: "tag soft" }, `${sc.candidates.filter((x) => x.displayed).length} objects`),
        ...sc.categories_present.map((k) => el("span", { class: "tag soft" }, k.name)),
        dev ? el("span", { class: "muted small" }, `context: ${(ctx.ranked || []).map((r) => `${r.label} ${r.confidence.toFixed(2)}`).join(" · ")} · analysed ${sc.unique_shots_analysed}/${sc.unique_shots} unique shots`) : null),
      cands.length ? grid : el("p", { class: "cempty" }, filter === "all" ? "No commercial objects found in this scene." : "Scene context only."),
      dev ? missedBox(sc) : null);
    d.open = state.commOpen ? state.commOpen.has(sc.scene_id) : i < 2;
    d.addEventListener("toggle", () => {
      state.commOpen = state.commOpen || new Set(c.scenes.slice(0, 2).map((z) => z.scene_id));
      d.open ? state.commOpen.add(sc.scene_id) : state.commOpen.delete(sc.scene_id);
    });
    return d;
  }).filter(Boolean));

  if (state.openCard) {      // keep the open object open across re-renders (after a review or a product decision)
    const again = [...document.querySelectorAll("#commScenes .ccard")].find((n) => n.dataset.key === state.openCard);
    state.openCard = null;
    if (again) { const d = again.closest("details"); if (d) d.open = true; again.click(); }
  }
  if (dev) {
    const m = c.model || {}, t = c.timings || {};
    $("#commDev").hidden = false;
    $("#commDev").replaceChildren(el("h4", {}, "Commercial analysis · developer"),
      el("table", { class: "kv" }, ...[
        ["Model", `${m.name || "?"} (${m.license || "?"})`], ["Version / device", `${m.version || "cached results"} · ${m.device || "not loaded"}`],
        ["Taxonomy version · detector key", `${c.taxonomy_version} · ${m.detector_key}`],
        ["Unique shots analysed", `${s.frames_analysed} of ${s.frames_needed} (inference avoided for ${s.inference_avoided_by_unique_shots} repeated shots)`],
        ["Inference per frame", s.mean_seconds_per_frame != null ? `${s.mean_seconds_per_frame.toFixed(2)} s` : "—"],
        ["This request", `${t.total} s (frames ${t.frames} · inference ${t.inference} · assembly ${t.assemble}); ${s.frames_from_cache} frames from cache, ${s.frames_inferred_now} inferred now`],
        ["Shown / hidden candidates", `${s.candidates_shown} / ${s.candidates_hidden}`],
        ["Thresholds", `confidence ≥ ${c.config.min_confidence}, relevance ≥ ${c.config.min_relevance}`],
        ["Review precision so far", state.commPrecision || "mark detections Correct / Wrong to measure"],
      ].map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {}, String(v))))));
  }
}
const VERDICTS = [
  ["correct", "✓", "Correct", "ok", "The object is there, the label is right, and it is commercially useful"],
  ["wrong", "✗", "Wrong", "bad", "There is no such object here"],
  ["unsure", "?", "Unsure", "mid", "Cannot tell"],
  ["wrong_label", "🏷", "Wrong label", "mid", "A real object, but it should be called something else"],
  ["not_commercial", "🚫", "Not useful", "mid", "Correctly detected, but of no commercial value"],
  ["bad_image", "👁", "Unclear image", "mid", "Too dark, blurred or cropped to judge"],
  ["duplicate", "🔁", "Duplicate", "mid", "The same object is already listed in this scene"],
];
function reviewButtons(x) {
  const wrap = el("div", { class: "rev" });
  const mk = ([verdict, icon, text, cls, title]) => el("button", { type: "button", title, class: (x.review === verdict ? "on " : "") + cls,
    onclick: async (ev) => {
      ev.stopPropagation();
      if (x.review === verdict) return sendReview({ key: x.key, verdict: null });
      if (verdict === "wrong_label") return labelPicker(wrap, x);
      await sendReview({ key: x.key, verdict });
    } }, `${icon} ${text}`);
  wrap.append(...VERDICTS.map(mk));
  const c = x.review_correction;
  if (x.review === "wrong_label" && c) wrap.append(el("span", { class: "revnote" }, `→ ${c.label}${c.in_taxonomy ? "" : " (new label)"}`));
  return wrap;
}
async function labelPicker(wrap, x) {
  if (wrap.querySelector(".relabel")) return;
  if (!state.taxonomy) state.taxonomy = await api("/api/commercial/taxonomy");
  const sel = el("select", {}, el("option", { value: "" }, "Correct label…"),
    ...state.taxonomy.categories.map((k) => el("optgroup", { label: k.name },
      ...k.object_types.filter((o) => o.id !== x.type_id).map((o) => el("option", { value: o.id }, o.label)))),
    el("option", { value: "__other" }, "Something else…"));
  const other = el("input", { type: "text", placeholder: "Type the correct label", hidden: "" });
  const save = el("button", { type: "button", class: "on ok" }, "Save");
  sel.addEventListener("change", () => { other.hidden = sel.value !== "__other"; if (!other.hidden) other.focus(); });
  const go = async () => {
    if (sel.value && sel.value !== "__other") await sendReview({ key: x.key, verdict: "wrong_label", corrected_type_id: sel.value });
    else if (other.value.trim()) await sendReview({ key: x.key, verdict: "wrong_label", corrected_label: other.value.trim() });
  };
  save.addEventListener("click", (e) => { e.stopPropagation(); go(); });
  other.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); go(); } });
  const box = el("div", { class: "relabel" }, sel, other, save);
  box.addEventListener("click", (e) => e.stopPropagation());
  wrap.append(box);
}
function actorName() {
  if (!state.reviewer) {
    try { state.reviewer = localStorage.getItem("sceneseen.reviewer") || ""; } catch (_) {}
    if (!state.reviewer) state.reviewer = (prompt("Your name (stored with your reviews and product decisions):", "") || "").trim();
    try { if (state.reviewer) localStorage.setItem("sceneseen.reviewer", state.reviewer); } catch (_) {}
  }
  return state.reviewer;
}
function missedBox(sc) {
  const input = el("input", { type: "text", placeholder: "Missed object, e.g. watch" });
  const add = async () => { if (input.value.trim()) await sendReview({ missed_label: input.value.trim(), scene_id: sc.scene_id }); };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") add(); });
  return el("div", { class: "missed" }, "Missed:", ...(sc.missed || []).map((m) => el("span", { class: "tag" }, m.label)),
    input, el("button", { class: "btn small ghost", type: "button", onclick: add }, "Add"));
}
async function sendReview(body) {
  try {
    const r = await post(`/api/videos/${state.vid}/commercial/review`, { ...body, reviewer: actorName() });
    const p = r.precision;
    state.commPrecision = p.reviewed ? `${pct(p.precision)} (${p.correct} correct of ${p.reviewed} judged, all reviewed videos)` : null;
    const fresh = await api(`/api/videos/${state.vid}/commercial`);
    state.commercial = fresh;
    renderCommercial();
  } catch (e) { $("#commState").textContent = e.message; }
}

// ------------------------------------------------------------------ unique shots
const GROUP_COLORS = ["#f2b544", "#6cb4a8", "#c97b9d", "#7d9bd8", "#d98b5f", "#9fbf6b", "#b49ad9", "#d9c48b", "#e07a7a", "#5fb0d9"];
async function loadUnique(vid) {
  const panel = $("#uniqPanel");
  try {
    state.unique = await api(`/api/videos/${vid}/unique-shots`);
    panel.hidden = false;
    renderUnique();
  } catch (_) { panel.hidden = true; }
}
function shotTile(u, shotId, { badge, cap, title, color, onclick }) {
  const b = el("button", { class: "ushot" + (color ? " rep-dot" : ""), type: "button", title: title || "",
    style: `background-image:url(${u.thumb_base}${shotId})${color ? `;--g:${color}` : ""}`, onclick });
  if (badge) b.append(el("span", { class: "badge" }, badge));
  if (cap) b.append(el("span", { class: "cap" }, cap));
  return b;
}
function renderUnique() {
  const u = state.unique;
  if (!u) return;
  const mode = state.uniqueMode || "unique", s = u.summary;
  $("#uniqModeUnique").classList.toggle("on", mode === "unique");
  $("#uniqModeAll").classList.toggle("on", mode === "all");
  $("#uniqSummary").textContent = `Original Shots: ${s.original_shots} · Unique Shots: ${s.unique_shots} · Repeated Shots: ${s.repeated_shots} · Reduction: ${Math.round(s.reduction * 100)}%`;
  const seek = (t) => { const p = $("#player"); state.previewEnd = null; p.currentTime = t + 0.01; $(".player-wrap").scrollIntoView({ behavior: "smooth", block: "center" }); };
  $("#uniqScenes").replaceChildren(...u.scenes.map((sc, i) => {
    const grid = el("div", { class: "ugrid" });
    if (mode === "unique") {
      sc.unique.forEach((g) => {
        const tile = shotTile(u, g.representative_shot_id, {
          badge: g.occurrence_count > 1 ? `×${g.occurrence_count}` : "", cap: fmt(g.first_seen),
          title: `${g.unique_shot_id}: ${g.occurrence_count} occurrence(s). Click to inspect.`,
          onclick: () => {
            const open = tile.classList.contains("open");
            grid.querySelectorAll(".uocc").forEach((n) => n.remove());
            grid.querySelectorAll(".ushot.open").forEach((n) => n.classList.remove("open"));
            if (open) return;
            tile.classList.add("open");
            const occGrid = el("div", { class: "ugrid" }, ...g.occurrences.map((o) => shotTile(u, o.shot_id, {
              badge: o.shot_id === g.representative_shot_id ? "representative" : `${Math.round(o.similarity_to_representative * 100)}%`,
              cap: `#${o.shot_id} · ${fmt(o.start)} → ${fmt(o.end)}`, title: "Play from this occurrence",
              onclick: () => seek(o.start) })));
            const maybe = (g.possible_duplicates_of || []).map((r) => r.unique_shot_id).join(", ");
            const box = el("div", { class: "uocc" },
              el("h5", {}, `${g.unique_shot_id} · ${g.occurrence_count} occurrence${g.occurrence_count > 1 ? "s" : ""} · ${fmtDur(g.total_duration)} on screen`),
              occGrid, maybe ? el("p", { class: "muted small", style: "margin:8px 0 0" }, `Possibly the same set-up as: ${maybe} (uncertain, kept separate)`) : null);
            tile.after(box);
          } });
        grid.append(tile);
      });
    } else {
      const groupOf = {};
      sc.unique.forEach((g, k) => g.shot_ids.forEach((id) => { groupOf[id] = [g, k]; }));
      Object.keys(groupOf).map(Number).sort((a, b) => a - b).forEach((id) => {
        const [g, k] = groupOf[id];
        const o = g.occurrences.find((x) => x.shot_id === id);
        grid.append(shotTile(u, id, { cap: `#${id} · ${fmt(o.start)}`, color: GROUP_COLORS[k % GROUP_COLORS.length],
          badge: g.occurrence_count > 1 ? g.unique_shot_id.split("-")[1] : "", title: `${g.unique_shot_id}${g.occurrence_count > 1 ? ` (repeated ×${g.occurrence_count})` : ""}`,
          onclick: () => seek(o.start) }));
      });
    }
    const stat = (k, v) => el("span", { class: "stat" }, `${k}: `, el("b", {}, String(v)));
    const d = el("details", { class: "uscene", style: `--c:${colorFor(i)}` },
      el("summary", {}, el("span", { class: "sname" }, `SCENE ${pad2(sc.scene_id)}`),
        stat("Original Shots", sc.original_shots), stat("Unique Shots", sc.unique_shots),
        stat("Repeated Shots", sc.repeated_shots), stat("Reduction", `${Math.round(sc.reduction * 100)}%`)),
      grid);
    if (state.uniqueOpen ? state.uniqueOpen.has(sc.scene_id) : i === 0) d.open = true;
    d.addEventListener("toggle", () => {
      state.uniqueOpen = state.uniqueOpen || new Set(u.scenes.slice(0, 1).map((x) => x.scene_id));
      d.open ? state.uniqueOpen.add(sc.scene_id) : state.uniqueOpen.delete(sc.scene_id);
    });
    return d;
  }));
}
$("#uniqModeUnique").addEventListener("click", () => { state.uniqueMode = "unique"; renderUnique(); });
$("#uniqModeAll").addEventListener("click", () => { state.uniqueMode = "all"; renderUnique(); });

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
    state.result = r; renderResult(r); loadUnique(state.vid).then(() => loadCommercial(state.vid));
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
    $("#editState").textContent = `Saved ${res.boundaries.length} boundaries → ${res.saved.split(/[\\/]/).slice(-2).join("/")}. Add it to dev, val or test in ground_truth/splits.json to use it.`;
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
  if (location.hash.startsWith("#/catalog")) { show("catalog"); return openCatalog(); }
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
window.addEventListener("DOMContentLoaded", route);   // after catalog.js has loaded
