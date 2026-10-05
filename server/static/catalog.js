"use strict";
// SceneSeen Phase 2B UI: products per scene (match + human verification) and the catalog page.
// Uses the helpers of app.js ($, el, api, post, state, pct, fmt, sleep).

const send = (method, path, body) => api(path, { method, body: body == null ? undefined : JSON.stringify(body) });
const when = (iso) => { try { return new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z").toLocaleString(); } catch (_) { return iso; } };

// ------------------------------------------------------------------ products per scene
const STATUS_UI = {
  confirmed: ["Confirmed", "s-ok"], high_confidence_candidate: ["High-confidence candidate", "s-high"],
  needs_review: ["Needs review", "s-mid"], unknown: ["Unknown product", "s-none"],
  no_match_confirmed: ["No match (checked)", "s-none"], no_catalog: ["Generic object", "s-none"], not_matched: ["Not matched yet", "s-none"],
};
async function loadProducts(vid) {
  try { state.products = await api(`/api/videos/${vid}/products`); }
  catch (e) { state.products = { matching: { status: "unavailable", reason: e.message }, scenes: [], status_counts: {} }; }
  if (vid !== state.vid) return;
  state.prodByKey = {};
  (state.products.scenes || []).forEach((sc) => sc.objects.forEach((o) => { state.prodByKey[o.key] = o; }));
  renderCommercial();
}
const productOf = (x) => (state.prodByKey || {})[x.key];
const productStatus = (x) => (productOf(x) || {}).status || "not_matched";

function renderProductBar() {
  const bar = $("#prodBar"), p = state.products, c = state.commercial;
  const usable = c && (c.status === "ready" || c.status === "partial") && p;
  bar.hidden = !usable; $("#prodFilters").replaceChildren();
  if (!usable) return;
  const m = p.matching || {}, cat = p.catalog, n = p.status_counts || {}, btn = $("#prodRun"), sum = $("#prodSummary");
  btn.hidden = false; btn.textContent = "Match Products"; $("#prodExport").hidden = true;
  if (!cat) {
    sum.textContent = "The product catalog is unavailable. Commercial objects are not affected.";
    btn.textContent = "Try again";
  } else if (!cat.products) {
    sum.replaceChildren("Add products to the ", el("a", { href: "#/catalog" }, "catalog"), " to find them in this video.");
    btn.hidden = true;
  } else if (m.status === "unavailable") {
    sum.textContent = "Product matching is unavailable. Commercial objects are not affected.";
    $("#prodState").textContent = devOn() ? (m.reason || "") : ""; btn.textContent = "Try again";
  } else if (m.status === "not_run") {
    sum.textContent = `Compare the detected objects with ${cat.products} catalog product${cat.products === 1 ? "" : "s"}.`;
  } else {
    const parts = [[n.confirmed, "confirmed"], [n.high_confidence_candidate, "high-confidence"], [n.needs_review, "to review"],
      [(n.unknown || 0) + (n.no_catalog || 0) + (n.no_match_confirmed || 0), "without a product"]].filter(([k]) => k);
    sum.textContent = parts.map(([k, t]) => `${k} ${t}`).join(" · ") || "No objects to match.";
    btn.textContent = cat.images_pending ? `Match again (${cat.images_pending} new image${cat.images_pending === 1 ? "" : "s"})` : "Match again";
    btn.classList.toggle("primary", !!cat.images_pending || m.status === "partial");
    $("#prodExport").hidden = false;
    const f = state.prodFilter || "all";
    const chip = (id, name, k) => el("button", { class: "chip" + (f === id ? " on" : ""), type: "button", role: "tab",
      onclick: () => { state.prodFilter = id; renderCommercial(); } }, name, el("span", { class: "n" }, String(k)));
    const total = Object.values(n).reduce((a, b) => a + b, 0);
    $("#prodFilters").replaceChildren(chip("all", "All objects", total),
      ...Object.entries(STATUS_UI).filter(([id]) => n[id]).map(([id, [name]]) => chip(id, name, n[id])));
    return;
  }
  btn.classList.add("primary");
}
async function runMatching() {
  const btn = $("#prodRun"), st = $("#prodState"), wrap = $("#prodBarWrap"), fill = $("#prodBarFill");
  btn.disabled = true; wrap.hidden = false; fill.style.width = "2%";
  st.textContent = "Starting…";
  try {
    let job = await post(`/api/videos/${state.vid}/products/match`);
    while (job.status === "queued" || job.status === "running") {
      st.textContent = `${job.stage_label || "Waiting for the previous job"}… ${Math.round(job.progress * 100)}%`;
      fill.style.width = `${Math.max(2, Math.round(job.progress * 100))}%`;
      await sleep(500);
      job = await api(`/api/jobs/${job.id}`);
    }
    st.textContent = job.status === "error" ? job.error : "";
  } catch (e) { st.textContent = e.message; }
  wrap.hidden = true; btn.disabled = false;
  await loadProducts(state.vid);
}
$("#prodRun").addEventListener("click", runMatching);
$("#prodExport").addEventListener("click", async () => {
  const data = await api(`/api/videos/${state.vid}/products/export`);
  const a = el("a", { href: URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })), download: "products.json" });
  a.click(); URL.revokeObjectURL(a.href);
});

function productBadge(x) {
  const o = productOf(x);
  if (!o || !x.displayed || o.status === "not_matched") return null;
  const [name, cls] = STATUS_UI[o.status];
  const top = o.candidates[0];
  const text = o.status === "confirmed" ? `${o.product.brand} · ${o.product.name}`
    : o.status === "high_confidence_candidate" && top ? `${top.product.brand} · ${top.product.name}?` : name;
  return el("span", { class: `pbadge ${cls}`, title: name }, o.status === "confirmed" ? "✓ " : "", text);
}
async function decide(x, body) {
  const actor = actorName();
  if (!actor) return;
  try {
    await post(`/api/videos/${state.vid}/products/verify`, { key: x.key, actor, ...body });
    state.openCard = x.key;
    await loadProducts(state.vid);
  } catch (e) { alert(e.message); }
}
function productRow(prod, sub, actions, img) {
  return el("div", { class: "prow" },
    el("div", { class: "pimg", style: (img || prod.image) ? `background-image:url(${img || prod.image}?size=160)` : "" }),
    el("div", { class: "pinfo" }, el("span", { class: "pname" }, prod.name),
      el("span", { class: "muted small" }, [prod.brand && (prod.brand.name || prod.brand), prod.sku].filter(Boolean).join(" · ")), sub),
    el("div", { class: "pact" }, ...actions));
}
function productBlock(x) {
  const o = productOf(x), wrap = el("div", { class: "pblock" });
  if (!o) return wrap;
  const [name, cls] = STATUS_UI[o.status], v = o.verification;
  const head = el("div", { class: "row between" }, el("h5", {}, "Product"), el("span", { class: `pbadge ${cls}` }, name));
  const fields = el("div", { class: "pfields" }, ...[
    ["Detection confidence", pct(o.detection_confidence)], ["Commercial relevance", pct(o.commercial_relevance)],
    ["Product match confidence", o.match_confidence == null || !o.candidates.length ? "—" : pct(o.match_confidence)],
    ["Verification", { unverified: "Not verified", confirmed: "Confirmed by a person", no_match: "Checked: no match" }[o.verification_status]],
  ].map(([k, val]) => el("div", {}, el("span", { class: "muted" }, k), el("b", {}, val))));
  const body = el("div", { class: "pbody" });
  const btn = (text, fn, cls2 = "") => el("button", { class: `btn small ${cls2}`, type: "button", onclick: (e) => { e.stopPropagation(); fn(); } }, text);
  const searchBox = el("div", { class: "psearch", hidden: "" });
  const openSearch = () => { searchBox.hidden = false; catalogSearchInto(searchBox, x); };
  const candidateRows = (dim) => o.candidates.map((k) => productRow(k.product,
    el("span", { class: "pconf" }, el("i", { style: `width:${Math.round(k.match_confidence * 100)}%` }),
      el("em", {}, `Match ${pct(k.match_confidence)}`),
      devOn() ? el("span", { class: "muted" }, ` · score ${k.score} · cos ${k.cosine} · colour ${k.colour} · lead ${k.margin}`) : null),
    [btn(o.status === "confirmed" ? "Select" : "Confirm", () => decide(x, { action: "confirm", product_id: k.product_id, source: "suggestion" }), dim ? "" : (k.rank === 1 ? "primary" : ""))],
    k.matched_image));
  if (o.status === "confirmed") {
    body.append(productRow(o.product, el("span", { class: "muted small" }, `Confirmed by ${v.decided_by} · ${when(v.decided_at)}`
      + (v.match_confidence != null ? ` · AI match confidence was ${pct(v.match_confidence)}` : v.source === "search" ? " · chosen from the catalog" : "")),
      [/^https?:\/\//i.test(o.product.url || "") ? el("a", { class: "btn small ghost", href: o.product.url, target: "_blank", rel: "noopener" }, "Open") : null,
        btn("Change", () => { more.hidden = false; }), btn("Undo", () => decide(x, { action: "clear" }), "ghost")]));
  } else if (o.status === "no_match_confirmed") {
    body.append(el("p", { class: "muted small" }, `${v.decided_by} checked this on ${when(v.decided_at)}: it is none of the catalog products. It stays a generic ${o.label.toLowerCase()}.`),
      el("div", { class: "row" }, btn("Undo", () => decide(x, { action: "clear" }), "ghost"), btn("Search catalog", openSearch)));
  } else if (o.status === "not_matched") {
    body.append(el("p", { class: "muted small" }, "Not compared with the catalog yet. Use “Match Products” above."),
      el("div", { class: "row" }, btn("Search catalog", openSearch)));
  } else if (o.status === "no_catalog") {
    body.append(el("p", { class: "muted small" }, `Generic object: the catalog has no products of this kind (${o.label.toLowerCase()}) yet.`),
      el("div", { class: "row" }, btn("Search catalog", openSearch), btn("No match", () => decide(x, { action: "no_match" }), "ghost")));
  } else {
    const msg = { high_confidence_candidate: "The first candidate is very likely this product. Please confirm it.",
      needs_review: o.match_state === "possible_match" ? "Possible match. A person should check before it is used." : "Uncertain. The candidates below are only suggestions.",
      unknown: "Unknown product. Nothing in the catalog is a reliable match; the closest items are listed only so you can check." }[o.status];
    body.append(el("p", { class: "muted small" }, msg + (o.match_note ? ` (${o.match_note}.)` : "")));
    const rows = el("div", { class: o.status === "unknown" ? "dim" : "" }, ...candidateRows(o.status === "unknown"));
    body.append(rows, el("div", { class: "row", style: "margin-top:8px" }, btn("No match", () => decide(x, { action: "no_match" })), btn("Search catalog", openSearch, "ghost")));
  }
  const more = el("div", { hidden: "" }, el("p", { class: "muted small" }, "Choose a different product:"), ...(o.status === "confirmed" ? candidateRows(true) : []),
    el("div", { class: "row" }, btn("Search catalog", openSearch), btn("No match", () => decide(x, { action: "no_match" }), "ghost")));
  const hist = el("div", { class: "phist" });
  const histBtn = el("button", { class: "linkbtn", type: "button", onclick: async (e) => {
    e.stopPropagation();
    const ev = (await api(`/api/videos/${state.vid}/products/history?key=${x.key}`)).events;
    hist.replaceChildren(...(ev.length ? ev.map((h) => el("div", {}, `${when(h.at)} · ${h.actor} · ${h.action.replace("_", " ")}`
      + (h.product_id ? ` · product #${h.product_id}` : "") + (h.payload && h.payload.match_confidence != null ? ` · AI ${pct(h.payload.match_confidence)}` : "")))
      : [el("div", {}, "No decisions yet.")]));
  } }, "History");
  wrap.append(head, fields, body, more, searchBox, histBtn, hist);
  wrap.addEventListener("click", (e) => e.stopPropagation());
  return wrap;
}
function catalogSearchInto(box, x) {
  if (box.firstChild) { box.querySelector("input").focus(); return; }
  const input = el("input", { type: "search", placeholder: `Search the catalog (${x.category_name})…` });
  const all = el("label", { class: "switch small" }, el("input", { type: "checkbox" }), el("span", {}, "All categories"));
  const out = el("div", {});
  let t;
  const run = async () => {
    const q = new URLSearchParams({ q: input.value.trim(), limit: "8" });
    if (!all.firstChild.checked) q.set("category", x.category);
    try {
      const r = await api(`/api/catalog/products?${q}`);
      out.replaceChildren(...(r.items.length ? r.items.map((p) => productRow(p, el("span", { class: "muted small" }, p.object_label || p.category_name),
        [el("button", { class: "btn small", type: "button", onclick: () => decide(x, { action: "confirm", product_id: p.id, source: "search" }) }, "Select")],
        p.primary_image && p.primary_image.url)) : [el("p", { class: "muted small" }, "No products found.")]),
        r.total > r.items.length ? el("p", { class: "muted small" }, `${r.total - r.items.length} more: refine the search.`) : "");
    } catch (e) { out.replaceChildren(el("p", { class: "muted small" }, e.message)); }
  };
  input.addEventListener("input", () => { clearTimeout(t); t = setTimeout(run, 220); });
  all.firstChild.addEventListener("change", run);
  box.append(el("div", { class: "row" }, input, all), out);
  input.focus(); run();
}

// ------------------------------------------------------------------ catalog page
const cat = { q: "", category: "", brand: "", archived: false, offset: 0, limit: 24, meta: null, brands: [], editing: null, pending: [] };
async function openCatalog() {
  try {
    [cat.meta, cat.brands] = await Promise.all([api("/api/catalog/meta"), api(`/api/catalog/brands`).then((r) => r.brands)]);
  } catch (e) {
    $("#catStats").textContent = ""; $("#catGrid").replaceChildren(); $("#catPager").replaceChildren(); $("#catCats").replaceChildren();
    $("#catNotice").hidden = false; $("#catNotice").textContent = `The catalog is unavailable: ${e.message}`;
    return;
  }
  const s = cat.meta.stats;
  $("#catStats").textContent = `${s.products.toLocaleString()} product${s.products === 1 ? "" : "s"} · ${s.brands} brand${s.brands === 1 ? "" : "s"} · ${s.images.toLocaleString()} images`
    + (s.archived_products ? ` · ${s.archived_products} archived` : "");
  const total = cat.meta.categories.reduce((a, k) => a + k.products, 0);
  const chip = (id, name, n) => el("button", { class: "chip" + (cat.category === id ? " on" : ""), type: "button", role: "tab",
    onclick: () => { cat.category = id; cat.offset = 0; openCatalog(); } }, name, el("span", { class: "n" }, String(n)));
  $("#catCats").replaceChildren(chip("", "All", total), ...cat.meta.categories.map((k) => chip(k.id, `${k.icon} ${k.name}`, k.products)));
  const bf = $("#catBrandFilter");
  bf.replaceChildren(el("option", { value: "" }, "All brands"), ...cat.brands.map((b) => el("option", { value: b.id }, `${b.name} (${b.product_count})`)));
  bf.value = cat.brand;
  $("#brandList").replaceChildren(...cat.brands.map((b) => el("option", { value: b.name })));
  catalogStatus();
  await loadCatalogPage();
}
async function catalogStatus() {
  const n = $("#catNotice");
  try {
    const st = await api("/api/catalog/status");
    n.hidden = !st.images_pending;
    if (st.images_pending) n.replaceChildren(`${st.images_pending} new image${st.images_pending === 1 ? " is" : "s are"} not prepared for matching yet. This happens automatically the next time you match a video. `,
      el("button", { class: "btn small", type: "button", onclick: prepareCatalog }, "Prepare now"));
  } catch (_) { n.hidden = true; }
}
async function prepareCatalog() {
  const n = $("#catNotice");
  try {
    let job = await post("/api/catalog/embed");
    while (job.status === "queued" || job.status === "running") {
      n.textContent = `${job.stage_label || "Waiting"}… ${Math.round(job.progress * 100)}%`;
      await sleep(500); job = await api(`/api/jobs/${job.id}`);
    }
    if (job.status === "error") { n.textContent = `Could not prepare the images: ${job.error}`; return; }
  } catch (e) { n.textContent = e.message; return; }
  catalogStatus();
}
async function loadCatalogPage() {
  const q = new URLSearchParams({ q: cat.q, category: cat.category, limit: cat.limit, offset: cat.offset, archived: cat.archived });
  if (cat.brand) q.set("brand_id", cat.brand);
  const r = await api(`/api/catalog/products?${q}`);
  const grid = $("#catGrid");
  if (!r.items.length) {
    const filtered = cat.q || cat.category || cat.brand;
    grid.replaceChildren(el("div", { class: "pempty" }, el("p", {}, filtered ? "No products match these filters." : "Your catalog is empty."),
      filtered ? null : el("p", { class: "muted small" }, "Add the products you want SceneSeen to recognise in videos: a name, a category and a few photos each."),
      filtered ? null : el("button", { class: "btn primary", type: "button", onclick: () => editProduct(null) }, "Add your first product")));
  } else {
    grid.replaceChildren(...r.items.map((p) => el("button", { class: "pcard" + (p.archived ? " arch" : ""), type: "button", onclick: () => editProduct(p.id) },
      el("div", { class: "pthumb", style: p.primary_image ? `background-image:url(${p.primary_image.url}?size=360)` : "" },
        p.primary_image ? null : el("span", {}, "No image"),
        p.image_count > 1 ? el("span", { class: "seen" }, `${p.image_count} images`) : null,
        p.archived ? el("span", { class: "seen left" }, "Archived") : null),
      el("div", { class: "cbody" }, el("span", { class: "ccat" }, p.brand.name), el("span", { class: "clabel" }, p.name),
        el("span", { class: "cmeta" }, [p.object_label || p.category_name, p.sku].filter(Boolean).join(" · ")),
        p.price != null ? el("span", { class: "cmeta" }, `${p.price.toLocaleString()} ${p.currency || ""}`) : null))));
  }
  const from = r.total ? r.offset + 1 : 0, to = r.offset + r.items.length;
  const nav = (text, off, dis) => el("button", { class: "btn small", type: "button", ...(dis ? { disabled: "" } : {}),
    onclick: () => { cat.offset = off; loadCatalogPage(); window.scrollTo(0, 0); } }, text);
  $("#catPager").replaceChildren(...(r.total > cat.limit ? [nav("← Previous", Math.max(0, r.offset - cat.limit), r.offset === 0),
    el("span", { class: "muted small" }, `${from.toLocaleString()}–${to.toLocaleString()} of ${r.total.toLocaleString()}`),
    nav("Next →", r.offset + cat.limit, to >= r.total)] : [el("span", { class: "muted small" }, r.total ? `${r.total} product${r.total === 1 ? "" : "s"}` : "")]));
}
{
  let t;
  $("#catSearch").addEventListener("input", (e) => { clearTimeout(t); t = setTimeout(() => { cat.q = e.target.value.trim(); cat.offset = 0; loadCatalogPage(); }, 220); });
  $("#catBrandFilter").addEventListener("change", (e) => { cat.brand = e.target.value; cat.offset = 0; loadCatalogPage(); });
  $("#catArchived").addEventListener("change", (e) => { cat.archived = e.target.checked; cat.offset = 0; loadCatalogPage(); });
  $("#catAdd").addEventListener("click", () => editProduct(null));
  $("#catBrands").addEventListener("click", openBrands);
}

// ---- product dialog
const pd = $("#productDlg"), pf = $("#productForm");
function fillTypes(category, selected) {
  const k = cat.meta.categories.find((c) => c.id === category);
  pf.object_type.replaceChildren(el("option", { value: "" }, "Any / not sure"), ...(k ? k.object_types : []).map((o) => el("option", { value: o.id }, o.label)));
  pf.object_type.value = selected || "";
}
function pdError(msg) { $("#pdError").hidden = !msg; $("#pdError").textContent = msg || ""; }
async function editProduct(id) {
  if (!cat.meta) return;
  pdError(""); pf.reset(); cat.pending = [];
  pf.category.replaceChildren(...cat.meta.categories.map((c) => el("option", { value: c.id }, c.name)));
  pf.availability.replaceChildren(...cat.meta.availability.map((a) => el("option", { value: a }, a.replace(/_/g, " "))));
  $("#pdRole").replaceChildren(...cat.meta.image_roles.map((r) => el("option", { value: r }, r)));
  const p = id ? await api(`/api/catalog/products/${id}`) : null;
  cat.editing = p;
  $("#pdTitle").textContent = p ? "Edit product" : "Add product";
  $("#pdSave").textContent = p ? "Save changes" : "Save product";
  $("#pdDelete").hidden = !p; $("#pdArchive").hidden = !p; $("#pdVariantsWrap").hidden = !p;
  if (p) {
    for (const k of ["name", "sku", "external_id", "url", "price", "currency", "description"]) pf[k].value = p[k] ?? "";
    pf.brand.value = p.brand.name; pf.category.value = p.category; pf.availability.value = p.availability;
    $("#pdArchive").textContent = p.archived ? "Restore" : "Archive";
  } else if (cat.category) pf.category.value = cat.category;
  fillTypes(pf.category.value, p && p.object_type);
  renderImages(); renderVariants();
  if (!pd.open) pd.showModal();
}
pf.category.addEventListener("change", () => fillTypes(pf.category.value, ""));
function renderImages() {
  const p = cat.editing, box = $("#pdImages");
  const tiles = (p ? p.images : []).map((im) => {
    const role = el("select", {}, ...cat.meta.image_roles.map((r) => el("option", { value: r }, r)));
    role.value = im.role;
    role.addEventListener("change", async () => { await send("PATCH", `/api/catalog/images/${im.id}`, { role: role.value }); im.role = role.value; });
    return el("div", { class: "imgtile" }, el("div", { class: "im", style: `background-image:url(${im.url}?size=240)` }), role,
      el("button", { class: "x", type: "button", title: "Remove image", onclick: async () => {
        await send("DELETE", `/api/catalog/images/${im.id}`); await editProduct(p.id);
      } }, "×"));
  });
  const queued = cat.pending.map((f, i) => el("div", { class: "imgtile" }, el("div", { class: "im", style: `background-image:url(${f.url})` }),
    el("span", { class: "muted small" }, f.role),
    el("button", { class: "x", type: "button", title: "Remove", onclick: () => { cat.pending.splice(i, 1); renderImages(); } }, "×")));
  box.replaceChildren(...tiles, ...queued);
}
async function uploadImages(productId, files, role) {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  fd.append("role", role);
  const res = await fetch(`/api/catalog/products/${productId}/images`, { method: "POST", body: fd });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}
async function addFiles(files) {
  files = [...files].filter((f) => /^image\/(jpeg|png|webp)$/.test(f.type));
  if (!files.length) return pdError("Only JPEG, PNG or WebP images can be added.");
  pdError("");
  const role = $("#pdRole").value;
  if (!cat.editing) {
    files.forEach((f) => cat.pending.push({ file: f, role, url: URL.createObjectURL(f) }));
    return renderImages();
  }
  try {
    const r = await uploadImages(cat.editing.id, files, role);
    await editProduct(cat.editing.id);
    if (r.failed.length) pdError(r.failed.map((f) => `${f.filename}: ${f.error}`).join(" · "));
  } catch (e) { pdError(e.message); }
}
{
  const drop = $("#pdDrop"), input = $("#pdFiles");
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { addFiles(input.files); input.value = ""; });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); addFiles(e.dataTransfer.files); });
}
function renderVariants() {
  const p = cat.editing;
  $("#pdVariants").replaceChildren(...(p ? p.variants : []).map((v) => el("div", { class: "vrow" },
    el("span", {}, [v.name, v.color, v.size].filter(Boolean).join(" · ")), el("span", { class: "muted small" }, v.sku || ""),
    el("button", { class: "linkbtn", type: "button", onclick: async () => { await send("DELETE", `/api/catalog/variants/${v.id}`); await editProduct(p.id); } }, "Remove"))));
}
$("#pdAddVariant").addEventListener("click", () => {
  const p = cat.editing, box = $("#pdVariants");
  if (box.querySelector("form, .vnew")) return;
  const f = { name: el("input", { placeholder: "Variant name, e.g. Black / 42" }), color: el("input", { placeholder: "Colour" }),
    size: el("input", { placeholder: "Size" }), sku: el("input", { placeholder: "SKU" }) };
  box.append(el("div", { class: "vnew" }, ...Object.values(f), el("button", { class: "btn small", type: "button", onclick: async () => {
    const body = Object.fromEntries(Object.entries(f).map(([k, i]) => [k, i.value.trim()]).filter(([, v]) => v));
    try { await post(`/api/catalog/products/${p.id}/variants`, body); await editProduct(p.id); } catch (e) { pdError(e.message); }
  } }, "Add")));
  f.name.focus();
});
pf.addEventListener("submit", async (e) => {
  e.preventDefault();
  pdError("");
  const v = (k) => pf[k].value.trim();
  if (!v("brand") || !v("name")) return pdError("A product needs a brand and a name.");
  $("#pdSave").disabled = true;
  try {
    let brand = cat.brands.find((b) => b.name.toLowerCase() === v("brand").toLowerCase());
    if (!brand) { brand = await post("/api/catalog/brands", { name: v("brand") }); cat.brands.push({ ...brand, product_count: 0 }); }
    const body = { brand_id: brand.id, name: v("name"), category: pf.category.value, object_type: pf.object_type.value || null,
      sku: v("sku") || null, external_id: v("external_id") || null, url: v("url") || null,
      price: v("price") === "" ? null : Number(v("price")), currency: v("currency").toUpperCase() || null,
      availability: pf.availability.value, description: v("description") || null };
    const p = cat.editing ? await send("PATCH", `/api/catalog/products/${cat.editing.id}`, body) : await post("/api/catalog/products", body);
    let failed = [];
    for (const role of [...new Set(cat.pending.map((f) => f.role))]) {
      const r = await uploadImages(p.id, cat.pending.filter((f) => f.role === role).map((f) => f.file), role).catch((err) => ({ failed: [{ filename: "images", error: err.message }] }));
      failed = failed.concat(r.failed || []);
    }
    cat.pending = [];
    if (failed.length) { await editProduct(p.id); pdError(`Saved, but some images were not added: ${failed.map((f) => `${f.filename}: ${f.error}`).join(" · ")}`); }
    else pd.close();
    await openCatalog();
  } catch (err) { pdError(err.message); }
  $("#pdSave").disabled = false;
});
$("#pdCancel").addEventListener("click", () => pd.close());
$("#pdClose").addEventListener("click", () => pd.close());
$("#pdArchive").addEventListener("click", async () => {
  await send("PATCH", `/api/catalog/products/${cat.editing.id}`, { archived: !cat.editing.archived });
  pd.close(); openCatalog();
});
$("#pdDelete").addEventListener("click", async () => {
  const p = cat.editing;
  if (!confirm(`Delete “${p.name}”? Its images are removed from the catalog.`)) return;
  const r = await send("DELETE", `/api/catalog/products/${p.id}`);
  pd.close(); await openCatalog();
  if (r.archived) { $("#catNotice").hidden = false; $("#catNotice").textContent = `“${p.name}” was archived instead of deleted: it is ${r.reason}.`; }
});

// ---- brands dialog
const bd = $("#brandDlg");
function bdError(msg) { $("#bdError").hidden = !msg; $("#bdError").textContent = msg || ""; }
async function openBrands() {
  bdError("");
  const brands = (await api("/api/catalog/brands?archived=true")).brands;
  $("#brandRows").replaceChildren(...(brands.length ? brands.map((b) => {
    const name = el("input", { value: b.name, "aria-label": "Brand name" }), site = el("input", { value: b.website || "", placeholder: "Website", type: "url" });
    const save = async (body) => { try { bdError(""); await send("PATCH", `/api/catalog/brands/${b.id}`, body); await openBrands(); } catch (e) { bdError(e.message); } };
    return el("div", { class: "brow" + (b.archived ? " arch" : "") }, name, site,
      el("span", { class: "muted small" }, b.archived ? "archived" : `${b.product_count ?? 0} products`),
      el("button", { class: "btn small", type: "button", onclick: () => save({ name: name.value, website: site.value || null }) }, "Save"),
      el("button", { class: "btn small ghost", type: "button", title: b.archived ? "" : "Hides the brand and its products from matching; nothing is deleted",
        onclick: () => save({ archived: !b.archived }) }, b.archived ? "Restore" : "Archive"));
  }) : [el("p", { class: "muted small" }, "No brands yet.")]));
  if (!bd.open) bd.showModal();
}
$("#brandForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  try { bdError(""); await post("/api/catalog/brands", { name: f.name.value, website: f.website.value || null }); f.reset(); await openBrands(); }
  catch (err) { bdError(err.message); }
});
$("#bdClose").addEventListener("click", () => { bd.close(); openCatalog(); });
bd.addEventListener("close", () => { if (!$("#catalog").hidden) openCatalog(); });
