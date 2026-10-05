"use strict";
// SceneSeen Phase 2B UI: human product identification per scene and the catalog page.
// Uses the helpers of app.js ($, el, api, post, state, pct, fmt, sleep).

const send = (method, path, body) => api(path, { method, body: body == null ? undefined : JSON.stringify(body) });
const when = (iso) => { try { return new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z").toLocaleString(); } catch (_) { return iso; } };

// ------------------------------------------------------------------ products per scene (human identification)
// The detector finds generic objects; a person says which brand / product each one is. Nothing here guesses.
const STATUS_UI = {
  confirmed: ["Confirmed", "s-ok"], product_identified: ["Product identified", "s-high"], brand_identified: ["Brand identified", "s-mid"],
  unidentified: ["Unidentified", "s-none"], unknown_product: ["Unknown product", "s-none"], no_product: ["No product", "s-none"],
  not_commercial: ["Not commercially useful", "s-none"],
};
async function loadProducts(vid) {
  try { state.products = await api(`/api/videos/${vid}/products`); }
  catch (e) { state.products = { catalog: null, catalog_error: e.message, scenes: [], status_counts: {} }; }
  if (vid !== state.vid) return;
  state.prodByKey = {};
  (state.products.scenes || []).forEach((sc) => sc.objects.forEach((o) => { state.prodByKey[o.key] = o; }));
  renderCommercial();
}
const productOf = (x) => (state.prodByKey || {})[x.key];
const productStatus = (x) => (productOf(x) || {}).status || "unidentified";
const identText = (i) => i ? [i.brand && i.brand.name, i.product && i.product.name, i.variant && (i.variant.name || i.variant.sku)].filter(Boolean).join(" · ") : "";

function renderProductBar() {
  const bar = $("#prodBar"), p = state.products, c = state.commercial;
  const usable = c && (c.status === "ready" || c.status === "partial") && p && (p.scenes || []).length;
  bar.hidden = !usable; $("#prodFilters").replaceChildren();
  if (!usable) return;
  const n = p.status_counts || {}, sum = $("#prodSummary");
  const total = Object.values(n).reduce((a, b) => a + b, 0), done = total - (n.unidentified || 0);
  $("#prodState").textContent = p.catalog ? "" : "The product catalog is unavailable; objects are still listed.";
  sum.textContent = `${done} of ${total} objects identified by a person`
    + (n.confirmed ? ` · ${n.confirmed} confirmed` : "") + (n.brand_identified ? ` · ${n.brand_identified} brand only` : "");
  $("#prodExport").hidden = false;
  const f = state.prodFilter || "all";
  const chip = (id, name, k) => el("button", { class: "chip" + (f === id ? " on" : ""), type: "button", role: "tab",
    onclick: () => { state.prodFilter = id; renderCommercial(); } }, name, el("span", { class: "n" }, String(k)));
  $("#prodFilters").replaceChildren(chip("all", "All objects", total),
    ...Object.entries(STATUS_UI).filter(([id]) => n[id]).map(([id, [name]]) => chip(id, name, n[id])));
}
$("#prodExport").addEventListener("click", async () => {
  const data = await api(`/api/videos/${state.vid}/products/export`);
  const a = el("a", { href: URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })), download: "products.json" });
  a.click(); URL.revokeObjectURL(a.href);
});

function productBadge(x) {
  const o = productOf(x);
  if (!o || !x.displayed) return null;
  const [name, cls] = STATUS_UI[o.status], i = o.identification;
  const text = i && (i.brand || i.product) ? identText(i) : name;
  return el("span", { class: "idrow" },
    el("span", { class: `pbadge ${cls}`, title: name }, o.status === "confirmed" ? "✓ " : "", text),
    el("button", { class: "idbtn", type: "button", onclick: (e) => { e.stopPropagation(); openIdentify(x); } },
      i ? "Edit" : "Identify Product"));
}
function productBlock(x) {
  const o = productOf(x), wrap = el("div", { class: "pblock" });
  if (!o) return wrap;
  const [name, cls] = STATUS_UI[o.status], i = o.identification;
  wrap.append(el("div", { class: "row between" }, el("h5", {}, "Product"), el("span", { class: `pbadge ${cls}` }, name)),
    el("div", { class: "pfields" }, ...[
      ["Detection confidence (AI)", pct(o.detection_confidence)], ["Commercial relevance (AI)", o.commercial_relevance_level],
      ["Brand (human)", i && i.brand ? i.brand.name : "—"], ["Product (human)", i && i.product ? i.product.name : i && i.brand ? "Unknown" : "—"],
    ].map(([k, val]) => el("div", {}, el("span", { class: "muted" }, k), el("b", {}, val)))),
    i ? el("p", { class: "muted small" }, `${name} by ${i.identified_by} · ${when(i.identified_at)}${i.notes ? ` · “${i.notes}”` : ""}. Applies to all ${x.seen_count} occurrence${x.seen_count === 1 ? "" : "s"} of this object in the scene.`)
      : el("p", { class: "muted small" }, "Nobody has identified this object yet. SceneSeen does not guess brands or products."),
    el("div", { class: "row" }, el("button", { class: "btn small primary", type: "button", onclick: () => openIdentify(x) }, i ? "Change identification" : "Identify Product")));
  wrap.addEventListener("click", (e) => e.stopPropagation());
  return wrap;
}

// ---- identification panel
const idd = $("#identifyDlg");
const ident = {};
async function saveIdentification(body) {
  const actor = actorName();
  if (!actor) return iddError("Your name is needed: every identification records who made it.");
  try {
    await post(`/api/videos/${state.vid}/products/identify`, { key: ident.x.key, actor, ...body });
    state.openCard = ident.x.key;
    idd.close();
    await loadProducts(state.vid);
  } catch (e) { iddError(e.message); }
}
function iddError(msg) { const n = $("#iddError"); if (n) { n.hidden = !msg; n.textContent = msg || ""; } }
function pickRow(prod, sub, action, img) {
  return el("button", { class: "pick", type: "button", onclick: action },
    el("span", { class: "pimg", style: (img || prod.image) ? `background-image:url(${img || prod.image}?size=120)` : "" }),
    el("span", { class: "pinfo" }, el("span", { class: "pname" }, prod.name), el("span", { class: "muted small" }, sub)));
}
async function openIdentify(x) {
  const o = productOf(x) || { status: "unidentified" }, i = o.identification, c = state.commercial;
  const sc = c.scenes.find((s) => s.candidates.some((k) => k.key === x.key));
  Object.assign(ident, { x, o, sc, brand: i && i.brand ? { ...i.brand } : null, product: null, variantId: i && i.variant ? i.variant.id : null,
    allCategories: false, creating: false, files: [] });
  if (!cat.meta) { try { cat.meta = await api("/api/catalog/meta"); } catch (_) { cat.meta = null; } }
  if (i && i.product) { try { ident.product = await api(`/api/catalog/products/${i.product.id}`); } catch (_) {} }
  renderIdentify();
  if (!idd.open) idd.showModal();
}
function renderIdentify() {
  const { x, o, sc } = ident, i = o.identification, c = state.commercial;
  const crops = (x.debug && x.debug.detections || [{ shot_id: x.best.shot_id, box: x.best.box }]).slice(0, 8);
  const cropUrl = (d) => `${c.frame_base}${d.shot_id}/50?box=${d.box.join(",")}&size=260`;
  const [stName, stCls] = STATUS_UI[o.status];

  // ---- left: what was detected
  const hist = el("div", { class: "phist" });
  api(`/api/videos/${state.vid}/products/history?key=${x.key}`).then((r) => {
    const say = (v) => v ? ([v.brand, v.product, v.variant].filter(Boolean).join(" · ") || STATUS_UI[v.status][0]) + ` (${STATUS_UI[v.status][0].toLowerCase()})` : "unidentified";
    hist.replaceChildren(...r.events.map((h) => el("div", {}, `${when(h.at)} · ${h.actor}: `,
      h.action === "identify" ? say(h.after) : h.action === "clear" ? `removed ${say(h.before)}` : `${say(h.before)} → ${say(h.after)}`)));
  }).catch(() => {});
  const left = el("div", { class: "idobj" },
    el("div", { class: "idk" }, "Detected object"), el("div", { class: "idv big" }, `${x.icon} ${x.label}`),
    el("div", { class: "idk" }, "Scene"), el("div", { class: "idv" }, `Scene ${pad2(sc.scene_id)} · ${fmt(x.first_seen)}–${fmt(x.last_seen)} · seen ×${x.seen_count}`),
    el("div", { class: "idk" }, "Occurrence images"),
    el("div", { class: "idcrops" }, ...crops.map((d) => el("img", { src: cropUrl(d), alt: "", loading: "lazy" }))),
    el("div", { class: "pfields" },
      el("div", {}, el("span", { class: "muted" }, "Detection confidence (AI)"), el("b", {}, pct(x.detection_confidence))),
      el("div", {}, el("span", { class: "muted" }, "Commercial relevance (AI)"), el("b", {}, o.commercial_relevance_level || "—"))),
    el("div", { class: "idk" }, "Current identification"),
    el("div", { class: "idv" }, el("span", { class: `pbadge ${stCls}` }, stName), i ? ` ${identText(i)}` : ""),
    i ? el("div", { class: "muted small" }, `by ${i.identified_by} · ${when(i.identified_at)}`) : null,
    el("div", { class: "idk" }, "History"), hist);

  // ---- right: brand
  const right = el("div", { class: "idpick" });
  const brandBox = el("div", {});
  if (ident.brand) {
    brandBox.append(el("div", { class: "picked" }, el("b", {}, ident.brand.name),
      el("button", { class: "x", type: "button", title: "Choose another brand", onclick: () => { ident.brand = null; ident.product = null; ident.variantId = null; renderIdentify(); } }, "×")));
  } else {
    const input = el("input", { type: "search", placeholder: "Search brand…", autocomplete: "off" }), out = el("div", { class: "picklist" });
    let t;
    const run = async () => {
      const q = input.value.trim();
      const r = await api(`/api/catalog/brands?${new URLSearchParams({ q, compatible_with: x.type_id, limit: "8" })}`).catch(() => ({ brands: [] }));
      const exact = r.brands.some((b) => b.name.toLowerCase() === q.toLowerCase());
      out.replaceChildren(...r.brands.map((b) => el("button", { class: "pick slim", type: "button", onclick: () => { ident.brand = b; renderIdentify(); } },
        el("span", { class: "pname" }, b.name), el("span", { class: "muted small" }, b.product_count ? `${b.product_count} matching product${b.product_count === 1 ? "" : "s"}` : ""))),
        q && !exact ? el("button", { class: "pick slim new", type: "button", onclick: async () => {
          try { ident.brand = await post("/api/catalog/brands", { name: q }); renderIdentify(); } catch (e) { iddError(e.message); }
        } }, `+ Create brand “${q}”`) : "",
        !r.brands.length && !q ? el("p", { class: "muted small" }, "No brands yet. Type a name to create one.") : "");
    };
    input.addEventListener("input", () => { clearTimeout(t); t = setTimeout(run, 180); });
    brandBox.append(input, out); run();
  }

  // ---- right: product
  const prodBox = el("div", {});
  if (ident.product) {
    const p = ident.product;
    prodBox.append(el("div", { class: "picked" },
      el("span", { class: "pimg", style: p.primary_image ? `background-image:url(${p.primary_image.url}?size=120)` : "" }),
      el("span", { class: "pinfo" }, el("b", {}, p.name), el("span", { class: "muted small" }, [p.brand.name, p.sku, p.object_label].filter(Boolean).join(" · "))),
      el("button", { class: "x", type: "button", title: "Choose another product", onclick: () => { ident.product = null; ident.variantId = null; renderIdentify(); } }, "×")));
    if (p.variants && p.variants.length) {
      const sel = el("select", {}, el("option", { value: "" }, "No specific variant"),
        ...p.variants.map((v) => el("option", { value: v.id }, [v.name, v.color, v.size, v.sku].filter(Boolean).join(" · "))));
      sel.value = ident.variantId || "";
      sel.addEventListener("change", () => { ident.variantId = sel.value ? Number(sel.value) : null; });
      prodBox.append(el("label", { class: "idk" }, "Variant (optional)"), sel);
    }
  } else {
    const input = el("input", { type: "search", placeholder: "Search product name or SKU…", autocomplete: "off" }), out = el("div", { class: "picklist" });
    const all = el("label", { class: "switch small" }, el("input", { type: "checkbox", ...(ident.allCategories ? { checked: "" } : {}) }),
      el("span", {}, "Show every category"));
    let t;
    const run = async () => {
      const q = new URLSearchParams({ q: input.value.trim(), limit: "12" });
      if (!all.firstChild.checked) q.set("compatible_with", x.type_id);
      if (ident.brand) q.set("brand_id", ident.brand.id);
      const r = await api(`/api/catalog/products?${q}`).catch((e) => ({ items: [], total: 0, error: e.message }));
      out.replaceChildren(...r.items.map((p) => pickRow(p, [p.brand.name, p.sku, p.object_label || p.category_name].filter(Boolean).join(" · "),
        async () => { ident.product = await api(`/api/catalog/products/${p.id}`); ident.brand = { ...p.brand }; ident.variantId = null; renderIdentify(); },
        p.primary_image && p.primary_image.url)),
        r.total > r.items.length ? el("p", { class: "muted small" }, `${r.total - r.items.length} more. Type a name or SKU to narrow down.`) : "",
        !r.items.length ? el("p", { class: "muted small" }, r.error || (ident.brand ? `No ${all.firstChild.checked ? "" : x.label.toLowerCase() + "-compatible "}products of ${ident.brand.name} match. You can save the brand only, or add the product below.` : "No matching products in the catalog.")) : "");
    };
    input.addEventListener("input", () => { clearTimeout(t); t = setTimeout(run, 180); });
    const hint = el("p", { class: "muted small", style: "margin:4px 0" });
    const setHint = () => { hint.textContent = all.firstChild.checked ? "Listing products of every category."
      : `Only products a ${x.type_id.replace(/_/g, " ")} could be are listed (${x.category_name}). This is a filter, not a suggestion.`; };
    all.firstChild.addEventListener("change", () => { ident.allCategories = all.firstChild.checked; setHint(); run(); });
    setHint();
    prodBox.append(el("div", { class: "row" }, input, all), hint, out);
    run();
  }

  const notes = el("input", { type: "text", placeholder: "Notes (optional)", value: (i && i.notes) || "", maxlength: "500" });
  const withNotes = (b) => ({ ...b, notes: notes.value.trim() || null });
  const actions = el("div", { class: "row idactions" });
  if (ident.product) {
    actions.append(el("button", { class: "btn primary", type: "button", onclick: () => saveIdentification(withNotes({ product_id: ident.product.id, variant_id: ident.variantId, confirm: true })) }, "Confirm Product"),
      el("button", { class: "btn ghost", type: "button", title: "Record the product now; someone confirms it later", onclick: () => saveIdentification(withNotes({ product_id: ident.product.id, variant_id: ident.variantId, confirm: false })) }, "Save, confirm later"));
  } else if (ident.brand) {
    actions.append(el("button", { class: "btn primary", type: "button", onclick: () => saveIdentification(withNotes({ brand_id: ident.brand.id })) }, "Save Brand Only"),
      el("span", { class: "muted small" }, "The exact product stays unknown."));
  } else actions.append(el("button", { class: "btn primary", type: "button", disabled: "" }, "Confirm Product"), el("span", { class: "muted small" }, "Choose a brand or a product first."));

  // ---- not in catalog
  const nic = el("details", { class: "nic", ...(ident.creating ? { open: "" } : {}) }, el("summary", {}, "Product not in catalog"));
  nic.addEventListener("toggle", () => { ident.creating = nic.open; });
  const f = { brand: el("input", { placeholder: "Brand", value: ident.brand ? ident.brand.name : "", list: "brandList" }), name: el("input", { placeholder: "Product name" }),
    variant: el("input", { placeholder: "Variant (optional), e.g. White / 42" }), sku: el("input", { placeholder: "SKU (optional)" }),
    url: el("input", { placeholder: "Product URL (optional)", type: "url" }),
    category: el("select", {}, ...((cat.meta && cat.meta.categories) || []).map((k) => el("option", { value: k.id }, k.name))),
    files: el("input", { type: "file", accept: "image/jpeg,image/png,image/webp", multiple: "" }) };
  f.category.value = x.category;
  const createBtn = el("button", { class: "btn", type: "button", onclick: async () => {
    iddError("");
    const bn = f.brand.value.trim(), pn = f.name.value.trim();
    if (!bn || !pn) return iddError("A new product needs a brand and a name.");
    createBtn.disabled = true;
    try {
      const found = (await api(`/api/catalog/brands?${new URLSearchParams({ q: bn, archived: "true" })}`)).brands.find((b) => b.name.toLowerCase() === bn.toLowerCase());
      const brand = found || await post("/api/catalog/brands", { name: bn });
      const p = await post("/api/catalog/products", { brand_id: brand.id, name: pn, category: f.category.value,
        object_type: f.category.value === x.category ? x.type_id : null, sku: f.sku.value.trim() || null, url: f.url.value.trim() || null });
      let variantId = null;
      if (f.variant.value.trim()) variantId = (await post(`/api/catalog/products/${p.id}/variants`, { name: f.variant.value.trim() })).id;
      if (f.files.files.length) await uploadImages(p.id, [...f.files.files], "front").catch((e) => iddError(`Product created, but the images were not added: ${e.message}`));
      await saveIdentification(withNotes({ product_id: p.id, variant_id: variantId, confirm: true }));
    } catch (e) { iddError(e.message); }
    createBtn.disabled = false;
  } }, "Create product and link it");
  nic.append(el("p", { class: "idk" }, "Create it now"),
    el("div", { class: "nicform" }, f.brand, f.name, f.variant, f.sku, f.url, f.category,
      el("label", { class: "muted small filelab" }, "Images (optional) ", f.files)),
    el("div", { class: "row" }, createBtn),
    el("p", { class: "idk" }, "…or mark this object as"),
    el("div", { class: "row" },
      ...[["unknown_product", "Unknown product", "A real product, but nobody can tell which"], ["no_product", "Generic / no specific product", "Nothing specific to link"],
        ["not_commercial", "Not commercially useful", "Not worth linking"]].map(([st, text, title]) =>
        el("button", { class: "btn small" + (o.status === st ? " primary" : ""), type: "button", title, onclick: () => saveIdentification(withNotes({ status: st })) }, text))));

  right.append(el("label", { class: "idk" }, "Brand"), brandBox, el("label", { class: "idk" }, "Product"), prodBox,
    el("label", { class: "idk" }, "Notes"), notes, actions, el("p", { class: "error left", id: "iddError", hidden: "" }), nic);
  idd.replaceChildren(el("div", { class: "dlg-body" },
    el("div", { class: "dlg-head" }, el("h3", {}, "Identify product"), el("button", { class: "x", type: "button", "aria-label": "Close", onclick: () => idd.close() }, "×")),
    el("div", { class: "dlg-cols idcols" }, left, right),
    el("div", { class: "dlg-foot" },
      i ? el("button", { class: "btn ghost small danger", type: "button", onclick: () => saveIdentification({ clear: true }) }, "Remove identification") : el("span", {}),
      el("button", { class: "btn ghost", type: "button", onclick: () => idd.close() }, "Cancel"))));
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
  $("#catNotice").hidden = true;
  await loadCatalogPage();
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
