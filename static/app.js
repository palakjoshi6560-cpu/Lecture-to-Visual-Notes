const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const rx = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
let mode = "text", data = null;

function mark(text, names) {
  const html = esc(text);
  const list = [...new Set(names)].filter(Boolean).sort((a, b) => b.length - a.length).map(n => rx(esc(n)));
  return list.length ? html.replace(new RegExp(`\\b(${list.join("|")})`, "gi"), '<strong class="concept-term">$1</strong>') : html;
}
const where = e => (e.time ? e.time : "S" + (e.sentence + 1));

document.querySelectorAll(".seg button").forEach(b => b.onclick = () => {
  mode = b.dataset.mode;
  document.querySelectorAll(".seg button").forEach(x => x.setAttribute("aria-pressed", x === b));
  $("#textbox").hidden = mode !== "text";
  $("#filebox").hidden = mode !== "file";
  $("#linkbox").hidden = mode !== "link";
  if (mode === "file") audioStatus();
});
function activateTab(name) {
  document.querySelectorAll("[role=tab]").forEach(x => x.setAttribute("aria-selected", x.dataset.tab === name));
  document.querySelectorAll(".pane").forEach(p => p.hidden = p.id !== "pane-" + name);
  document.querySelectorAll("[data-open-tab]").forEach(x => x.classList.toggle("selected", x.dataset.openTab === name));
}
document.querySelectorAll("[role=tab]").forEach(t => t.onclick = () => activateTab(t.dataset.tab));
document.querySelectorAll("[data-open-tab]").forEach(card => card.onclick = () => {
  activateTab(card.dataset.openTab);
  document.querySelector(".result-detail")?.scrollIntoView({behavior: "smooth", block: "start"});
});
let audioChecked = false;
async function audioStatus() {  // tells the user, before they upload, whether audio transcription is ready
  if (audioChecked) return;
  const target = $("#audiostatus");
  target.textContent = "Checking audio transcription…";
  try {
    const r = await fetch("/api/audio-status");
    if (!r.ok) throw new Error("Audio status is unavailable.");
    const s = await r.json();
    target.textContent = !s.available
      ? `Audio transcription is not available: ${s.note}`
      : (s.cached || s.loaded)
        ? `Audio transcription is ready (model: ${s.model}).`
        : (s.note || `Audio transcription is available (model: ${s.model}); the model may download on first use.`);
    audioChecked = true;
  } catch (e) {
    target.textContent = "Could not check audio status. Try again when the server is available.";
  }
}
$("#sample").onclick = async () => {
  const button = $("#sample");
  button.disabled = true;
  try {
    const r = await fetch("/api/sample");
    if (!r.ok) throw new Error("Could not load the sample lecture.");
    const sample = await r.json();
    $("#text").value = sample.text || "";
    document.querySelector('[data-mode=text]').click();
    $("#status").textContent = "Sample lecture loaded.";
    $("#status").className = "";
  } catch (e) {
    $("#status").textContent = e.message;
    $("#status").className = "err";
  } finally {
    button.disabled = false;
  }
};

$("#form").onsubmit = async ev => {
  ev.preventDefault();
  const fd = new FormData();
  if (mode === "text") fd.append("text", $("#text").value);
  else if (mode === "file" && $("#file").files[0]) fd.append("file", $("#file").files[0]);
  else if (mode === "link") fd.append("link_url", $("#lecture-link").value.trim());
  const st = $("#status"), go = $("#go");
  st.className = "";
  st.textContent = mode === "file" && $("#file").files[0] && !/\.(txt|md)$/i.test($("#file").files[0].name)
    ? "Transcribing and analysing. This can take a few minutes."
    : mode === "link" ? "Reading captions or audio from the public lecture link. This may take several minutes…"
    : "Analysing…";
  go.disabled = true;
  $("#results-section").hidden = true;
  try {
    const r = await fetch("/api/process", {method: "POST", body: fd});
    const contentType = r.headers.get("content-type") || "";
    const j = contentType.includes("application/json") ? await r.json() : {};
    if (!r.ok) throw new Error(j.error || `Request failed (${r.status}).`);
    if (!j || !j.id || !Array.isArray(j.notes)) throw new Error("The server returned an unexpected response.");
    data = j;
    render(j);
    st.textContent = "";
  } catch (e) {
    st.className = "err";
    st.textContent = e.message || "Something went wrong. Please try again.";
  } finally {
    go.disabled = false;
  }
};

function render(d) {
  $("#results-section").hidden = false;
  const f = n => `/files/${d.id}/${n}`;
  $("#dl").innerHTML = `<a href="${f("concept-map.png")}?download=1">▧ Download PNG</a><a href="${f("concept-map.svg")}?download=1">✣ Download SVG</a><a href="${f("notes.json")}?download=1">▤ Download JSON</a>${["audio", "link"].includes(d.meta.source) ? `<a href="${f("transcript.txt")}?download=1">⇩ Transcript</a>` : ""}`;
  const m = d.meta || {};
  // Keep engineering/debug metadata out of the student-facing Study notes.
  // Detailed processing information is available on demand in the evidence tab.
  $("#meta").hidden = false;
  $("#meta").textContent = `${m.sentences ?? 0} sentences · ${m.topics ?? 0} topics · ${m.concepts ?? 0} concepts`;
  const processing = [];
  processing.push(`${m.sentences ?? 0} sentences · ${m.topics ?? 0} topics · ${m.concepts ?? 0} concepts · ${m.relations ?? 0} relationships`);
  processing.push(`Language analysis: ${m.nlp_backend === "spacy" ? "spaCy" : "built-in rules"}`);
  if (m.audio) {
    const secs = m.audio.duration_seconds, len = secs ? `${Math.floor(secs / 60)}:${String(Math.round(secs % 60)).padStart(2, "0")}` : "duration unavailable";
    processing.push(`Audio: ${len}, ${m.audio.language || "language unknown"}, ${m.audio.segments ?? 0} segments`);
  }
  if (m.warnings?.length) processing.push(`Warnings: ${m.warnings.join(" ")}`);
  if (m.terminology_corrections?.length) {
    const changes = [...new Set(m.terminology_corrections.map(c => `${c.original} → ${c.corrected}`))];
    processing.push(`Terminology corrections: ${changes.join("; ")}`);
  }
  $("#pane-notes").innerHTML = d.notes.map(n => `
    <h2>${esc(n.title)}</h2>
    ${n.key_terms.length ? `<p class="terms">Key terms: ${esc(n.key_terms.join(", "))}</p>` : ""}
    ${n.bullets.map(b => `<div class="note"><div><button class="chip" aria-expanded="false" data-s="${b.sentence}" title="Show source sentence">${where(b)}</button></div>
      <div><span>${mark(b.text, b.concepts)}</span></div></div>`).join("")}`).join("");
  $("#pane-notes").querySelectorAll(".chip").forEach(c => c.onclick = () => {
    const open = c.getAttribute("aria-expanded") === "true", box = c.closest(".note").lastElementChild;
    c.setAttribute("aria-expanded", !open);
    if (open) box.querySelector(".src")?.remove();
    else box.insertAdjacentHTML("beforeend", `<p class="src">${esc(d.transcript[+c.dataset.s].text)}</p>`);
  });
  const q = d.quick_revision;
  $("#pane-quick").innerHTML = `<div class="qr"><h3>Key terms</h3><dl>${q.terms.map(t => `<dt>${esc(t.term)}</dt><dd>${esc(t.definition)}</dd>`).join("")}</dl>
    ${q.relationships.length ? `<h3>How they connect</h3><ul>${q.relationships.map(r => `<li>${esc(r.source)} <em>${esc(r.label)}</em> ${esc(r.target)}</li>`).join("")}</ul>` : ""}
    ${q.remember.length ? `<h3>Stressed in the lecture</h3><ul>${q.remember.map(r => `<li>${esc(r.text)}</li>`).join("")}</ul>` : ""}</div>`;
  $("#pane-map").innerHTML = `<div class="map"><img alt="Concept map of the lecture" src="${f("concept-map.svg")}?t=${Date.now()}"></div>`;
  const label = k => (d.concepts.find(c => c.id === k) || {name: k}).name;
  $("#pane-evidence").innerHTML = `<h3>Concepts by importance</h3>` + d.concepts.map(c => `<details class="c"><summary>
      <span><strong>${esc(c.name)}</strong>${c.aliases.length ? ` <span class="alias">(${esc(c.aliases.join(", "))})</span>` : ""}</span>
      <span class="meter" title="importance"><i style="width:${Math.round(c.importance * 100)}%"></i></span><span class="alias">${c.importance.toFixed(2)}</span></summary>
      ${c.definition ? `<p class="rel">Defined as: ${esc(c.definition)}</p>` : ""}
      ${c.evidence.map(e => `<blockquote>${where(e)}: ${mark(e.text, [c.name, ...c.aliases])}</blockquote>`).join("")}</details>`).join("") +
    `<h3>Relationships</h3>` + d.relations.filter(r => r.kind === "pattern").map(r => `<details class="c"><summary><span>${esc(label(r.source))} <em>${esc(r.label)}</em> ${esc(label(r.target))}</span></summary>
      ${r.evidence.map(e => `<blockquote>${where(e)}: ${esc(e.text)}</blockquote>`).join("")}</details>`).join("") +
    `<details class="c processing-details"><summary>Processing details (optional)</summary><ul>${processing.map(x => `<li>${esc(x)}</li>`).join("")}</ul><p class="hint">Original ASR wording and full correction records are preserved in the downloaded JSON.</p></details>`;
  activateTab("notes");
  $("#results-section").scrollIntoView({behavior: "smooth", block: "start"});
}
