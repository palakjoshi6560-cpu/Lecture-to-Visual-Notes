"""Concept-map graph (NetworkX) and renderer (Matplotlib, PNG/SVG)."""
from __future__ import annotations

import io
import math
import textwrap

import networkx as nx
from matplotlib.figure import Figure
from matplotlib.patches import Patch

FILL = ["#CFE0FF", "#FFE3B3", "#CDEFD6", "#F8D1E0", "#E0D4F7", "#C9EBEE", "#EADFC8", "#DCE6C1"]
EDGE = ["#2F4BD8", "#B8791A", "#2E8B57", "#C2407A", "#6B4FC2", "#1E8A94", "#8A7440", "#6B8A2A"]
W, H = 13.0, 8.5  # figure inches; layout units are "13-inch widths"


def select_graph(concepts, relations, max_nodes=14):
    """Select a compact, topic-balanced concept map using source-backed edges."""
    ranked = sorted(concepts, key=lambda c: -c["importance"])
    lookup = {c["id"]: c for c in concepts}
    importance = {c["id"]: c["importance"] for c in concepts}
    max_nodes = max(1, int(max_nodes))
    # Keep the requested node budget strict for every lecture, regardless
    # of how many topic sections it contains.

    # These are definition descriptors or incomplete noun phrases, not useful
    # map concepts. Their full meaning remains in the notes/definitions.
    generic_phrases = {
        "minimal set", "organized collection", "organised collection", "related data",
        "logical unit", "single value", "repeating list", "matching row",
        "domain rule", "system property", "concurrent operation", "example table", "table", "row",
    }
    fragment_starts = {"a", "an", "the", "or", "and", "of", "to", "for", "with", "from", "by", "in", "on", "at", "as", "into", "than"}

    def usable(key):
        words = key.lower().split()
        return bool(words) and words[0] not in fragment_starts and key.lower() not in generic_phrases

    direct, contextual = [], []
    for rel in relations:
        if rel.get("kind") not in {"pattern", "context"}:
            continue
        source, target = rel.get("source"), rel.get("target")
        if source not in lookup or target not in lookup or source == target:
            continue
        if not usable(source) or not usable(target) or not rel.get("evidence"):
            continue
        if rel["kind"] == "context":
            if rel.get("label") not in {"includes", "has property"}:
                continue
            contextual.append(rel)
        else:
            direct.append(rel)

    selected_nodes, selected_edges, seen = set(), [], set()

    def add_edge(rel):
        source, target = rel["source"], rel["target"]
        signature = (source, rel.get("label", ""), target, rel.get("kind", "pattern"))
        if signature in seen or len(selected_nodes | {source, target}) > max_nodes:
            return False
        selected_nodes.update((source, target))
        selected_edges.append(rel)
        seen.add(signature)
        return True

    # Build a priority list of representative concepts for topic balancing.
    # A concept enters the graph only through a source-supported edge, so the
    # final map cannot contain isolated anchor nodes.
    topic_ids = sorted({c.get("topic", 0) for c in ranked})
    topic_anchors = []
    for topic in topic_ids:
        candidate = next((c for c in ranked
                          if c.get("topic", 0) == topic and usable(c["id"])), None)
        if candidate:
            topic_anchors.append(candidate["id"])

    # Select explicit enumerations as complete clusters so no list member
    # is silently omitted from a compact map.
    clusters = {}
    for rel in contextual:
        clusters.setdefault(rel["source"], []).append(rel)
    for parent, edges in sorted(clusters.items(), key=lambda item: -importance.get(item[0], 0.0)):
        members = {parent} | {edge["target"] for edge in edges}
        if len(members) <= max_nodes and len(selected_nodes | members) <= max_nodes:
            for edge in edges:
                add_edge(edge)

    # Treat coordinated components of one result as a single small group
    # (student ID + course ID -> composite key).
    formed_groups = {}
    for rel in direct:
        if rel.get("label") == "forms part of":
            formed_groups.setdefault(rel["target"], []).append(rel)
    for target, edges in sorted(formed_groups.items(), key=lambda item: -max(
            (importance.get(e["source"], 0.0) + importance.get(target, 0.0)) for e in item[1])):
        members = {target} | {edge["source"] for edge in edges}
        if len(members) <= max_nodes and len(selected_nodes | members) <= max_nodes:
            for edge in sorted(edges, key=lambda e: -importance.get(e["source"], 0.0)):
                add_edge(edge)

    # Keep a complete prerequisite chain together (for example, 1NF -> 2NF ->
    # 3NF) rather than showing only one isolated link from that topic.
    requires_edges = [r for r in direct if r.get("label") == "requires"]
    req_graph = nx.Graph()
    for rel in requires_edges:
        req_graph.add_edge(rel["source"], rel["target"])
    for component in sorted(nx.connected_components(req_graph), key=lambda group: -len(group)):
        component_edges = [r for r in requires_edges
                           if r["source"] in component and r["target"] in component]
        if len(component_edges) >= 2 and len(selected_nodes | set(component)) <= max_nodes:
            for rel in component_edges:
                add_edge(rel)

    # Preserve explicit selection/definition links (e.g. candidate key ->
    # primary key) before lower-value incidental relations consume the node budget.
    for rel in sorted((r for r in direct if r.get("label") == "selected as"),
                      key=lambda r: -(r.get("weight", 0.0) *
                                      (importance[r["source"]] + importance[r["target"]]))):
        add_edge(rel)

    # Prefer source-backed edges that connect a topic's representative
    # concept to another already selected concept. This reduces isolated
    # topic islands without manufacturing cross-topic relationships.
    for anchor in topic_anchors:
        candidates = [r for r in direct if anchor in {r["source"], r["target"]}]
        candidates.sort(key=lambda r: -(r.get("weight", 0.0) *
            (importance[r["source"]] + importance[r["target"]])))
        for rel in candidates:
            if add_edge(rel):
                break

    # Reserve at least one meaningful direct relation for each topic before
    # spending remaining space on the globally strongest edges.
    topic_ids = sorted({c.get("topic", 0) for c in ranked})
    for topic in topic_ids:
        candidates = [r for r in direct
                      if lookup[r["source"]].get("topic", 0) == topic
                      and lookup[r["target"]].get("topic", 0) == topic]
        candidates.sort(key=lambda r: -(r.get("weight", 0.0) *
            (importance[r["source"]] + importance[r["target"]])))
        for rel in candidates:
            if add_edge(rel):
                break

    for rel in sorted(direct, key=lambda r: -(r.get("weight", 0.0) *
            (importance[r["source"]] + importance[r["target"]]))):
        add_edge(rel)

    # Do not add standalone concepts as a fallback. Every displayed node
    # must be an endpoint of at least one evidence-backed selected edge.

    nodes = [dict(id=c["id"], label=c["name"], importance=c["importance"], topic=c.get("topic", 0))
             for c in ranked if c["id"] in selected_nodes]
    edges = []
    for r in selected_edges:
        source, target = r["source"], r["target"]
        # Remove harmless leading quantifiers from graph labels when the
        # canonical concept exists ("one candidate key" -> "candidate key").
        for endpoint_name in ("source", "target"):
            value = source if endpoint_name == "source" else target
            parts = value.split()
            if parts and parts[0] in {"one", "two", "three", "several", "many", "some", "each"}:
                canonical = " ".join(parts[1:])
                if canonical in lookup and canonical in selected_nodes:
                    if endpoint_name == "source": source = canonical
                    else: target = canonical
        edges.append(dict(source=source, target=target, label=r["label"], kind=r["kind"],
                          weight=round(r.get("weight", 0.0), 2),
                          evidence=[e.get("sentence") if isinstance(e, dict) else e for e in r.get("evidence", [])]))
    return dict(nodes=nodes, edges=edges)


def _wrap(label, width=15):
    return "\n".join(textwrap.wrap(label, width)) or label


def _layout(graph, sizes):
    """Lay out each connected component separately and pack them into a grid of cells."""
    G = nx.Graph()
    G.add_nodes_from(n["id"] for n in graph["nodes"])
    G.add_edges_from((e["source"], e["target"]) for e in graph["edges"])
    comps = sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: (-len(c), c))
    top = H / W
    big = len(comps) > 1 and len(comps[0]) >= 5
    rest = comps[1:] if big else comps
    cols = 1 if big or len(rest) <= 1 else min(3, math.ceil(math.sqrt(len(rest))))
    rows = max(1, math.ceil(len(rest) / cols))
    rx, rw = (0.62, 0.34) if big else (0.05, 0.9)
    cells = [(0.03, 0.08 * top, 0.56, 0.74 * top)] if big else []
    cw, ch = rw / cols, 0.74 * top / rows
    cells += [(rx + (k % cols) * cw, 0.08 * top + (rows - 1 - k // cols) * ch, cw, ch) for k in range(len(rest))]
    P = {}
    for comp, (x0, y0, cw_, ch_) in zip(comps, cells):
        sub = G.subgraph(comp)
        if len(comp) == 1:
            local = {comp[0]: (0.5, 0.5)}
        elif len(comp) == 2:
            local = {comp[0]: (0.0, 0.5), comp[1]: (1.0, 0.5)}
        else:
            raw = nx.kamada_kawai_layout(sub)
            xs, ys = [p[0] for p in raw.values()], [p[1] for p in raw.values()]
            sx, sy = (max(xs) - min(xs)) or 1, (max(ys) - min(ys)) or 1
            local = {k: ((p[0] - min(xs)) / sx, (p[1] - min(ys)) / sy) for k, p in raw.items()}
        for k, (lx, ly) in local.items():
            P[k] = [x0 + cw_ * (0.12 + 0.76 * lx), y0 + ch_ * (0.12 + 0.76 * ly)]
    ids = list(P)
    for _ in range(400):  # push overlapping label boxes apart
        moved = False
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                (wa, ha), (wb, hb) = sizes[ids[a]], sizes[ids[b]]
                dx, dy = P[ids[b]][0] - P[ids[a]][0], P[ids[b]][1] - P[ids[a]][1]
                ox, oy = (wa + wb) / 2 + 0.015 - abs(dx), (ha + hb) / 2 + 0.015 - abs(dy)
                if ox > 0 and oy > 0:
                    moved = True
                    if ox < oy:
                        s = (1 if dx >= 0 else -1) * ox / 2
                        P[ids[a]][0] -= s; P[ids[b]][0] += s
                    else:
                        s = (1 if dy >= 0 else -1) * oy / 2
                        P[ids[a]][1] -= s; P[ids[b]][1] += s
        for k in ids:
            w, h = sizes[k]
            P[k][0] = min(max(P[k][0], w / 2 + 0.01), 1 - w / 2 - 0.01)
            P[k][1] = min(max(P[k][1], h / 2 + 0.07), top - h / 2 - 0.07)
        if not moved:
            break
    return P


def _edge_point(c, o, size):
    dx, dy = o[0] - c[0], o[1] - c[1]
    t = min((size[0] / 2) / abs(dx) if dx else 9e9, (size[1] / 2) / abs(dy) if dy else 9e9)
    return c[0] + dx * t, c[1] + dy * t


def render(graph, topics, fmt="png"):
    """Render the concept map and return bytes (fmt: 'png' or 'svg')."""
    fig = Figure(figsize=(W, H), dpi=150, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, H / W); ax.axis("off")
    ax.text(0.02, H / W - 0.03, "Concept map", fontsize=17, fontweight="bold", color="#1A2233", va="top")
    ax.text(0.02, H / W - 0.062, "Larger boxes = more important concepts. Solid arrows = direct statements; dashed arrows = a list introduced in the preceding sentence.",
            fontsize=8.5, color="#5B6578", va="top")
    if not graph["nodes"]:
        ax.text(0.5, H / (2 * W), "No concepts to draw", ha="center", fontsize=14, color="#5B6578")
    max_imp = max([n["importance"] for n in graph["nodes"]] or [1]) or 1
    labels, fonts, sizes = {}, {}, {}
    for n in graph["nodes"]:
        labels[n["id"]] = _wrap(n["label"])
        fonts[n["id"]] = 9 + 4 * n["importance"] / max_imp
        lines = labels[n["id"]].split("\n")
        sizes[n["id"]] = (max(len(l) for l in lines) * fonts[n["id"]] * 0.00052 + 0.018, len(lines) * fonts[n["id"]] * 0.0011 + 0.012)
    pos = _layout(graph, sizes) if graph["nodes"] else {}
    for e in graph["edges"]:
        a, b = pos[e["source"]], pos[e["target"]]
        p, q = _edge_point(a, b, sizes[e["source"]]), _edge_point(b, a, sizes[e["target"]])
        solid = e["kind"] == "pattern"
        contextual = e["kind"] == "context"
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>" if (solid or contextual) else "-", color="#4B5568" if solid else "#8A7440",
                    lw=1.5 if solid else 1.2, ls="-" if solid else "--", shrinkA=2, shrinkB=2), zorder=1)
        if solid or contextual:
            ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, e["label"], fontsize=8, style="italic", color="#2B3445", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.92), zorder=3)
    used = sorted({n["topic"] for n in graph["nodes"]})
    for n in graph["nodes"]:
        c = n["topic"] % len(FILL)
        top3 = n["importance"] >= 0.8 * max_imp
        ax.text(*pos[n["id"]], labels[n["id"]], ha="center", va="center", fontsize=fonts[n["id"]], fontweight="bold" if top3 else "normal",
                color="#1A2233", linespacing=1.15, zorder=4,
                bbox=dict(boxstyle="round,pad=0.35", fc=FILL[c], ec=EDGE[c], lw=2.2 if top3 else 1.2))
    if used:
        titles = {t["id"]: t["title"] for t in topics}
        handles = [Patch(fc=FILL[t % len(FILL)], ec=EDGE[t % len(FILL)], label=titles.get(t, f"Topic {t + 1}")) for t in used]
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False, fontsize=9, title="Topics", title_fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=150, facecolor="white")
    return buf.getvalue()
