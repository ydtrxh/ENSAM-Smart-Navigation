const state = {
  mode: "select",
  svg: null,
  graphLayer: null,
  edgeLayer: null,
  nodeLayer: null,
  nodes: [],
  edges: [],
  selectedNodeId: null,
  selectedEdgeKey: null,
  pendingEdgeNodeId: null,
  drag: null,
  svgFileName: "",
};

const els = {
  svgInput: document.getElementById("svgInput"),
  graphInput: document.getElementById("graphInput"),
  loadDefaultSvgBtn: document.getElementById("loadDefaultSvgBtn"),
  exportBtn: document.getElementById("exportBtn"),
  modeButtons: [...document.querySelectorAll(".mode-btn")],
  modeHint: document.getElementById("modeHint"),
  nodeForm: document.getElementById("nodeForm"),
  nodeId: document.getElementById("nodeId"),
  nodeLabel: document.getElementById("nodeLabel"),
  nodeRole: document.getElementById("nodeRole"),
  visibleDestination: document.getElementById("visibleDestination"),
  nodeX: document.getElementById("nodeX"),
  nodeY: document.getElementById("nodeY"),
  edgeInfo: document.getElementById("edgeInfo"),
  deleteBtn: document.getElementById("deleteBtn"),
  clearSelectionBtn: document.getElementById("clearSelectionBtn"),
  fitBtn: document.getElementById("fitBtn"),
  nodeCount: document.getElementById("nodeCount"),
  edgeCount: document.getElementById("edgeCount"),
  cursorCoords: document.getElementById("cursorCoords"),
  mapName: document.getElementById("mapName"),
  svgMeta: document.getElementById("svgMeta"),
  svgHost: document.getElementById("svgHost"),
  emptyState: document.getElementById("emptyState"),
  canvasWrap: document.getElementById("canvasWrap"),
};

const roleColors = {
  building: "#2563eb",
  entrance: "#16a34a",
  waypoint: "#f59e0b",
  intersection: "#7c3aed",
};

function svgPointFromEvent(event) {
  if (!state.svg) return null;
  const point = state.svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  const matrix = state.svg.getScreenCTM();
  if (!matrix) return null;
  const transformed = point.matrixTransform(matrix.inverse());
  return { x: transformed.x, y: transformed.y };
}

function getSvgDimensions(svg) {
  const viewBox = svg.viewBox?.baseVal;
  if (viewBox && viewBox.width && viewBox.height) {
    return {
      minX: viewBox.x,
      minY: viewBox.y,
      width: viewBox.width,
      height: viewBox.height,
    };
  }
  const width = parseFloat(svg.getAttribute("width")) || 1000;
  const height = parseFloat(svg.getAttribute("height")) || 1000;
  return { minX: 0, minY: 0, width, height };
}

function ensureViewBox(svg) {
  const dims = getSvgDimensions(svg);
  if (!svg.getAttribute("viewBox")) {
    svg.setAttribute("viewBox", `${dims.minX} ${dims.minY} ${dims.width} ${dims.height}`);
  }
  svg.setAttribute("width", String(dims.width));
  svg.setAttribute("height", String(dims.height));
  return dims;
}

function createSvgElement(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}

function installGraphLayer() {
  state.graphLayer?.remove();
  state.graphLayer = createSvgElement("g", { id: "graph-editor-layer" });
  state.edgeLayer = createSvgElement("g", { id: "graph-editor-edges" });
  state.nodeLayer = createSvgElement("g", { id: "graph-editor-nodes" });
  state.graphLayer.append(state.edgeLayer, state.nodeLayer);
  state.svg.appendChild(state.graphLayer);
}

function loadSvgText(svgText, fileName = "uploaded SVG") {
  const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
  const parsedSvg = doc.querySelector("svg");
  const parserError = doc.querySelector("parsererror");
  if (!parsedSvg || parserError) {
    alert("Could not parse SVG file.");
    return;
  }

  els.svgHost.innerHTML = "";
  const importedSvg = document.importNode(parsedSvg, true);
  importedSvg.removeAttribute("id");
  importedSvg.setAttribute("id", "editableSvg");
  importedSvg.style.touchAction = "none";
  els.svgHost.appendChild(importedSvg);

  state.svg = importedSvg;
  state.svgFileName = fileName;
  const dims = ensureViewBox(importedSvg);
  installGraphLayer();
  bindSvgEvents();

  els.emptyState.style.display = "none";
  els.mapName.textContent = fileName;
  els.svgMeta.textContent = `viewBox ${dims.minX} ${dims.minY} ${dims.width} ${dims.height}`;
  render();
}

function bindSvgEvents() {
  state.svg.addEventListener("click", onSvgClick);
  state.svg.addEventListener("mousemove", onSvgMouseMove);
  state.svg.addEventListener("mouseup", stopDrag);
  state.svg.addEventListener("mouseleave", stopDrag);
}

function onSvgMouseMove(event) {
  const point = svgPointFromEvent(event);
  if (!point) return;
  els.cursorCoords.textContent = `x: ${point.x.toFixed(2)}, y: ${point.y.toFixed(2)}`;

  if (!state.drag) return;
  const node = getNode(state.drag.nodeId);
  if (!node) return;
  node.coords = [round(point.x), round(point.y)];
  node.lat = node.coords[1];
  node.lon = node.coords[0];
  updateDistancesForNode(node.id);
  fillNodeForm(node);
  render();
}

function onSvgClick(event) {
  if (!state.svg || event.target.closest("#graph-editor-layer")) return;
  if (state.mode !== "node") return;
  const point = svgPointFromEvent(event);
  if (!point) return;
  addNode(point.x, point.y);
}

function setMode(mode) {
  state.mode = mode;
  state.pendingEdgeNodeId = null;
  els.modeButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === mode));
  const hints = {
    select: "Select, drag, and edit existing nodes.",
    node: "Click the SVG to add a navigation node at exact SVG coordinates.",
    edge: "Click two nodes to create an edge.",
  };
  els.modeHint.textContent = hints[mode];
  render();
}

function round(value) {
  return Math.round(value * 100) / 100;
}

function roleToType(role) {
  return {
    building: "Building",
    entrance: "Entrance",
    waypoint: "Waypoint",
    intersection: "Couloir/Intersection",
  }[role] || "Couloir/Intersection";
}

function makeUniqueId(base) {
  const used = new Set(state.nodes.map((node) => node.id));
  let candidate = base;
  let index = 1;
  while (used.has(candidate)) {
    candidate = `${base}_${index++}`;
  }
  return candidate;
}

function addNode(x, y) {
  const id = makeUniqueId(`N_${state.nodes.length + 1}`);
  const node = {
    id,
    name: id,
    label: id,
    coords: [round(x), round(y)],
    lat: round(y),
    lon: round(x),
    type: "Couloir/Intersection",
    node_role: "waypoint",
    visible_destination: false,
  };
  state.nodes.push(node);
  selectNode(id);
  render();
}

function edgeKey(from, to) {
  return [from, to].sort().join("__");
}

function getNode(id) {
  return state.nodes.find((node) => node.id === id);
}

function distanceBetween(a, b) {
  const dx = a.coords[0] - b.coords[0];
  const dy = a.coords[1] - b.coords[1];
  return round(Math.sqrt(dx * dx + dy * dy));
}

function addEdge(from, to) {
  if (!from || !to || from === to) return;
  if (state.edges.some((edge) => edgeKey(edge.from, edge.to) === edgeKey(from, to))) return;
  const a = getNode(from);
  const b = getNode(to);
  if (!a || !b) return;
  const distance = distanceBetween(a, b);
  state.edges.push({ from, to, distance, weight: distance });
  state.selectedEdgeKey = edgeKey(from, to);
  state.selectedNodeId = null;
  render();
}

function updateDistancesForNode(nodeId) {
  state.edges.forEach((edge) => {
    if (edge.from !== nodeId && edge.to !== nodeId) return;
    const fromNode = getNode(edge.from);
    const toNode = getNode(edge.to);
    if (!fromNode || !toNode) return;
    const distance = distanceBetween(fromNode, toNode);
    edge.distance = distance;
    edge.weight = distance;
  });
}

function selectNode(id) {
  state.selectedNodeId = id;
  state.selectedEdgeKey = null;
  fillNodeForm(getNode(id));
  render();
}

function selectEdge(key) {
  state.selectedEdgeKey = key;
  state.selectedNodeId = null;
  fillNodeForm(null);
  render();
}

function clearSelection() {
  state.selectedNodeId = null;
  state.selectedEdgeKey = null;
  state.pendingEdgeNodeId = null;
  fillNodeForm(null);
  render();
}

function fillNodeForm(node) {
  const disabled = !node;
  [els.nodeId, els.nodeLabel, els.nodeRole, els.visibleDestination, els.nodeX, els.nodeY].forEach((input) => {
    input.disabled = disabled;
  });
  if (!node) {
    els.nodeId.value = "";
    els.nodeLabel.value = "";
    els.nodeRole.value = "waypoint";
    els.visibleDestination.checked = false;
    els.nodeX.value = "";
    els.nodeY.value = "";
    return;
  }
  els.nodeId.value = node.id;
  els.nodeLabel.value = node.label || node.name || node.id;
  els.nodeRole.value = node.node_role || "waypoint";
  els.visibleDestination.checked = Boolean(node.visible_destination);
  els.nodeX.value = node.coords[0];
  els.nodeY.value = node.coords[1];
}

function applyNodeForm(event) {
  event.preventDefault();
  const oldId = state.selectedNodeId;
  const node = getNode(oldId);
  if (!node) return;

  const newId = els.nodeId.value.trim();
  if (!newId) {
    alert("Node id cannot be empty.");
    return;
  }
  if (newId !== oldId && state.nodes.some((item) => item.id === newId)) {
    alert(`Node id "${newId}" already exists.`);
    return;
  }

  node.id = newId;
  node.label = els.nodeLabel.value.trim() || newId;
  node.name = node.label;
  node.node_role = els.nodeRole.value;
  node.type = roleToType(node.node_role);
  node.visible_destination = els.visibleDestination.checked;
  node.coords = [round(Number(els.nodeX.value)), round(Number(els.nodeY.value))];
  node.lat = node.coords[1];
  node.lon = node.coords[0];

  if (newId !== oldId) {
    state.edges.forEach((edge) => {
      if (edge.from === oldId) edge.from = newId;
      if (edge.to === oldId) edge.to = newId;
    });
    state.selectedNodeId = newId;
  }
  updateDistancesForNode(newId);
  render();
}

function deleteSelected() {
  if (state.selectedNodeId) {
    const id = state.selectedNodeId;
    state.nodes = state.nodes.filter((node) => node.id !== id);
    state.edges = state.edges.filter((edge) => edge.from !== id && edge.to !== id);
    clearSelection();
    return;
  }
  if (state.selectedEdgeKey) {
    state.edges = state.edges.filter((edge) => edgeKey(edge.from, edge.to) !== state.selectedEdgeKey);
    clearSelection();
  }
}

function render() {
  if (!state.svg || !state.graphLayer) return;
  renderEdges();
  renderNodes();
  updateStats();
  updateEdgeInfo();
}

function renderEdges() {
  state.edgeLayer.innerHTML = "";
  state.edges.forEach((edge) => {
    const from = getNode(edge.from);
    const to = getNode(edge.to);
    if (!from || !to) return;
    const key = edgeKey(edge.from, edge.to);
    const line = createSvgElement("line", {
      class: `graph-edge ${state.selectedEdgeKey === key ? "selected" : ""}`,
      x1: from.coords[0],
      y1: from.coords[1],
      x2: to.coords[0],
      y2: to.coords[1],
      "data-key": key,
    });
    line.addEventListener("click", (event) => {
      event.stopPropagation();
      selectEdge(key);
    });
    state.edgeLayer.appendChild(line);
  });
}

function renderNodes() {
  state.nodeLayer.innerHTML = "";
  state.nodes.forEach((node) => {
    const group = createSvgElement("g", {
      class: `graph-node ${state.selectedNodeId === node.id ? "selected" : ""} ${state.drag?.nodeId === node.id ? "dragging" : ""}`,
      transform: `translate(${node.coords[0]} ${node.coords[1]})`,
      "data-id": node.id,
    });
    const color = roleColors[node.node_role] || roleColors.waypoint;
    const hit = createSvgElement("circle", { class: "node-hit", r: 12 });
    const circle = createSvgElement("circle", { class: "node-circle", r: 6, fill: color });
    const label = createSvgElement("text", { class: "node-label", x: 9, y: -9 });
    label.textContent = node.label || node.name || node.id;

    group.append(hit, circle, label);
    group.addEventListener("click", (event) => {
      event.stopPropagation();
      if (state.mode === "edge") {
        if (!state.pendingEdgeNodeId) {
          state.pendingEdgeNodeId = node.id;
          selectNode(node.id);
        } else {
          addEdge(state.pendingEdgeNodeId, node.id);
          state.pendingEdgeNodeId = null;
        }
      } else {
        selectNode(node.id);
      }
    });
    group.addEventListener("mousedown", (event) => {
      if (state.mode === "edge") return;
      event.preventDefault();
      event.stopPropagation();
      state.drag = { nodeId: node.id };
      selectNode(node.id);
    });
    state.nodeLayer.appendChild(group);
  });
}

function updateStats() {
  els.nodeCount.textContent = state.nodes.length;
  els.edgeCount.textContent = state.edges.length;
}

function updateEdgeInfo() {
  if (!state.selectedEdgeKey) {
    els.edgeInfo.textContent = "No edge selected.";
    return;
  }
  const edge = state.edges.find((item) => edgeKey(item.from, item.to) === state.selectedEdgeKey);
  if (!edge) {
    els.edgeInfo.textContent = "No edge selected.";
    return;
  }
  els.edgeInfo.innerHTML = `<strong>${edge.from}</strong> -> <strong>${edge.to}</strong><br>distance: ${edge.distance}`;
}

function stopDrag() {
  state.drag = null;
  render();
}

function importGraph(graph) {
  const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
  const edges = Array.isArray(graph.edges) ? graph.edges : [];
  state.nodes = nodes.map((node) => {
    const coords = node.coords || [node.x_svg || 0, node.y_svg || 0];
    const role = node.node_role || node.role || inferRole(node);
    return {
      ...node,
      id: String(node.id),
      name: node.name || node.label || String(node.id),
      label: node.label || node.name || String(node.id),
      coords: [round(Number(coords[0] || 0)), round(Number(coords[1] || 0))],
      lat: round(Number(node.lat ?? coords[1] ?? 0)),
      lon: round(Number(node.lon ?? coords[0] ?? 0)),
      type: node.type || roleToType(role),
      node_role: role,
      visible_destination: Boolean(node.visible_destination),
    };
  });

  const nodeIds = new Set(state.nodes.map((node) => node.id));
  state.edges = edges
    .map((edge) => ({
      from: edge.from ?? edge.source,
      to: edge.to ?? edge.target,
      distance: edge.distance ?? edge.weight ?? edge.distance_meters,
      weight: edge.weight ?? edge.distance ?? edge.distance_meters,
    }))
    .filter((edge) => edge.from && edge.to && nodeIds.has(edge.from) && nodeIds.has(edge.to));

  state.edges.forEach((edge) => {
    const from = getNode(edge.from);
    const to = getNode(edge.to);
    const distance = edge.distance != null ? round(Number(edge.distance)) : distanceBetween(from, to);
    edge.distance = Number.isFinite(distance) ? distance : distanceBetween(from, to);
    edge.weight = edge.distance;
  });
  clearSelection();
  render();
}

function inferRole(node) {
  const text = `${node.id || ""} ${node.name || ""} ${node.type || ""}`.toLowerCase();
  if (text.includes("building") || text.includes("academic") || text.includes("administration") || text.includes("student services")) return "building";
  if (text.includes("entrance") || text.includes("entree") || text.includes("entrée")) return "entrance";
  if (text.includes("intersection")) return "intersection";
  return "waypoint";
}

function exportGraph() {
  const graph = {
    departments: {},
    meta: {
      source: "svg_graph_editor",
      svg_file: state.svgFileName,
      units: "svg_pixels",
    },
    nodes: state.nodes.map((node) => ({
      id: node.id,
      name: node.label || node.name || node.id,
      label: node.label || node.name || node.id,
      coords: [round(node.coords[0]), round(node.coords[1])],
      lat: round(node.coords[1]),
      lon: round(node.coords[0]),
      type: node.type || roleToType(node.node_role),
      node_role: node.node_role,
      visible_destination: Boolean(node.visible_destination),
    })),
    edges: state.edges.map((edge) => {
      const from = getNode(edge.from);
      const to = getNode(edge.to);
      const distance = from && to ? distanceBetween(from, to) : round(Number(edge.distance || 0));
      return {
        from: edge.from,
        to: edge.to,
        distance,
        weight: distance,
      };
    }),
  };

  const blob = new Blob([JSON.stringify(graph, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "campus_graph.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function fitMap() {
  els.canvasWrap.scrollTo({ left: 0, top: 0, behavior: "smooth" });
}

els.svgInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  loadSvgText(await file.text(), file.name);
});

els.loadDefaultSvgBtn.addEventListener("click", async () => {
  try {
    const response = await fetch("../app/static/campus_map_2d1.svg");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    loadSvgText(await response.text(), "campus_map_2d1.svg");
  } catch (error) {
    alert("Could not load default SVG automatically. Use the file picker instead.");
  }
});

els.graphInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    importGraph(JSON.parse(await file.text()));
  } catch (error) {
    alert("Could not parse graph JSON.");
  }
});

els.exportBtn.addEventListener("click", exportGraph);
els.modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
els.nodeForm.addEventListener("submit", applyNodeForm);
els.deleteBtn.addEventListener("click", deleteSelected);
els.clearSelectionBtn.addEventListener("click", clearSelection);
els.fitBtn.addEventListener("click", fitMap);
document.addEventListener("keydown", (event) => {
  if (event.key === "Delete" || event.key === "Backspace") {
    if (["INPUT", "SELECT"].includes(document.activeElement?.tagName)) return;
    deleteSelected();
  }
  if (event.key === "Escape") clearSelection();
});

fillNodeForm(null);
