const statusEl = document.querySelector("#status");
const banner = document.querySelector("#banner");
const bannerText = document.querySelector("#banner-text");
const filmstrip = document.querySelector("#filmstrip");
const photo = document.querySelector("#photo");
const overlay = document.querySelector("#overlay");
const empty = document.querySelector("#empty");
const working = document.querySelector("#working");
const caption = document.querySelector("#caption");
const hits = document.querySelector("#hits");
const countEl = document.querySelector("#count");
const timing = document.querySelector("#timing");
const confInput = document.querySelector("#conf");
const iouInput = document.querySelector("#iou");
const confValue = document.querySelector("#conf-value");
const iouValue = document.querySelector("#iou-value");
const sizeInput = document.querySelector("#imgsz");
const showLabels = document.querySelector("#show-labels");
const fileInput = document.querySelector("#file");

const ns = "http://www.w3.org/2000/svg";
const state = {
  ready: false,
  samples: [],
  sampleId: null,
  upload: null,
  result: null,
  scale: "all",
  request: 0,
  hot: -1,
};

function svg(name, attrs) {
  const node = document.createElementNS(ns, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

async function refreshHealth() {
  const response = await fetch("/api/health");
  const body = await response.json();
  state.ready = body.ready;
  if (body.ready) {
    const millions = (body.parameters / 1e6).toFixed(1);
    statusEl.textContent = `${millions}M parameters · ${body.device} · ${body.classes.length} classes`;
    banner.hidden = true;
  } else {
    statusEl.textContent = "Detector is waiting for weights";
    banner.hidden = false;
    bannerText.textContent = body.error || "Official weights are not loaded.";
  }
}

function renderFilm() {
  filmstrip.replaceChildren();
  for (const sample of state.samples) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.id = sample.id;
    if (sample.id === state.sampleId) button.classList.add("on");
    const image = document.createElement("img");
    image.src = sample.url;
    image.alt = "";
    const label = document.createElement("span");
    label.textContent = sample.name;
    const kind = document.createElement("small");
    kind.textContent = sample.kind;
    label.appendChild(kind);
    button.append(image, label);
    button.addEventListener("click", () => selectSample(sample.id));
    filmstrip.appendChild(button);
  }
}

async function loadSamples() {
  const response = await fetch("/api/samples");
  const body = await response.json();
  state.samples = body.samples;
  renderFilm();
  if (state.samples.length && !state.sampleId && !state.upload) {
    selectSample(state.samples[0].id);
  }
}

function selectSample(id) {
  state.sampleId = id;
  state.upload = null;
  fileInput.value = "";
  const sample = state.samples.find((item) => item.id === id);
  showImage(sample.url, sample.name);
  renderFilm();
  runDetection();
}

function showImage(url, name) {
  empty.hidden = true;
  photo.hidden = false;
  overlay.hidden = false;
  photo.alt = name;
  photo.src = url;
  caption.textContent = name;
  state.result = null;
  draw();
}

function visibleDetections() {
  if (!state.result) return [];
  return state.result.detections.filter((det) => state.scale === "all" || det.scale === state.scale);
}

function draw() {
  overlay.replaceChildren();
  const result = state.result;
  if (!result) return;
  const width = result.image.width;
  const height = result.image.height;
  overlay.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const font = Math.max(14, width / 42);

  if (showLabels.checked) {
    for (const label of result.labels) {
      const rect = svg("rect", {
        x: label.x1,
        y: label.y1,
        width: Math.max(0, label.x2 - label.x1),
        height: Math.max(0, label.y2 - label.y1),
        class: "shape gt",
      });
      overlay.appendChild(rect);
    }
  }

  visibleDetections().forEach((det, index) => {
    const group = svg("g", { "data-index": index });
    const boxWidth = Math.max(0, det.x2 - det.x1);
    const boxHeight = Math.max(0, det.y2 - det.y1);
    const shape = svg("rect", {
      x: det.x1,
      y: det.y1,
      width: boxWidth,
      height: boxHeight,
      class: index === state.hot ? "shape hot-shape" : "shape",
      stroke: det.color,
    });
    const text = `${det.class_name} ${det.confidence.toFixed(2)}`;
    const labelWidth = text.length * font * 0.56 + font * 0.8;
    const labelHeight = font * 1.35;
    let labelY = det.y1 - labelHeight;
    if (labelY < 0) labelY = det.y1;
    const plate = svg("rect", {
      x: det.x1,
      y: labelY,
      width: labelWidth,
      height: labelHeight,
      fill: det.color,
    });
    const type = svg("text", {
      x: det.x1 + font * 0.35,
      y: labelY + font,
      class: "tag",
      fill: "#1a2010",
      "font-size": font,
    });
    type.textContent = text;
    group.append(shape, plate, type);
    group.addEventListener("pointerenter", () => highlight(index));
    group.addEventListener("pointerleave", () => highlight(-1));
    overlay.appendChild(group);
  });
  renderHits();
}

function renderHits() {
  const detections = visibleDetections();
  hits.replaceChildren();
  countEl.classList.remove("error");
  if (!state.result) {
    countEl.textContent = state.ready ? "No detections yet" : "Weights required";
    timing.textContent = "";
    return;
  }
  const noun = detections.length === 1 ? "detection" : "detections";
  countEl.textContent = `${detections.length} ${noun}`;
  const scales = { large: 0, medium: 0, small: 0 };
  for (const det of state.result.detections) scales[det.scale] += 1;
  timing.textContent = `${state.result.elapsed_ms} ms · input ${state.result.img_size} · large ${scales.large} · medium ${scales.medium} · small ${scales.small}`;
  detections.forEach((det, index) => {
    const item = document.createElement("li");
    if (index === state.hot) item.classList.add("hot");
    const swatch = document.createElement("i");
    swatch.className = "swatch";
    swatch.style.background = det.color;
    const text = document.createElement("span");
    const name = document.createElement("strong");
    name.textContent = det.class_name;
    const meta = document.createElement("em");
    meta.textContent = `${det.scale} head`;
    text.append(name, meta);
    const score = document.createElement("b");
    score.textContent = det.confidence.toFixed(2);
    item.append(swatch, text, score);
    item.addEventListener("pointerenter", () => highlight(index));
    item.addEventListener("pointerleave", () => highlight(-1));
    hits.appendChild(item);
  });
  if (!detections.length) {
    const item = document.createElement("li");
    item.textContent = "Nothing cleared this confidence threshold.";
    hits.appendChild(item);
  }
}

function highlight(index) {
  if (state.hot === index) return;
  state.hot = index;
  overlay.querySelectorAll("g[data-index]").forEach((group) => {
    const shape = group.querySelector("rect.shape");
    if (!shape) return;
    const hot = Number(group.getAttribute("data-index")) === index;
    shape.setAttribute("class", hot ? "shape hot-shape" : "shape");
  });
  hits.querySelectorAll("li").forEach((item, itemIndex) => {
    item.classList.toggle("hot", itemIndex === index);
  });
}

async function runDetection() {
  if (!state.ready) return;
  const token = ++state.request;
  working.hidden = false;
  const form = new FormData();
  form.set("conf", confInput.value);
  form.set("iou", iouInput.value);
  form.set("imgsz", sizeInput.value);
  if (state.upload) form.set("file", state.upload);
  else if (state.sampleId) form.set("sample", state.sampleId);
  else {
    working.hidden = true;
    return;
  }
  try {
    const response = await fetch("/api/detect", { method: "POST", body: form });
    const body = await response.json();
    if (token !== state.request) return;
    if (!response.ok) throw new Error(body.detail || "Detection failed");
    state.result = body;
    state.hot = -1;
    caption.textContent = `${body.source} · ${body.image.width}×${body.image.height}`;
    draw();
  } catch (error) {
    if (token !== state.request) return;
    countEl.textContent = "Detection failed";
    countEl.classList.add("error");
    timing.textContent = error.message;
  } finally {
    if (token === state.request) working.hidden = true;
  }
}

let timer = 0;
function scheduleDetection() {
  confValue.textContent = Number(confInput.value).toFixed(2);
  iouValue.textContent = Number(iouInput.value).toFixed(2);
  window.clearTimeout(timer);
  timer = window.setTimeout(runDetection, 180);
}

confInput.addEventListener("input", scheduleDetection);
iouInput.addEventListener("input", scheduleDetection);
sizeInput.addEventListener("change", runDetection);
showLabels.addEventListener("change", draw);

document.querySelectorAll("#scales button").forEach((button) => {
  button.addEventListener("click", () => {
    state.scale = button.dataset.scale;
    document.querySelectorAll("#scales button").forEach((item) => item.classList.toggle("on", item === button));
    state.hot = -1;
    draw();
  });
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  state.upload = file;
  state.sampleId = null;
  renderFilm();
  showImage(URL.createObjectURL(file), file.name);
  runDetection();
});

document.querySelector("#download").addEventListener("click", async () => {
  const button = document.querySelector("#download");
  button.disabled = true;
  bannerText.textContent = "Downloading the official weights…";
  try {
    const response = await fetch("/api/weights/download", { method: "POST" });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Download failed");
    await refreshHealth();
    runDetection();
  } catch (error) {
    bannerText.textContent = error.message;
    button.disabled = false;
  }
});

document.addEventListener("keydown", (event) => {
  if (!state.samples.length || state.upload) return;
  if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
  const index = state.samples.findIndex((sample) => sample.id === state.sampleId);
  const next = event.key === "ArrowRight" ? index + 1 : index - 1;
  const sample = state.samples[(next + state.samples.length) % state.samples.length];
  selectSample(sample.id);
});

const frame = document.querySelector("#frame");
frame.addEventListener("dragover", (event) => event.preventDefault());
frame.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (!file) return;
  state.upload = file;
  state.sampleId = null;
  renderFilm();
  showImage(URL.createObjectURL(file), file.name);
  runDetection();
});

refreshHealth().then(loadSamples);
