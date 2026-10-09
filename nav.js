const DT = 0.025;
const TAU = 2 * Math.PI;
const ZONES = ["Brehat", "Cardinales", "Croisic", "Crouesty", "Lorient", "Paimpol", "Quiberon", "WR"];
const LIGHT_COLORS = { W: "#f4f1df", R: "#f16e66", G: "#7ed7a0", Y: "#f5d66f", B: "#17272b" };
const TILE_CACHE = new Map();

const app = document.querySelector("#app");
app.innerHTML = `
  <header class="toolbar">
    <div class="brand"><span class="brand-mark">N</span><div><strong>Balisage de nuit</strong><small>Simulateur de navigation</small></div></div>
    <label class="field">Zone<select id="zone"></select></label>
    <label class="field pattern-field">Motif lumineux<input id="pattern" placeholder="ex. Fl(2).6s" autocomplete="off"></label>
    <label class="field numeric-field">Dérive <small>nd</small><input id="drift" type="number" value="0" step="0.1"></label>
    <label class="field numeric-field">Réduction de vitesse <small>MN</small><input id="obstacle" type="number" value="5" min="0.1" step="0.5"></label>
    <label class="check-field"><input id="reflection" type="checkbox"> Réflexion</label>
    <label class="field map-opacity-field">Opacité carte<input id="map-opacity" type="range" min="0" max="100" value="65" aria-label="Opacité de la carte"><span class="map-opacity-ends"><small>0 %</small><small>100 %</small></span></label>
    <button id="restart" type="button" title="Redémarrer la simulation">↻ <span>Redémarrer</span></button>
  </header>
  <main class="stage"><canvas id="scene" aria-label="Simulation de navigation de nuit"></canvas><div id="light-tooltip" class="light-tooltip" role="tooltip" hidden></div></main>
  <footer class="statusbar">
    <div class="readout"><i></i><span id="status">Chargement de la zone…</span></div>
    <div class="telemetry"><span><small>CAP</small><b id="heading">000°</b></span><span><small>FEU LE PLUS PROCHE</small><b id="distance">-- MN</b></span></div>
    <div class="keys">ZQSD : déplacement · A/E : latéral · R : vue arrière · Survolez un feu pour afficher son rythme lumineux</div>
    <small class="copyright">© 2024 Olivier Kermorgant</small>
  </footer>`;

const canvas = document.querySelector("#scene");
const ctx = canvas.getContext("2d");
const lightTooltip = document.querySelector("#light-tooltip");
const $ = (selector) => document.querySelector(selector);
const ui = { status: $("#status"), heading: $("#heading"), distance: $("#distance") };
const controls = { zone: $("#zone"), pattern: $("#pattern"), drift: $("#drift"), obstacle: $("#obstacle"), reflection: $("#reflection"), mapOpacity: $("#map-opacity") };
const boatSprite = new Image();
boatSprite.src = "./images/bto.png";
let sim = null;
let running = true;
let lastFrame = 0;
let frameId = 0;
let accumulator = 0;
let pixelWidth = 1000;
let pixelHeight = 750;

for (const zone of ZONES) {
  const option = document.createElement("option");
  option.value = option.textContent = zone;
  controls.zone.append(option);
}
const requestedZone = new URLSearchParams(location.search).get("zone");
controls.zone.value = ZONES.includes(requestedZone) ? requestedZone : "Cardinales";

function parseYaml(source) {
  const data = {};
  let nested = null;
  let sectionKey = null;
  for (const raw of source.split(/\r?\n/)) {
    const line = raw.replace(/\s+#.*$/, "");
    if (!line.trim() || line.trimStart().startsWith("#")) continue;
    const indent = line.length - line.trimStart().length;
    if (indent && line.trimStart().startsWith("- ")) {
      const section = data[sectionKey];
      if (Array.isArray(section)) section.push(line.trimStart().slice(2).trim());
      else if (section && typeof section === "object" && Object.keys(section).length === 0) data[sectionKey] = [line.trimStart().slice(2).trim()];
      nested = null;
      continue;
    }
    const match = line.trim().match(/^([^:]+):(?:\s*(.*))?$/);
    if (!match) continue;
    const [, key, value = ""] = match;
    if (!indent) {
      sectionKey = key.trim();
      if (value.trim()) { data[sectionKey] = value.trim(); nested = null; }
      else { data[sectionKey] = {}; nested = data[sectionKey]; }
    } else if (nested) nested[key.trim()] = value.trim();
  }
  return data;
}

function coordinate(text) {
  return text.split(",").map((part) => {
    const direction = part.trim().match(/([NSEW])\.?$/i)?.[1].toUpperCase();
    const pieces = [...part.matchAll(/[+-]?\d+(?:[.,]\d+)?/g)].map((match) => Number(match[0].replace(",", ".")));
    const degrees = pieces.reduce((sum, number, index) => sum + number / 60 ** index, 0);
    return direction === "S" || direction === "W" ? -degrees : degrees;
  });
}

function wrap(angle, positive = false) {
  if (positive) return ((angle % TAU) + TAU) % TAU;
  return ((angle + Math.PI) % TAU + TAU) % TAU - Math.PI;
}

function mixColor(from, to, amount) {
  const channels = [1, 3, 5].map((offset) => {
    const start = Number.parseInt(from.slice(offset, offset + 2), 16);
    const end = Number.parseInt(to.slice(offset, offset + 2), 16);
    return Math.round(start + (end - start) * amount);
  });
  return `rgb(${channels.join(", ")})`;
}

function phaseOffset(pattern, period, cycleLength) {
  let hash = 2166136261;
  for (const character of `${pattern}\0${period}`) {
    hash = Math.imul(hash ^ character.charCodeAt(0), 16777619);
  }
  return (hash >>> 0) % cycleLength;
}

function angle(from, to) { return Math.atan2(to[1] - from[1], to[0] - from[0]); }

function parsePattern(source) {
  let pattern = source;
  if (pattern.startsWith("N")) pattern = "Q";
  else if (pattern.startsWith("E")) pattern = "Q(3).15s";
  else if (pattern.startsWith("S")) pattern = "Q(6)+LFl.15s";
  else if (pattern.startsWith("W")) pattern = "Q(9).15s";
  const meta = { m: 3, s: 4, M: 5 };
  const dot = pattern.indexOf(".");
  if (dot >= 0) {
    const details = pattern.slice(dot + 1);
    pattern = pattern.slice(0, dot);
    for (const match of details.matchAll(/([\d.,]+)([smM])/g)) meta[match[2]] = Number(match[1].replace(",", "."));
  }
  let colors = [...pattern].filter((char) => "WRGYB".includes(char)).join("") || "W";
  pattern = pattern.replace(colors, "").replace(/^\.+|\.+$/g, "");
  const phasePattern = pattern;
  if (colors.length === 2) { const other = [...colors].find((color) => color !== "W"); colors = `${other}W${other}`; }
  else if (colors.length === 3) colors = "RWG";
  if (colors === "B") pattern = "Iso";
  return { pattern, phasePattern, colors, meta };
}

function lightFor(label, shape) {
  let inverted = label.startsWith("Oc");
  const { pattern, phasePattern, colors, meta } = parsePattern(inverted ? label.replace("Oc", "Fl") : label);
  const light = { pattern: label, c: null, colors, height: meta.m, range: meta.M, on: Array(Math.max(1, Math.floor(meta.s / DT))).fill(false), cur: 0, sectors: [], d: 100, a: 0 };
  light.cur = !phasePattern.includes("(") && phasePattern.includes("Q") ? -1 : phaseOffset(phasePattern, meta.s, light.on.length);
  if (shape && typeof shape === "object") {
    light.c = coordinate(shape.pos);
    const points = Object.entries(shape).filter(([key]) => key !== "pos").map(([key, value]) => {
      const point = coordinate(value);
      return { color: key[0], angle: angle(light.c, point), point };
    }).sort((a, b) => a.angle - b.angle);
    light.sectors = points.map((point, index) => ({ color: point.color, start: wrap(point.angle, true), span: wrap(points[(index + 1) % points.length].angle - point.angle, true), range: Math.max(3, Math.min(10, meta.M)), point: point.point }));
  } else light.c = coordinate(shape);
  const durations = { Q: 0.5, VQ: 0.25, Fl: 1, LFl: 2.5 };
  const setLit = (start, duration) => {
    for (let i = Math.floor(start / DT); i < Math.min(light.on.length, Math.floor((start + duration) / DT)); i++) light.on[i] = true;
  };
  if (pattern === "Iso") setLit(0, meta.s / 2);
  else {
    let type = pattern, count = 1, bonus = 0, other = "";
    if (pattern.includes("(")) {
      const [head, tail = ""] = pattern.split(")");
      type = head.split("(")[0]; other = tail.replace(/^\+/, "");
      [count, bonus = 0] = head.split("(")[1].split("+").map(Number);
    } else if (pattern.includes("Q")) count = Math.floor(meta.s / (durations[type] || 0.5));
    const duration = durations[type] || 1;
    let end = count * 2 * duration;
    for (let i = 0; i < Math.min(light.on.length, Math.floor(end / DT)); i++) light.on[i] = i % (2 * Math.floor(duration / DT)) < Math.floor(duration / DT);
    if (bonus) { setLit(end + 2, duration); end += 2 + duration; }
    if (other && durations[other]) setLit(end, durations[other]);
  }
  if (inverted) light.on = light.on.map((value) => !value);
  return light;
}

function buildSimulation(source) {
  const config = parseYaml(source);
  const override = controls.pattern.value.trim();
  if (override && config["Fl.3s"]) { config[override] = config["Fl.3s"]; delete config["Fl.3s"]; }
  const start = coordinate(config.start);
  const lights = Object.entries(config).filter(([key]) => key !== "start").flatMap(([key, geometry]) => (Array.isArray(geometry) ? geometry : [geometry]).map((shape) => lightFor(key, shape)));
  if (!lights.length) throw new Error("Cette zone ne contient aucun feu.");
  const all = [start, ...lights.flatMap((light) => [light.c, ...light.sectors.map((sector) => sector.point)])];
  const minimum = [0, 1].map((axis) => Math.min(...all.map((point) => point[axis])));
  const maximum = [0, 1].map((axis) => Math.max(...all.map((point) => point[axis])));
  const center = minimum.map((value, axis) => (value + maximum[axis]) / 2);
  const lonNm = 60 * Math.cos(center[0] * Math.PI / 180);
  const toLocal = (point) => [(point[0] - center[0]) * 60, (point[1] - center[1]) * lonNm];
  const points = all.map(toLocal);
  const spanX = Math.max(0.5, Math.max(...points.map((point) => point[1])) - Math.min(...points.map((point) => point[1])));
  const spanY = Math.max(0.5, Math.max(...points.map((point) => point[0])) - Math.min(...points.map((point) => point[0])));
  const scale = Math.min(1000 / spanX, 500 / spanY) / 1.1;
  const scaleX = scale, scaleY = scale;
  const chartMarkers = all.map((point) => {
    const local = toLocal(point);
    return [500 + local[1] * scaleX, 250 - local[0] * scaleY];
  });
  const compass = findCompassPosition(chartMarkers);
  const mapTiles = createMapTiles(center, lonNm, scaleX, scaleY);
  const localStart = toLocal(start);
  const lightCenters = lights.map((light) => toLocal(light.c)).sort((a, b) => a[0] - b[0]);
  const midpoint = lightCenters[Math.floor(lightCenters.length / 2)];
  const boat = { c: localStart, theta: Math.atan2(midpoint[1] - localStart[1], midpoint[0] - localStart[0]), vx: 0, vy: 0, turn: 0, forward: 0, drift: Number(controls.drift.value) || 0, factor: 1, obstacle: Number(controls.obstacle.value) || 5, nearest: 100, targetSpeed: 200 / Math.min(scaleX, scaleY) };
  const moon = { bearing: wrap(boat.theta + (Math.random() - 0.5) * Math.PI * 0.75, true), elevation: (35 + Math.random() * 30) * Math.PI / 180 };
  return { lights, boat, center, lonNm, toLocal, scaleX, scaleY, compass, mapTiles, moon };
}

function distance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

function adaptSpeed() {
  const boat = sim.boat;
  let factor = 1;
  boat.nearest = 100;
  for (const light of sim.lights) {
    const position = sim.toLocal(light.c);
    light.d = distance(position, boat.c);
    light.a = wrap(Math.atan2(boat.c[1] - position[1], boat.c[0] - position[0]), true);
    boat.nearest = Math.min(boat.nearest, light.d);
    if (light.d < boat.obstacle) {
      factor = Math.min(factor, Math.sqrt(light.d / boat.obstacle));
      if (light.sectors.length) {
        const edge = light.sectors.reduce((best, sector) => Math.abs(wrap(light.a - sector.start)) < Math.abs(wrap(light.a - best.start)) ? sector : best);
        factor = Math.min(factor, Math.abs(wrap(light.a - edge.start)) / (2 * Math.PI / 180));
      }
    }
  }
  boat.factor = 0.5 * (boat.factor + Math.max(factor, 0.1));
}

function keyHandler(event, down) {
  if (!sim) return;
  const boat = sim.boat;
  const code = event.code;
  if (down) {
    if (["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE", "KeyR"].includes(code)) event.preventDefault();
    if (code === "KeyW") boat.vx = boat.targetSpeed;
    else if (code === "KeyS") boat.vx = -boat.targetSpeed;
    else if (code === "KeyA") boat.turn = -2;
    else if (code === "KeyD") boat.turn = 2;
    else if (code === "KeyQ") boat.vy = -boat.targetSpeed / 10;
    else if (code === "KeyE") boat.vy = boat.targetSpeed / 10;
    else if (code === "KeyR" && !event.repeat) boat.forward = Math.PI - boat.forward;
  } else if (code === "KeyW" || code === "KeyS") boat.vx = 0;
  else if (code === "KeyA" || code === "KeyD") boat.turn = 0;
  else if (code === "KeyQ" || code === "KeyE") boat.vy = 0;
}

function moveBoat(step) {
  const boat = sim.boat, c = Math.cos(boat.theta), s = Math.sin(boat.theta);
  boat.c[0] += (c * boat.vx - s * boat.vy) * boat.factor * step;
  boat.c[1] += (s * boat.vx + c * boat.vy) * boat.factor * step;
  boat.c[0] -= boat.drift * boat.factor * step;
  boat.theta += boat.turn * Math.sqrt(boat.factor) * step;
}

function mapPoint(point) {
  const local = sim.toLocal(point);
  return [500 + local[1] * sim.scaleX, 250 - local[0] * sim.scaleY];
}

function updateLightTooltip(event) {
  if (!sim) return;
  const bounds = canvas.getBoundingClientRect();
  const x = (event.clientX - bounds.left) * 1000 / bounds.width;
  const y = (event.clientY - bounds.top) * 750 / bounds.height;
  if (x < 0 || x > 1000 || y < 0 || y > 750) return hideLightTooltip();

  let nearest = null;
  if (y <= 500) {
    let nearestPointerDistance = 12;
    for (const light of sim.lights) {
      const [lightX, lightY] = mapPoint(light.c);
      const pointerDistance = Math.hypot(x - lightX, y - lightY);
      if (pointerDistance < nearestPointerDistance) { nearest = light; nearestPointerDistance = pointerDistance; }
    }
  } else {
    const horizon = 687.5;
    let nearestLightDistance = Infinity;
    for (const light of sim.lights) {
      const relative = wrap(light.a + Math.PI - (sim.boat.theta + sim.boat.forward));
      if (Math.abs(relative) > Math.PI / 2) continue;
      const lightX = ((relative * 1000 / Math.PI + 500) % 1000 + 1000) % 1000;
      const lightY = horizon - Math.max(2, Math.min(30 * Math.log((light.height - 2) / Math.max(light.d, 0.001)), 80));
      const radius = Math.max(2, Math.min(6, Math.floor(2 * light.height / Math.max(light.d, 0.001))));
      const pointerDistance = Math.hypot(x - lightX, y - lightY);
      const poleHalfWidth = radius * (0.6 + 0.9 * (y - lightY) / Math.max(horizon - lightY, 1)) + 4;
      const overBeacon = pointerDistance <= Math.max(10, radius + 4);
      const overPole = y >= lightY && y <= horizon && Math.abs(x - lightX) <= poleHalfWidth;
      if ((overBeacon || overPole) && light.d < nearestLightDistance) {
        nearest = light;
        nearestLightDistance = light.d;
      }
    }
  }
  if (!nearest) return hideLightTooltip();

  lightTooltip.textContent = nearest.pattern;
  lightTooltip.hidden = false;
  const stageBounds = canvas.parentElement.getBoundingClientRect();
  const pointerX = event.clientX - stageBounds.left, pointerY = event.clientY - stageBounds.top;
  const left = pointerX - lightTooltip.offsetWidth - 12;
  const top = pointerY - lightTooltip.offsetHeight - 12;
  lightTooltip.style.left = `${left >= 8 ? left : Math.min(pointerX + 12, stageBounds.width - lightTooltip.offsetWidth - 8)}px`;
  lightTooltip.style.top = `${Math.max(8, Math.min(top, stageBounds.height - lightTooltip.offsetHeight - 8))}px`;
  canvas.style.cursor = "help";
}

function hideLightTooltip() {
  lightTooltip.hidden = true;
  canvas.style.cursor = "default";
}

function createMapTiles(center, lonNm, scaleX, scaleY) {
  const latitudeSpan = 250 / scaleY / 60;
  const longitudeSpan = 500 / scaleX / lonNm;
  const zoom = Math.max(8, Math.min(16, Math.round(Math.log2(156543.03392 * Math.cos(center[0] * Math.PI / 180) * scaleX / 1852))));
  const northWest = tileAt(center[0] + latitudeSpan, center[1] - longitudeSpan, zoom);
  const southEast = tileAt(center[0] - latitudeSpan, center[1] + longitudeSpan, zoom);
  const worldSize = 2 ** zoom;
  const tiles = [];
  for (let x = northWest.x; x <= southEast.x; x++) {
    for (let y = northWest.y; y <= southEast.y; y++) {
      if (y < 0 || y >= worldSize) continue;
      const wrappedX = ((x % worldSize) + worldSize) % worldSize;
      const key = `${zoom}/${wrappedX}/${y}`;
      let image = TILE_CACHE.get(key);
      if (!image) {
        image = new Image(); image.crossOrigin = "anonymous";
        image.src = `https://tile.openstreetmap.org/${key}.png`;
        TILE_CACHE.set(key, image);
      }
      tiles.push({ image, northWest: tileCorner(x, y, zoom), southEast: tileCorner(x + 1, y + 1, zoom) });
    }
  }
  return tiles;
}

function tileAt(latitude, longitude, zoom) {
  const worldSize = 2 ** zoom;
  const lat = Math.max(-85.05112878, Math.min(85.05112878, latitude)) * Math.PI / 180;
  return {
    x: Math.floor((longitude + 180) / 360 * worldSize),
    y: Math.floor((1 - Math.asinh(Math.tan(lat)) / Math.PI) / 2 * worldSize)
  };
}

function tileCorner(x, y, zoom) {
  const worldSize = 2 ** zoom;
  const latitude = Math.atan(Math.sinh(Math.PI * (1 - 2 * y / worldSize))) * 180 / Math.PI;
  const longitude = x / worldSize * 360 - 180;
  return [latitude, longitude];
}

function findCompassPosition(markers) {
  let best = { x: 48, y: 48, clearance: -Infinity };
  for (let y = 44; y <= 456; y += 20) {
    for (let x = 44; x <= 956; x += 20) {
      let clearance = Infinity;
      for (const [markerX, markerY] of markers) clearance = Math.min(clearance, Math.hypot(x - markerX, y - markerY));
      if (clearance > best.clearance) best = { x, y, clearance };
    }
  }
  return best;
}

function drawCompassRose(position) {
  ctx.save(); ctx.translate(position.x, position.y);
  ctx.beginPath(); ctx.arc(0, 0, 35, 0, TAU);
  ctx.fillStyle = "rgba(9, 28, 32, 0.86)"; ctx.fill();
  ctx.strokeStyle = "#435d5b"; ctx.lineWidth = 1; ctx.stroke();
  for (let index = 0; index < 8; index++) {
    const angle = -Math.PI / 2 + index * Math.PI / 4;
    const length = index % 2 === 0 ? 21 : 14;
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(Math.cos(angle) * length, Math.sin(angle) * length);
    ctx.strokeStyle = index === 0 ? "#e2b970" : "#95a29c";
    ctx.lineWidth = index % 2 === 0 ? 2 : 1; ctx.stroke();
  }
  ctx.fillStyle = "#e2b970"; ctx.font = "bold 9px monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("N", 0, -27);
  ctx.fillStyle = "#c1cbc2"; ctx.font = "8px monospace";
  ctx.fillText("E", 27, 0); ctx.fillText("S", 0, 28); ctx.fillText("O", -27, 0);
  ctx.restore();
}

function drawHalfMoon(moon, boat, horizon) {
  const relative = wrap(moon.bearing - (boat.theta + boat.forward));
  if (Math.abs(relative) > Math.PI / 2) return;
  const radius = 15;
  const x = 500 + relative * 1000 / Math.PI;
  const y = horizon - 20 - 145 * Math.sin(moon.elevation);
  ctx.save();
  ctx.shadowBlur = 0;
  ctx.fillStyle = "#efe4c5"; ctx.beginPath(); ctx.arc(x, y, radius, 0, TAU); ctx.fill();
  ctx.shadowBlur = 0;
  ctx.beginPath(); ctx.arc(x, y, radius, 0, TAU); ctx.clip();
  ctx.fillStyle = "#0b242b"; ctx.beginPath(); ctx.arc(x + 8, y, radius * 0.94, 0, TAU); ctx.fill();
  ctx.restore();
}

function drawSector(light, sector) {
  const center = mapPoint(light.c), count = Math.max(2, Math.floor(Math.abs(sector.span) * 180 / Math.PI));
  ctx.beginPath(); ctx.moveTo(...center);
  for (let i = 0; i <= count; i++) {
    const a = sector.start + sector.span * i / count;
    const point = [light.c[0] + sector.range * Math.cos(a) / 60, light.c[1] + sector.range * Math.sin(a) / sim.lonNm];
    ctx.lineTo(...mapPoint(point));
  }
  ctx.closePath(); ctx.fillStyle = `${LIGHT_COLORS[sector.color] || LIGHT_COLORS.W}66`; ctx.fill();
  if (sector.color !== "W") {
    for (const a of [sector.start, sector.start + sector.span]) {
      ctx.beginPath(); ctx.moveTo(...center);
      ctx.lineTo(...mapPoint([light.c[0] + sector.range * Math.cos(a) / 60, light.c[1] + sector.range * Math.sin(a) / sim.lonNm]));
      ctx.strokeStyle = `${LIGHT_COLORS[sector.color]}ff`; ctx.stroke();
    }
  }
}

function drawMap() {
  const mapOpacity = Number(controls.mapOpacity.value) / 100;
  ctx.fillStyle = "#091c20"; ctx.fillRect(0, 0, 1000, 500);
  for (const tile of sim.mapTiles) {
    if (!tile.image.complete || !tile.image.naturalWidth) continue;
    const [left, top] = mapPoint(tile.northWest), [right, bottom] = mapPoint(tile.southEast);
    ctx.globalAlpha = mapOpacity;
    ctx.drawImage(tile.image, left, top, right - left, bottom - top);
    ctx.globalAlpha = 1;
  }
  ctx.fillStyle = "rgba(5, 16, 19, 0.35)"; ctx.fillRect(0, 0, 1000, 500);
  for (const light of sim.lights) for (const sector of light.sectors) drawSector(light, sector);
  for (const light of sim.lights) {
    const [x, y] = mapPoint(light.c); ctx.beginPath(); ctx.arc(x, y, light.sectors.length ? 4 : 3, 0, TAU);
    ctx.fillStyle = light.sectors.length ? "#f6df9b" : LIGHT_COLORS[light.colors[0]] || LIGHT_COLORS.W; ctx.fill();
  }
  const boatGeo = [sim.center[0] + sim.boat.c[0] / 60, sim.center[1] + sim.boat.c[1] / sim.lonNm];
  ctx.save(); ctx.translate(...mapPoint(boatGeo)); ctx.rotate(sim.boat.theta);
  if (boatSprite.complete && boatSprite.naturalWidth) ctx.drawImage(boatSprite, -11, -23, 22, 46);
  else { ctx.fillStyle = "#f4f1df"; ctx.beginPath(); ctx.moveTo(0, -13); ctx.lineTo(7, 11); ctx.lineTo(0, 8); ctx.lineTo(-7, 11); ctx.closePath(); ctx.fill(); }
  ctx.restore();
  drawCompassRose(sim.compass);
  ctx.fillStyle = "rgba(9, 28, 32, 0.78)"; ctx.fillRect(766, 480, 226, 16);
  ctx.fillStyle = "#d2d9cd"; ctx.font = "9px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "alphabetic";
  ctx.fillText("© OpenStreetMap contributors", 988, 492);
}

function drawSailorLamp() {
  ctx.save(); ctx.translate(34, 538);
  ctx.beginPath(); ctx.moveTo(8, -14); ctx.lineTo(38, -22); ctx.lineTo(38, -4); ctx.closePath();
  ctx.fillStyle = "rgba(226, 185, 112, 0.18)"; ctx.fill();
  ctx.fillStyle = "#52766f"; ctx.beginPath();
  ctx.moveTo(-12, 18); ctx.lineTo(-10, 7); ctx.lineTo(-5, 2); ctx.lineTo(5, 2); ctx.lineTo(11, 8); ctx.lineTo(12, 18); ctx.closePath(); ctx.fill();
  ctx.fillStyle = "#d8c9a8"; ctx.beginPath(); ctx.arc(0, -10, 7, 0, TAU); ctx.fill();
  ctx.beginPath(); ctx.moveTo(4, -13); ctx.lineTo(10, -10); ctx.lineTo(4, -7); ctx.closePath(); ctx.fill();
  ctx.fillStyle = "#334b49"; ctx.beginPath(); ctx.ellipse(-1, -16, 8, 3, 0, Math.PI, TAU); ctx.fill();
  ctx.strokeStyle = "#e2b970"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(-7, -14); ctx.lineTo(7, -14); ctx.stroke();
  ctx.fillStyle = "#f4e3b2"; ctx.fillRect(6, -16, 4, 4);
  ctx.restore();
}

function drawHorizon() {
  const top = 500, horizon = top + 187.5;
  ctx.fillStyle = "#0b242b"; ctx.fillRect(0, top, 1000, 250);
  ctx.fillStyle = "#10414b"; ctx.fillRect(0, horizon, 1000, 63);
  ctx.fillStyle = "#091c20"; ctx.fillRect(0, horizon + 63, 1000, 63);
  ctx.strokeStyle = "#6da7a5"; ctx.beginPath(); ctx.moveTo(0, horizon); ctx.lineTo(1000, horizon); ctx.stroke();
  drawSailorLamp();
  drawHalfMoon(sim.moon, sim.boat, horizon);
  const heading = Math.floor(((sim.boat.theta % TAU) + TAU) % TAU * 180 / Math.PI);
  ctx.fillStyle = "#f4f1df"; ctx.font = "600 16px monospace";
  ctx.fillText(`${String(heading).padStart(3, "0")}°`, 914, top + 232);
  for (const light of [...sim.lights].sort((a, b) => b.d - a.d)) {
    const relative = wrap(light.a + Math.PI - (sim.boat.theta + sim.boat.forward));
    if (Math.abs(relative) > Math.PI / 2) continue;
    const x = ((relative * 1000 / Math.PI + 500) % 1000 + 1000) % 1000;
    const y = horizon - Math.max(2, Math.min(30 * Math.log((light.height - 2) / Math.max(light.d, 0.001)), 80));
    const radius = Math.max(2, Math.min(6, Math.floor(2 * light.height / Math.max(light.d, 0.001))));
    const lit = light.on[light.cur] && light.d <= light.range;
    ctx.beginPath(); ctx.moveTo(x - radius * 0.6, y); ctx.lineTo(x + radius * 0.6, y); ctx.lineTo(x + radius * 1.5, horizon); ctx.lineTo(x - radius * 1.5, horizon); ctx.closePath();
    ctx.fillStyle = lit ? "#18201e" : "#020809"; ctx.fill();
    if (!lit) continue;
    let color = LIGHT_COLORS[light.colors[0]] || LIGHT_COLORS.W;
    if (light.sectors.length) {
      let nearestIndex = 0;
      for (let index = 1; index < light.sectors.length; index++) {
        if (Math.abs(wrap(light.a - light.sectors[index].start)) < Math.abs(wrap(light.a - light.sectors[nearestIndex].start))) nearestIndex = index;
      }
      const sector = light.sectors[nearestIndex];
      const offset = wrap(light.a - sector.start);
      const margin = Math.max(0.000001, Math.min(2 * Math.PI / 180, sector.span / 4));
      const blend = Math.max(0, Math.min(1, (offset / margin + 1) / 2));
      const previous = light.sectors[(nearestIndex - 1 + light.sectors.length) % light.sectors.length];
      color = mixColor(LIGHT_COLORS[previous.color] || LIGHT_COLORS.W, LIGHT_COLORS[sector.color] || LIGHT_COLORS.W, blend);
    }
    ctx.globalAlpha = Math.max(0.04, Math.min(1, (1 - light.d / light.range) / 0.25));
    ctx.fillStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 14;
    ctx.beginPath(); ctx.arc(x, y, radius, 0, TAU); ctx.fill(); ctx.shadowBlur = 0; ctx.globalAlpha = 1;
    if (controls.reflection.checked && light.d < 1) {
      ctx.globalAlpha = 0.2; ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(x, horizon); ctx.lineTo(500, top + 500); ctx.stroke(); ctx.globalAlpha = 1;
    }
  }
  ctx.fillStyle = "#9fa6a2";
  if (sim.boat.forward) { ctx.beginPath(); ctx.arc(500, top + 250 - 1000 / 30 + 500, 500, 0, TAU); ctx.fill(); }
  else { ctx.beginPath(); ctx.moveTo(500, top + 250 - 1000 / 16); ctx.lineTo(450, top + 247); ctx.lineTo(550, top + 247); ctx.closePath(); ctx.fill(); }
  ctx.fillStyle = "#f4f1df"; ctx.fillRect(0, top + 248, 1000, 2);
}

function draw(now) {
  if (sim) {
    const step = Math.min(0.1, (now - (lastFrame || now)) / 1000 || DT);
    lastFrame = now;
    if (running) {
      accumulator += step;
      while (accumulator >= DT) {
        adaptSpeed();
        moveBoat(DT);
        for (const light of sim.lights) light.cur = (light.cur + 1) % light.on.length;
        accumulator -= DT;
      }
    }
    ctx.setTransform(canvas.width / 1000, 0, 0, canvas.height / 750, 0, 0);
    drawMap(); drawHorizon();
    ui.heading.textContent = `${String(Math.floor(((sim.boat.theta % TAU) + TAU) % TAU * 180 / Math.PI)).padStart(3, "0")}°`;
    ui.distance.textContent = `${sim.boat.nearest.toFixed(2).replace(".", ",")} MN`;
    ui.status.textContent = `${controls.zone.options[controls.zone.selectedIndex].textContent} · ${running ? "en navigation" : "à l'arrêt"}`;
  }
  frameId = requestAnimationFrame(draw);
}

async function loadZone() {
  running = true; lastFrame = 0; ui.status.textContent = "Chargement de la zone…";
  try {
    const response = await fetch(`./zones/${controls.zone.value}.yaml`);
    if (!response.ok) throw new Error(`Impossible de charger la zone ${controls.zone.value}.yaml`);
    sim = buildSimulation(await response.text());
  } catch (error) { sim = null; ui.status.textContent = error.message; console.error(error); }
}

function resize() {
  const bounds = canvas.getBoundingClientRect(), ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(bounds.width * ratio); canvas.height = Math.round(bounds.height * ratio);
}

window.addEventListener("keydown", (event) => keyHandler(event, true));
window.addEventListener("keyup", (event) => keyHandler(event, false));
window.addEventListener("blur", () => { if (sim) { sim.boat.vx = sim.boat.vy = sim.boat.turn = 0; } });
window.addEventListener("resize", resize);
canvas.addEventListener("mousemove", updateLightTooltip);
canvas.addEventListener("mouseleave", hideLightTooltip);
controls.zone.addEventListener("change", loadZone);
controls.pattern.addEventListener("change", loadZone);
controls.drift.addEventListener("change", () => { if (sim) sim.boat.drift = Number(controls.drift.value) || 0; });
controls.obstacle.addEventListener("change", () => { if (sim) sim.boat.obstacle = Math.max(0.1, Number(controls.obstacle.value) || 5); });
$("#restart").addEventListener("click", loadZone);
resize();
loadZone();
if (!frameId) frameId = requestAnimationFrame(draw);