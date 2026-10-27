// SPDX-FileCopyrightText: Copyright (c) 2026 Patch
// SPDX-License-Identifier: Apache-2.0

// Plays a crowdbricks take: every character, and each one's plan (the motion the model generated that it
// has not played yet) as pink holograms and footprints. Open with ?take=path/to.take, or drop a take on the page.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { Rig, loadCharacterPack } from "./character.js";

const PINK = 0xf53cd1;
const PLAN_RATE = 30;
const GHOST_STEP = 7;
const DISSOLVE = 0.28;
const PRINT = 0.32;

// ---------------------------------------------------------------- the take

function parseTake(buffer) {
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
  if (magic !== "CBT1") throw new Error(`not a crowdbricks take (magic ${magic})`);
  const headerLength = new DataView(buffer).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)));
  const base = 8 + headerLength;
  const types = { float32: Float32Array, int16: Int16Array, int32: Int32Array, uint8: Uint8Array };
  const read = (entry) => new types[entry.dtype](buffer, base + entry.offset, entry.length);
  const take = {
    header,
    world: header.world || {},
    capacity: header.capacity,
    modelJoints: header.model_joints,
    times: read(header.times),
    alive: read(header.alive),
    positions: read(header.positions),
    rotations: read(header.rotations),
    byCharacter: Array.from({ length: header.capacity }, () => []),
    footfalls: [],
    plans: null,
  };
  const plans = header.plans;
  if (plans && plans.positions) {
    const p = {
      count: plans.count,
      character: read(plans.character),
      clock: read(plans.clock),
      frames: read(plans.frames),
      labels: plans.labels,
      positions: read(plans.positions),
      rotations: read(plans.rotations),
      contacts: read(plans.contacts),
      feet: read(plans.feet),
      offsets: new Int32Array(plans.count),
    };
    for (let g = 1; g < p.count; g++) p.offsets[g] = p.offsets[g - 1] + p.frames[g - 1];
    for (let g = 0; g < p.count; g++) take.byCharacter[p.character[g]].push(g);
    take.plans = p;
    take.footfalls = Array.from({ length: p.count }, (_, g) => footfalls(p, g));
  }
  return take;
}

// Contact joints are LeftFoot, LeftToeBase, RightFoot, RightToeBase. A footfall is where a foot's contact rises
// past one half; it sits between the foot and the toe two frames later, pointing heel to toe.
function footfalls(p, g) {
  const out = [];
  const frames = p.frames[g];
  const offset = p.offsets[g];
  for (const [heel, toe] of [[0, 1], [2, 3]]) {
    let was = true;
    for (let f = 0; f < frames; f++) {
      const down = Math.max(p.contacts[(offset + f) * 4 + heel], p.contacts[(offset + f) * 4 + toe]) >= 128;
      if (down && !was) {
        const k = Math.min(f + 2, frames - 1);
        const at = (joint) => (offset + k) * 12 + joint * 3;
        const hx = p.feet[at(heel)], hy = p.feet[at(heel) + 1], tx = p.feet[at(toe)], ty = p.feet[at(toe) + 1];
        out.push({ frame: f, x: (hx + tx) / 2, y: (hy + ty) / 2, yaw: Math.atan2(ty - hy, tx - hx) });
      }
      was = down;
    }
  }
  return out;
}

function lastAtOrBefore(sorted, s) {
  let lo = 0;
  let hi = sorted.length - 1;
  if (s <= sorted[0]) return 0;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (sorted[mid] <= s) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

// ---------------------------------------------------------------- holograms

const HOLO_VERTEX = /* glsl */ `
#include <common>
#include <skinning_pars_vertex>
varying vec3 vNormalView;
varying vec3 vToCamera;
varying vec3 vWorld;
void main() {
  #include <skinbase_vertex>
  #include <begin_vertex>
  #include <beginnormal_vertex>
  #include <skinnormal_vertex>
  #include <defaultnormal_vertex>
  #include <skinning_vertex>
  #include <project_vertex>
  vNormalView = normalize(transformedNormal);
  vToCamera = normalize(-mvPosition.xyz);
  vWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;
}`;

// A fresnel rim with fine scanlines; a new plan prints from the feet up. Blended by MAX, so dozens of
// overlapping plans stay pink instead of summing to white.
const HOLO_FRAGMENT = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
uniform float uReveal;
uniform float uTime;
varying vec3 vNormalView;
varying vec3 vToCamera;
varying vec3 vWorld;
void main() {
  float top = uReveal * 2.2 - 0.05;
  if (vWorld.y > top) discard;
  float facing = clamp(abs(dot(normalize(vNormalView), normalize(vToCamera))), 0.0, 1.0);
  float rim = pow(1.0 - facing, 2.4);
  float lines = mix(0.55, 1.0, 0.5 + 0.5 * sin(vWorld.y * 150.0 - uTime * 5.0));
  float seam = (1.0 - smoothstep(0.0, 0.07, top - vWorld.y)) * (1.0 - step(0.999, uReveal));
  vec3 color = uColor * (0.05 + 1.1 * rim + seam * 0.8) * lines * 1.6 + vec3(1.0, 0.82, 0.96) * (rim * rim * 0.9 + seam * 0.6);
  gl_FragColor = vec4(color * uOpacity, 1.0);
}`;

function holoMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(PINK) }, uOpacity: { value: 1 }, uReveal: { value: 1 }, uTime: { value: 0 } },
    vertexShader: HOLO_VERTEX,
    fragmentShader: HOLO_FRAGMENT,
    transparent: true,
    depthWrite: false,
    blending: THREE.CustomBlending,
    blendEquation: THREE.MaxEquation,
  });
}

function footprintTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 128;
  const g = canvas.getContext("2d");
  const gradient = g.createRadialGradient(32, 64, 4, 32, 64, 30);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.6, "rgba(255,255,255,0.75)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  g.translate(32, 64);
  g.scale(1, 2);
  g.translate(-32, -64);
  g.fillStyle = gradient;
  g.fillRect(0, 0, 64, 128);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

// ---------------------------------------------------------------- scene

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.NeutralToneMapping;
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
scene.fog = new THREE.Fog(0x000000, 30, 100);
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 300);
camera.position.set(14, 12, 16);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.8, 0);
controls.enableDamping = true;
const zup = new THREE.Group();
zup.rotation.x = -Math.PI / 2;
scene.add(zup);
scene.add(new THREE.HemisphereLight(0xffffff, 0x000000, 0.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(-20, 30, 15);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
Object.assign(sun.shadow.camera, { left: -30, right: 30, top: 30, bottom: -30, near: 1, far: 90 });
scene.add(sun);

const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 }));
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass(new THREE.Vector2(1, 1), 0.8, 0.5, 0.82));
composer.addPass(new OutputPass());
const resize = () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
};
window.addEventListener("resize", resize);
resize();

const floor = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({ color: 0x0c0c0c, roughness: 0.95 }));
floor.receiveShadow = true;
zup.add(floor);
const grid = new THREE.GridHelper(80, 80, 0x2a2a2a, 0x161616);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.002;
zup.add(grid);

const characterMaterial = new THREE.MeshStandardMaterial({ color: 0xf1f1f1, roughness: 0.55 });
const world = new THREE.Group();
zup.add(world);
let pack;
let take = null;
let characters = [];
const ghosts = [];
let ghostsUsed = 0;
const FOOTPRINT_MAX = 8192;
const footprints = new THREE.InstancedMesh(
  new THREE.PlaneGeometry(0.11, 0.24),
  new THREE.MeshBasicMaterial({
    map: footprintTexture(), color: new THREE.Color(PINK).multiplyScalar(2.2), transparent: true, premultipliedAlpha: true,
    depthWrite: false, blending: THREE.CustomBlending, blendEquation: THREE.MaxEquation,
  }),
  FOOTPRINT_MAX,
);
footprints.setColorAt(0, new THREE.Color(0, 0, 0));
footprints.frustumCulled = false;
zup.add(footprints);
let footprintsUsed = 0;
const qa = new THREE.Quaternion();
const qb = new THREE.Quaternion();
const matrix = new THREE.Matrix4();
const shade = new THREE.Color();

function buildWorld(description) {
  world.clear();
  const white = new THREE.MeshStandardMaterial({ color: 0xe9e9e9, roughness: 0.8 });
  const box = (w, d, h, x, y, z, yaw) => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, d, h), white);
    mesh.position.set(x, y, z);
    mesh.rotation.z = yaw;
    mesh.castShadow = mesh.receiveShadow = true;
    world.add(mesh);
  };
  for (const cube of description.cubes || []) {
    const [x, y, z] = cube.center;
    if (cube.kind === "duck") {
      // A bar at the height the duck sequences are authored for, on two posts, across the way through.
      const across = cube.yaw - Math.PI / 2;
      const bottom = z - cube.size[2] / 2;
      box(1.5, 0.07, 0.07, x, y, bottom + 0.035, across);
      for (const side of [-1, 1]) box(0.06, 0.06, bottom, x + side * 0.75 * Math.cos(across), y + side * 0.75 * Math.sin(across), bottom / 2, across);
      continue;
    }
    box(...cube.size, x, y, z, cube.yaw);
  }
}

function useTake(next) {
  take = next;
  for (const rig of characters) zup.remove(rig.group);
  characters = [];
  for (let slot = 0; slot < take.capacity; slot++) {
    const rig = new Rig(pack, characterMaterial, { castShadow: true });
    rig.visible = false;
    zup.add(rig.group);
    characters.push(rig);
  }
  buildWorld(take.world);
  const follow = document.getElementById("follow");
  follow.innerHTML = '<option value="-1">none</option>' + Array.from({ length: take.capacity }, (_, k) => `<option value="${k}">${k}</option>`).join("");
  state.time = take.times[0];
  slider.min = take.times[0];
  slider.max = take.times[take.times.length - 1];
  state.playing = true;
}

function ghostRig() {
  let ghost = ghosts[ghostsUsed];
  if (!ghost) {
    const material = holoMaterial();
    ghost = { rig: new Rig(pack, material), material };
    zup.add(ghost.rig.group);
    ghosts.push(ghost);
  }
  ghostsUsed += 1;
  ghost.rig.visible = true;
  return ghost;
}

// ---------------------------------------------------------------- posing

function poseFromPlan(rig, g, frame) {
  const p = take.plans;
  const at = p.offsets[g] + Math.max(0, Math.min(frame, p.frames[g] - 1));
  rig.bones[0].position.set(p.positions[at * 3], p.positions[at * 3 + 1], p.positions[at * 3 + 2]);
  take.modelJoints.forEach((joint, m) => {
    const o = at * 120 + m * 4;
    rig.bones[joint].quaternion.set(p.rotations[o], p.rotations[o + 1], p.rotations[o + 2], p.rotations[o + 3]).normalize();
  });
}

function poseCrowd(s) {
  const capacity = take.capacity;
  const i = lastAtOrBefore(take.times, s);
  const j = Math.min(i + 1, take.times.length - 1);
  const span = take.times[j] - take.times[i];
  const t = span > 0 ? THREE.MathUtils.clamp((s - take.times[i]) / span, 0, 1) : 0;
  let alive = 0;
  for (let slot = 0; slot < capacity; slot++) {
    const rig = characters[slot];
    const aliveA = take.alive[i * capacity + slot];
    const aliveB = take.alive[j * capacity + slot];
    rig.visible = Boolean(aliveA || aliveB);
    if (!aliveA && !aliveB) continue;
    alive += 1;
    const from = aliveA ? i : j;
    const to = aliveB ? j : i;
    const pa = (from * capacity + slot) * 3;
    const pb = (to * capacity + slot) * 3;
    rig.bones[0].position.set(
      take.positions[pa] + (take.positions[pb] - take.positions[pa]) * t,
      take.positions[pa + 1] + (take.positions[pb + 1] - take.positions[pa + 1]) * t,
      take.positions[pa + 2] + (take.positions[pb + 2] - take.positions[pa + 2]) * t,
    );
    const ra = (from * capacity + slot) * 120;
    const rb = (to * capacity + slot) * 120;
    take.modelJoints.forEach((joint, m) => {
      const o = m * 4;
      qa.set(take.rotations[ra + o], take.rotations[ra + o + 1], take.rotations[ra + o + 2], take.rotations[ra + o + 3]).normalize();
      qb.set(take.rotations[rb + o], take.rotations[rb + o + 1], take.rotations[rb + o + 2], take.rotations[rb + o + 3]).normalize();
      rig.bones[joint].quaternion.slerpQuaternions(qa, qb, t);
    });
  }
  return alive;
}

function addFootprint(x, y, yaw, value) {
  if (footprintsUsed >= FOOTPRINT_MAX || value <= 0.003) return;
  matrix.makeRotationZ(yaw - Math.PI / 2);
  matrix.setPosition(x, y, 0.006 + footprintsUsed * 1e-6);
  footprints.setMatrixAt(footprintsUsed, matrix);
  footprints.setColorAt(footprintsUsed, shade.setScalar(value));
  footprintsUsed += 1;
}

function drawPlan(g, now, { fade = 1, age = 1, time = 0, withGhosts = true, withPrints = true }) {
  const p = take.plans;
  const frames = p.frames[g];
  if (frames < 2 || fade <= 0) return;
  const last = frames - 1;
  if (withGhosts) {
    const marks = [];
    for (let f = GHOST_STEP; f < last - 2; f += GHOST_STEP) marks.push(f);
    marks.push(last);
    for (const f of marks) {
      const ahead = f - now;
      if (ahead <= 0.5) continue;
      const reveal = THREE.MathUtils.clamp((age - (f / frames) * PRINT * 0.8) / (PRINT * 0.45), 0, 1);
      if (reveal <= 0) continue;
      const ghost = ghostRig();
      poseFromPlan(ghost.rig, g, f);
      const near = THREE.MathUtils.clamp((ahead - 0.5) / 5, 0, 1);
      const far = 1 - 0.45 * (f / Math.max(last, 1));
      ghost.material.uniforms.uOpacity.value = fade * near * far * (f === last ? 1.35 : 1);
      ghost.material.uniforms.uReveal.value = reveal;
      ghost.material.uniforms.uTime.value = time;
    }
  }
  if (withPrints) {
    for (const step of take.footfalls[g]) {
      const ahead = step.frame - now;
      const reveal = THREE.MathUtils.clamp((age - (step.frame / frames) * PRINT * 0.8) / 0.08, 0, 1);
      const landed = ahead < 0 ? Math.max(0, 1 + ahead / 15) * 1.6 : 1;
      addFootprint(step.x, step.y, step.yaw, fade * reveal * landed * 0.55);
    }
  }
}

function drawPlans(s, options) {
  ghostsUsed = 0;
  footprintsUsed = 0;
  if (take.plans) {
    for (let slot = 0; slot < take.capacity; slot++) {
      if (!characters[slot].group.visible) continue;
      const list = take.byCharacter[slot];
      let index = -1;
      for (let k = 0; k < list.length && take.plans.clock[list[k]] <= s + 1e-6; k++) index = k;
      if (index < 0) continue;
      const g = list[index];
      const clock = take.plans.clock[g];
      drawPlan(g, (s - clock) * PLAN_RATE, { ...options, age: s - clock, time: s });
      if (index > 0 && s - clock < DISSOLVE) {
        const before = list[index - 1];
        drawPlan(before, (s - take.plans.clock[before]) * PLAN_RATE, { ...options, fade: 1 - (s - clock) / DISSOLVE, age: 9, time: s });
      }
    }
  }
  for (let k = ghostsUsed; k < ghosts.length; k++) ghosts[k].rig.visible = false;
  footprints.count = footprintsUsed;
  footprints.instanceMatrix.needsUpdate = true;
  if (footprints.instanceColor) footprints.instanceColor.needsUpdate = true;
}

// ---------------------------------------------------------------- the page

const hud = document.getElementById("hud");
const slider = document.getElementById("time");
const clockText = document.getElementById("clock");
const playButton = document.getElementById("play");
const state = { time: 0, playing: true, speed: 1 };
playButton.onclick = () => { state.playing = !state.playing; };
slider.oninput = () => { state.time = Number(slider.value); };
document.getElementById("speed").onchange = (event) => { state.speed = Number(event.target.value); };
window.addEventListener("keydown", (event) => { if (event.code === "Space") { state.playing = !state.playing; event.preventDefault(); } });

async function openBuffer(buffer, name) {
  useTake(parseTake(buffer));
  take.name = name;
}
const fileInput = document.getElementById("file");
document.getElementById("open").onclick = () => fileInput.click();
fileInput.onchange = async () => { const file = fileInput.files[0]; if (file) await openBuffer(await file.arrayBuffer(), file.name); };
const drop = document.getElementById("drop");
window.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("show"); });
window.addEventListener("dragleave", () => drop.classList.remove("show"));
window.addEventListener("drop", async (event) => {
  event.preventDefault();
  drop.classList.remove("show");
  const file = event.dataTransfer.files[0];
  if (file) await openBuffer(await file.arrayBuffer(), file.name);
});

const clock = new THREE.Clock();
function tick() {
  requestAnimationFrame(tick);
  const dt = Math.min(clock.getDelta(), 0.1);
  if (take) {
    const end = take.times[take.times.length - 1];
    if (state.playing) {
      state.time += dt * state.speed;
      if (state.time > end) state.time = take.times[0];
    }
    slider.value = state.time;
    clockText.textContent = `${state.time.toFixed(2)} s`;
    playButton.textContent = state.playing ? "pause" : "play";
    const alive = poseCrowd(state.time);
    drawPlans(state.time, { withGhosts: document.getElementById("plans").checked, withPrints: document.getElementById("prints").checked });
    const follow = Number(document.getElementById("follow").value);
    if (follow >= 0 && characters[follow]?.group.visible) {
      const root = characters[follow].bones[0].position;
      const target = new THREE.Vector3(root.x, 0.9, -root.y);
      camera.position.add(target.clone().sub(controls.target));
      controls.target.copy(target);
    }
    const batches = take.header.batches || [];
    const sizes = batches.map((b) => b[1]);
    const gpu = batches.map((b) => b[2]).sort((a, b) => a - b);
    hud.innerHTML = `<b>${take.name || "take"}</b><br>${alive} characters · ${ghostsUsed} plan ghosts<br>` +
      `<span class="quiet">${take.plans ? take.plans.count : 0} generations in ${batches.length} batches` +
      (batches.length ? ` (mean ${(sizes.reduce((a, b) => a + b, 0) / sizes.length).toFixed(1)}, median ${gpu[gpu.length >> 1].toFixed(2)} ms GPU)` : "") +
      `</span><br><span class="quiet">drag to orbit · space to pause</span>`;
  }
  controls.update();
  composer.render();
}

(async () => {
  try {
    pack = await loadCharacterPack("soma.bin");
  } catch (error) {
    hud.innerHTML = "soma.bin is missing: run <b>python viewer/export_character.py</b> with your MotionBricks checkout on PYTHONPATH.";
    return;
  }
  const url = new URLSearchParams(location.search).get("take");
  if (url) {
    const response = await fetch(url);
    if (!response.ok) hud.textContent = `could not load ${url}: HTTP ${response.status}`;
    else await openBuffer(await response.arrayBuffer(), url.split("/").pop());
  } else {
    hud.innerHTML = "open a take (<b>examples/futures.py --record</b> writes one), or drop it here";
  }
  tick();
})();
