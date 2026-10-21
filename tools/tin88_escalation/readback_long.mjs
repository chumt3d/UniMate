// Research composition only: unchanged fixed60 original-skin reader, one retained copy per frame.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {prepareRigContext, readbackMotion} from '/opt/readback/native-readback-v2.mjs';
import {correctCreatureSkinFloorResidual} from '/opt/readback/dist/e5-core.mjs';

const [input, output, mode] = process.argv.slice(2);
if (!input || !output || !['raw', 'floor-only'].includes(mode)) throw Error('Expected input, new output, raw|floor-only');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const stringify = value => JSON.stringify(value, (_, v) => ArrayBuffer.isView(v) ? Array.from(v) : v);
const write = (name, value) => fs.writeFileSync(path.join(output, name), stringify(value) + '\n', {flag: 'wx'});
if (fs.statSync(input).size > 4_000_000) throw Error('Outside the retained410 decoded JSON envelope');
const bytes = fs.readFileSync(input), sourceSha256 = hash(bytes), motion = JSON.parse(bytes);
if (motion.frameCount !== 410 || motion.fps !== 30 || motion.jointCount !== 48 || motion.rootTranslationJointIndex !== 1 ||
    motion.rotationsXyzw.length !== 410 * 48 * 4 || motion.rootTranslations.length !== 410 * 3 ||
    !motion.rotationsXyzw.every(Number.isFinite) || !motion.rootTranslations.every(Number.isFinite)) throw Error('Invalid complete410 original-rig channels');
const rig = prepareRigContext({asset:'/opt/fox/fox-animations.glb', geometry:'/opt/fox/geometry.json', profile:'/opt/fox/profile.json'});
fs.mkdirSync(output);
fs.mkdirSync(path.join(output, 'assets'));
fs.writeFileSync(path.join(output, 'assets/original-fox.glb'), rig.asset, {flag:'wx'});
write('rig-context.json', rig.metadata);
const results = [], lifts = [], rootFrames = [], rotationFrames = [], minima = [], joints = [];
let feet, render;
const buffers = {'mesh-positions.f64le':[], 'joint-world-matrices.f64le':[]};
let triangleBytes;
for (let offset = 0; offset <= 350; offset += 50) {
  const from = offset === 0 ? 0 : 10;
  let part = {...motion, frameCount:60, rotationsXyzw:Float64Array.from(motion.rotationsXyzw.slice(offset * 48 * 4, (offset + 60) * 48 * 4)), rootTranslations:Float64Array.from(motion.rootTranslations.slice(offset * 3, (offset + 60) * 3))};
  if (mode === 'floor-only') {
    const correction = correctCreatureSkinFloorResidual({...rig.poseInput, ...part, floorY:rig.metadata.restFullSkinMinimumY,
      geometrySha256:rig.metadata.geometrySha256, skinVertices:rig.probes.map((probe, vertexIndex) => ({...probe, vertexIndex}))});
    if (correction.rotationsXyzw.some((value, i) => value !== part.rotationsXyzw[i])) throw Error('Floor correction changed rotations');
    lifts.push(...correction.worldYLiftsMeters.slice(from));
    part = {...part, rotationsXyzw:correction.rotationsXyzw, rootTranslations:correction.rootTranslations};
  }
  const directory = path.join(output, 'window-' + String(offset).padStart(3, '0'));
  const identity = {sourceSha256, input, offset, keepLocalFrames:[from,60], mode,
    readbackSha256:hash(fs.readFileSync('/opt/readback/native-readback-v2.mjs')),
    coreSha256:hash(fs.readFileSync('/opt/readback/dist/readback-core.mjs')),
    e5CoreSha256:mode === 'floor-only' ? hash(fs.readFileSync('/opt/readback/dist/e5-core.mjs')) : null};
  const receipt = readbackMotion(rig, part, directory, identity, () => {if (hash(fs.readFileSync(input)) !== sourceSha256) throw Error('Input changed');});
  const r = JSON.parse(fs.readFileSync(path.join(directory,'render-inputs.json')));
  const sole = JSON.parse(fs.readFileSync(path.join(directory,'sole-probes.json')));
  if (!render) { render = r; feet = sole.feet.map(foot => ({...foot, probeWorldPositions:[], contactWorldPositions:[]})); }
  for (const [name, spec] of [['mesh-positions.f64le',r.positions], ['joint-world-matrices.f64le',r.jointWorldMatrices]]) {
    const raw = fs.readFileSync(path.join(directory,name)), stride = spec.shape.slice(1).reduce((a,b) => a*b, 1)*8;
    if (raw.length !== 60 * stride) throw Error('Readback binary size differs');
    buffers[name].push(raw.subarray(from * stride));
  }
  const triangles = fs.readFileSync(path.join(directory,'triangle-indices.u32le'));
  if (triangleBytes && !triangleBytes.equals(triangles)) throw Error('Original triangle identities changed between windows');
  triangleBytes = triangles;
  sole.feet.forEach((foot,index) => {
    const staticFoot = value => stringify({...value, probeWorldPositions:[], contactWorldPositions:[]});
    if (staticFoot(foot) !== staticFoot(feet[index])) throw Error('Foot ordering or identity differs');
    feet[index].probeWorldPositions.push(...foot.probeWorldPositions.slice(from));
    feet[index].contactWorldPositions.push(...foot.contactWorldPositions.slice(from));
  });
  minima.push(...sole.minimumSkinY.slice(from)); joints.push(...sole.jointWorldPositions.slice(from));
  rootFrames.push(...part.rootTranslations.slice(from * 3));
  rotationFrames.push(...part.rotationsXyzw.slice(from * 48 * 4));
  results.push({offset, keepLocalFrames:[from,60], receiptSha256:hash(fs.readFileSync(path.join(directory,'readback-receipt.json')))});
}
if (minima.length !== 410 || rootFrames.length !== 1230 || rotationFrames.length !== 410*48*4) throw Error('Overlap was duplicated or frames lost');
if (rotationFrames.some((v,i) => v !== motion.rotationsXyzw[i]) || (mode === 'raw' && rootFrames.some((v,i) => v !== motion.rootTranslations[i]))) throw Error('Assembly changed original channels');
for (const [name, parts] of Object.entries(buffers)) fs.writeFileSync(path.join(output,name),Buffer.concat(parts),{flag:'wx'});
fs.writeFileSync(path.join(output,'triangle-indices.u32le'),triangleBytes,{flag:'wx'});
write('original-rig-motion.json',{...motion,rotationsXyzw:rotationFrames,rootTranslations:rootFrames});
write('sole-probes.json',{frameCount:410,fps:30,feet,minimumSkinY:minima,jointWorldPositions:joints,
  scope:'Actual original weighted skin; no inferred contact labels or gait acceptance'});
render.frameCount = 410; render.positions.shape[0] = 410; render.jointWorldMatrices.shape[0] = 410;
render.originalAsset.path = 'assets/original-fox.glb'; render.cleanupApplied = mode === 'floor-only';
render.motionProvenance = motion.provenance ?? {kind:'unmodified model decode',input,sourceSha256};
// Per-window camera bounds cannot describe the assembled clip; renderer computes complete bounds from vertices.
delete render.cameras;
render.composition = 'Eight unchanged60-frame readbacks; retain0:60 then10:60 seven times, with joined root channels already supplied';
write('render-inputs.json',render);
if (mode === 'floor-only') write('floor-only.json',{method:'unimate.skin_floor_residual.v1', worldYLiftsMeters:lifts,
  scope:'Source-defined upward full-skin correction, frame-local; no gait/contact fix, no lowered airborne frames'});
const names = ['original-rig-motion.json','sole-probes.json','render-inputs.json','mesh-positions.f64le','joint-world-matrices.f64le','triangle-indices.u32le'];
write('readback-receipt.json',{complete:true, mode, frameCount:410, sourceSha256, windows:results,
  files:names.map(name => {const value=fs.readFileSync(path.join(output,name)); return {name,bytes:value.length,sha256:hash(value)};})});
console.log(JSON.stringify({output,frameCount:410,mode,minimumSkinY:Math.min(...minima)}));
