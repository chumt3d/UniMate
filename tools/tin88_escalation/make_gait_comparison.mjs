// Explicit research comparator. Existing procedural kernel owns every solve; no model-gait acceptance.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {prepareRigContext} from '/opt/readback/native-readback-v2.mjs';
import {pose, Vector3} from '/opt/readback/dist/readback-core.mjs';
import {synthesizeProceduralCreatureGaitWindowV1, CreatureProceduralLocomotionProfileV1Schema} from '/opt/readback/dist/gait-kernel.mjs';

const [inputPath, output, arm] = process.argv.slice(2);
if (!inputPath || !output || !['procedural','hybrid'].includes(arm)) throw Error('Expected decoded410 input, new output and procedural|hybrid');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const stringify = value => JSON.stringify(value,(_,v)=>ArrayBuffer.isView(v)?Array.from(v):v);
const rawBytes=fs.readFileSync(inputPath), raw=JSON.parse(rawBytes);
if(raw.frameCount!==410||raw.jointCount!==48||raw.fps!==30||raw.rotationsXyzw.length!==410*48*4||!raw.rotationsXyzw.every(Number.isFinite)) throw Error('Expected complete decoded410 source');
const rig=prepareRigContext({asset:'/opt/fox/fox-animations.glb',geometry:'/opt/fox/geometry.json',profile:'/opt/fox/profile.json'});
const controlBytes=fs.readFileSync('/opt/fox/control-declaration.json'), control=JSON.parse(controlBytes);
const reference=JSON.parse(fs.readFileSync('/opt/fox/control.json'));
if(control.assetSha256!==rig.metadata.assetSha256||reference.rootTranslationJointIndex!==rig.metadata.rootTranslationJointIndex) throw Error('Control rig differs');
const rest=Float64Array.from(rig.metadata.restRotationsXyzw), restRoot=Float64Array.from(rig.metadata.restRootTranslation);
const restWorlds=pose(rig.poseInput,rest,restRoot,0), rootIndex=rig.metadata.rootTranslationJointIndex;
const rootWorld=new Vector3().setFromMatrixPosition(restWorlds[rootIndex]);
const index=new Map(rig.metadata.joints.map((joint,i)=>[joint.targetId,i]));
const h=control.locomotion.limbs.reduce((sum,limb)=>sum+new Vector3().setFromMatrixPosition(restWorlds[index.get(limb.chainTargetIds[0])]).y-rig.metadata.restFullSkinMinimumY,0)/control.locomotion.limbs.length;
if(!(h>0)) throw Error('No positive measured hip height');
const fps=30, speed=control.speedMetersPerSecond, period=control.parameters.cycleFrames;
const meanSwing=control.parameters.limbs.reduce((sum,limb)=>sum+limb.swingHeightMeters,0)/control.parameters.limbs.length;
// One knot makes gravity/Froude irrelevant to interpolation; 9.80665 is standard gravity, not an empirical gait fit.
const gravity=9.80665;
const profile=CreatureProceduralLocomotionProfileV1Schema.parse({schemaVersion:'creature_procedural_locomotion.v1',
  sourceIdentity:'TIN88 explicit diagnostic comparator derived from exposed known-plant control; not a naturalness or anatomical profile',
  gravityMetersPerSecondSquared:gravity,
  groups:control.parameters.limbs.map(limb=>({id:limb.id,limbIds:[limb.id]})),
  speedKnots:[{froude:0,strideOverHipHeight:speed*period/(fps*h),swingHeightOverHipHeight:meanSwing/h,
    groups:control.parameters.limbs.map(limb=>({id:limb.id,phase:limb.phase,dutyFactor:limb.dutyFactor}))}],
  body:null,axial:[],variation:null});
const forward=new Vector3().fromArray(control.locomotion.forwardWorld).normalize();
const routeEndFrame=410+Math.ceil(2*period); // Includes the kernel's next-stance footprint lookahead.
const waypoints=[{arrivalFrame:0,positionWorld:rootWorld.toArray()},
  {arrivalFrame:routeEndFrame,positionWorld:rootWorld.clone().addScaledVector(forward,speed*routeEndFrame/fps).toArray()}];
const sourceSha256=arm==='hybrid'?hash(rawBytes):hash(controlBytes);
const recipe={...reference.recipe,sourceSha256,arm:arm==='hybrid'?'reference-seeded':'authored-only'};
delete recipe.rule; delete recipe.algorithmVersion; delete recipe.parameters; delete recipe.exportEligible;
if(arm==='hybrid') recipe.reference={sourceIdentity:'UniMate v3 b78 decoded original-rig nonleg rotations; synthetic authored legs/root',
  licenseIdentity:'UniMate released weights CC BY-NC 4.0; commercial use unverified',fileSha256:hash(rawBytes),
  calibrationSha256:hash(controlBytes),commercialLicenseVerified:false};
fs.mkdirSync(output);
const write=(name,value)=>fs.writeFileSync(path.join(output,name),stringify(value)+'\n',{flag:'wx'});
write('declared-experiment.json',{arm,sourceSha256,inputPath,controlSha256:hash(controlBytes),profile,waypoints,
  parametersSource:control.parameterSource,meanHipHeightMeters:h,meanSwingMeters:meanSwing,calibrationFroude:speed*speed/(gravity*h),
  speedKnotMeaning:'Schema-required zero anchor for a constant diagnostic profile; actual calibration speed and Froude are recorded separately, not a measured zero-speed gait.',
  deviations:['Existing window profile uses one shared mean swing height, while the exposed control uses per-leg heights.',
    'Legs and root are authored from the known-plant diagnostic recipe; hybrid retains generated nonleg rotations only.',
    'Root world XZ follows the authored path; root Y additionally follows the unchanged clearance/reach spring. body:null disables only optional bob, tilt and support shift.',
    'Unrestricted diagnostic bend limits do not establish professional joint limits or naturalness.'],
  kernelSha256:hash(fs.readFileSync('/opt/readback/dist/gait-kernel.mjs'))});
const rotations=[],roots=[],windows=[],phaseComparisons=[]; let previous;
for(let start=0;start<410;start+=60){
  const frameCount=Math.min(60,410-start), q=new Float64Array(frameCount*48*4), translations=new Float64Array(frameCount*3);
  for(let f=0;f<frameCount;f++){
    q.set(arm==='hybrid'?raw.rotationsXyzw.slice((start+f)*48*4,(start+f+1)*48*4):rest,f*48*4);
    translations.set(restRoot,f*3);
  }
  const data={frameCount,fps,joints:rig.metadata.joints,rootParentWorldMatrix:rig.metadata.rootParentWorldMatrix,
    rootTranslationJointIndex:rootIndex,restRotationsXyzw:rest,rotationsXyzw:q,rootTranslations:translations,
    floorY:rig.metadata.restFullSkinMinimumY,nonLegSkinProbes:rig.metadata.nonLegSkinProbes,
    locomotion:control.locomotion,parameters:control.parameters,recipe};
  const result=synthesizeProceduralCreatureGaitWindowV1({rig:data,profile,frameStart:start,waypoints,previous});
  const driven=new Set([...result.drivenRotationTargetIds,rig.metadata.joints[rootIndex].targetId]);
  let unchanged=0;
  for(let j=0;j<48;j++) if(!driven.has(rig.metadata.joints[j].targetId)){
    for(let f=0;f<frameCount;f++) for(let k=0;k<4;k++) if(result.rotationsXyzw[(f*48+j)*4+k]!==q[(f*48+j)*4+k]) throw Error('Unexpected nonleg rotation change');
    unchanged++;
  }
  const name='window-'+String(start).padStart(3,'0')+'.json';
  write(name,result); // Only advance the carried state after the owner has retained the complete result.
  for(const limb of result.plan.limbs){
    const parameter=control.parameters.limbs.find(value=>value.id===limb.id), mismatchedFrames=[];
    for(let f=0;f<frameCount;f++){
      // Retain the explicit known20-frame control comparison; never retune the profile from its output.
      const localFrames=(((start+f)-parameter.phase*period)%period+period)%period;
      const expected=localFrames<parameter.dutyFactor*period?'stance':'swing';
      if(limb.states[f]!==expected) mismatchedFrames.push(start+f);
    }
    phaseComparisons.push({start,id:limb.id,mismatchedFrames,globalTouchdownFrames:limb.touchdownFrames.map(frame=>start+frame)});
  }
  previous=result.state;
  rotations.push(...result.rotationsXyzw); roots.push(...result.rootTranslations);
  windows.push({start,frameCount,path:name,unmodifiedJoints:unchanged,drivenRotationTargetIds:result.drivenRotationTargetIds,
    maximumSolverResidualMeters:Math.max(...result.solverResidualsMeters),findings:result.findings,
    completeCycles:result.plan.limbs.map(limb=>({id:limb.id,count:limb.completeCycles})),nextFrame:previous.nextFrame});
}
if(previous.nextFrame!==410||rotations.length!==410*48*4||roots.length!==410*3) throw Error('Incomplete procedural composition');
write('original-rig-410.json',{frameCount:410,fps:30,jointCount:48,rootTranslationJointIndex:rootIndex,
  originalNodeIndices:rig.metadata.originalNodeIndices,rotationsXyzw:rotations,rootTranslations:roots,
  provenance:{kind:arm,sourceSha256,kernel:'creature_gait.window.v1',legsAndRoot:'authored route and source clearance/reach solver',
    nonlegRotations:arm==='hybrid'?'unchanged generated original-rig local rotations':'original rig rest rotations',
    naturalnessQualified:false,commercialUseVerified:false}});
write('receipt.json',{scope:'Exposed diagnostic procedural/hybrid comparator; not raw model gait or product acceptance',arm,windows,phaseComparisons,
  sourceSha256,outputSha256:hash(fs.readFileSync(path.join(output,'original-rig-410.json')))});
console.log(JSON.stringify({output,arm,frames:410,windows:windows.length}));
