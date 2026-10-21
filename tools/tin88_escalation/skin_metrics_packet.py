"""Actual original-skin measurements. Raw contact/gait classification remains unqualified."""
import hashlib
import json
import time
from pathlib import Path
import modal

image=modal.Image.from_id('im-KhepHmwYkODnf6SDKdLAjZ')
volume=modal.Volume.from_name('tin88-unimate-eval',environment_name='main')
app=modal.App('tin88-unimate-escalation')

@app.function(image=image,volumes={'/evaluation':volume},cpu=(2,2),memory=(2048,2048),timeout=120,retries=0,max_containers=1,scaledown_window=2)
def measure(label: str):
    import numpy as np
    if label not in ('final','first-pass'): raise ValueError('Unknown report identity')
    root=Path('/evaluation/unimate-20261021-escalation-skin-metrics')/label
    root.mkdir(parents=True,exist_ok=False)
    started=time.time()
    cases=[(f'{case}-{seed}',Path('/evaluation/unimate-20261021-escalation-long-skin')/case/str(seed)/'raw',False)
       for case in ('v3-walk','truebones-forward') for seed in (1729,2718,31415)]
    cases += [(arm,Path('/evaluation/unimate-20261021-escalation-gait')/arm,True) for arm in ('procedural','hybrid')]
    cases += [('v3-walk-1729-floor-only',Path('/evaluation/unimate-20261021-escalation-long-skin/v3-walk/1729/floor-only'),False)]
    result={'scope':'Actual original-skin kinematics; no qualified raw contact or naturalness verdict',
      'rawGaitClassification':'F1-F3 unscorable: stance evaluator qualification not established',
      'numpyVersion':np.__version__,'cases':[],'missing':[]}
    for name,folder,authored in cases:
        if not (folder/'receipt.json').is_file(): result['missing'].append({'case':name,'reason':'no completed receipt'}); continue
        receipt=json.loads((folder/'receipt.json').read_text())
        if receipt['status']!='passed': result['missing'].append({'case':name,'reason':receipt['status']}); continue
        readback=folder/'readback'
        skin_path=readback/'sole-probes.json'
        skin=json.loads(skin_path.read_text()); rig=json.loads((readback/'rig-context.json').read_text())
        if skin['frameCount']!=410 or skin['fps']!=30: raise ValueError('Unexpected comparison envelope')
        positions=np.asarray(skin['jointWorldPositions'],dtype=float)
        if positions.shape!=(410,48,3) or not np.isfinite(positions).all(): raise ValueError('Invalid joint-world arrays')
        floor=rig['restFullSkinMinimumY']; minima=np.asarray(skin['minimumSkinY'])
        if minima.shape!=(410,) or not np.isfinite(minima).all() or not np.isfinite(floor): raise ValueError('Invalid skin minima/floor')
        roots=positions[:,rig['rootTranslationJointIndex']]
        boundaries=np.arange(60,410,60 if authored else 50)
        centers=np.arange(1,409)
        interior=np.all(np.abs(centers[:,None]-boundaries[None,:])>1,axis=1)
        acceleration=np.linalg.norm(np.diff(positions,n=2,axis=0),axis=2)*30**2
        p99=np.quantile(acceleration[interior],.99,axis=0)
        row={'case':name,'inputSha256':hashlib.sha256(skin_path.read_bytes()).hexdigest(),
          'frameCount':410,'fps':30,'floorY':floor,'fullSkinMinimumYRange':[float(minima.min()),float(minima.max())],
          'fullSkinPenetrationMaxMeters':max(0,float(floor-minima.min())),
          'rootPlanarDisplacementMeters':float(np.linalg.norm((roots[-1]-roots[0])[[0,2]])),
          'rootPlanarPathLengthMeters':float(np.linalg.norm(np.diff(roots[:,[0,2]],axis=0),axis=1).sum()),
          'jointSeams':[],'feet':[],
          'contactEvidence':'authored target states, not detected actual contact' if authored else 'unknown; no raw stance classifier applied'}
        for boundary in boundaries:
            peak=acceleration[[boundary-2,boundary-1]].max(axis=0)
            ratios=np.divide(peak,p99,out=np.zeros_like(peak),where=p99>0)
            row['jointSeams'].append({'firstFrame':int(boundary),'jointsAboveOwnInteriorP99':int(np.sum(peak>p99)),
              'maximumFiniteRatio':float(ratios.max()),'zeroInteriorWithNonzeroSeam':bool(np.any((p99==0)&(peak>0)))})
        states={}; targets={}
        if authored:
            solve_receipt=json.loads((folder/'solve/receipt.json').read_text())
            row['phaseComparison']=solve_receipt['phaseComparisons']
            for window in solve_receipt['windows']:
                data=json.loads((folder/'solve'/window['path']).read_text())
                for limb in data['plan']['limbs']:
                    states.setdefault(limb['id'],[]).extend(limb['states'])
                    targets.setdefault(limb['id'],[]).extend(limb['targetSolesWorld'])
            if any(len(value)!=410 for value in states.values()): raise ValueError('Authored plan lost frames')
        for foot in skin['feet']:
            length=foot['legLengthMeters']; points=np.asarray(foot['contactWorldPositions'])
            probes=np.asarray(foot['probeWorldPositions'])
            if (length<=0 or points.shape!=(410,3) or probes.ndim!=3 or probes.shape[0]!=410 or probes.shape[2]!=3
                    or not np.isfinite(points).all() or not np.isfinite(probes).all()): raise ValueError('Invalid bound foot')
            heights=probes[:,:,1].min(axis=1)-floor
            planar=points[:,[0,2]]
            velocity=np.empty_like(planar)
            velocity[1:-1]=(planar[2:]-planar[:-2])*15
            velocity[0]=(planar[1]-planar[0])*30; velocity[-1]=(planar[-1]-planar[-2])*30
            speed=np.linalg.norm(velocity,axis=1)/length
            low=heights<=.05*length  # Declared descriptive height band; not the rejected contact classifier.
            buried=heights < -.02*length  # Reviewer-requested burial band; report measurements without gait classification.
            low_intervals=[]; low_start=None
            for f in range(411):
                if f<410 and low[f] and low_start is None: low_start=f
                if low_start is not None and (f==410 or not low[f]):
                    relative=(points-roots)[low_start:f][:,[0,2]]
                    low_intervals.append({'startFrame':low_start,'endFrameExclusive':f,
                      'minimumWorldHorizontalSpeedLPerSecond':float(speed[low_start:f].min()),
                      'rootPlanarDisplacementMeters':float(np.linalg.norm((roots[f-1]-roots[low_start])[[0,2]])),
                      'rootPlanarPathLengthMeters':float(np.linalg.norm(np.diff(roots[low_start:f][:,[0,2]],axis=0),axis=1).sum()),
                      'rootTranslationRelativeFootDisplacementMeters':float(np.linalg.norm(relative[-1]-relative[0])),
                      'rootTranslationRelativeFootPathLengthMeters':float(np.linalg.norm(np.diff(relative,axis=0),axis=1).sum()),
                      'referenceFrame':'World XZ minus root translation; orientation is not removed.'})
                    low_start=None
            measurement={'legId':foot['legId'],'legLengthMeters':length,
              'soleHeightRangeL':[float(heights.min()/length),float(heights.max()/length)],
              'solePenetrationMaxL':max(0,float(-heights.min()/length)),
              'framesBelowBurialBand':int(buried.sum()),'burialBandL':-.02,
              'lowHeightIntervals':low_intervals,
              'lowIntervalMethod':'Minimum sole height<=0.05 per-leg length, no speed filtering or contact label; fixed contact-probe horizontal central velocity with one-sided endpoints.',
              'seamContactProbeStepL':[{'firstFrame':int(f),'stepL':float(np.linalg.norm(points[f]-points[f-1])/length)} for f in boundaries],
              'declaredStance':None}
            if authored:
                selected=np.asarray([state=='stance' for state in states[foot['legId']]])
                planned=np.asarray(targets[foot['legId']])
                episodes=[]; start=None
                for f in range(411):
                    if f<410 and selected[f] and start is None: start=f
                    if start is not None and (f==410 or not selected[f]):
                        section=points[start:f]
                        # Exact pairwise excursion for each short declared stance; not an inferred contact label.
                        section_planar=section[:,[0,2]]
                        excursion=float(np.linalg.norm(section_planar[:,None,:]-section_planar[None,:,:],axis=-1).max())
                        episodes.append({'startFrame':start,'endFrameExclusive':f,'planarExcursionL':excursion/length})
                        start=None
                residual=np.linalg.norm(points-planned,axis=1)
                measurement['declaredStance']={'targetStanceFraction':float(selected.mean()),'episodes':episodes,
                  'maximumStanceTargetResidualL':float(residual[selected].max()/length) if selected.any() else None,
                  'maximumStancePlanarExcursionL':max((episode['planarExcursionL'] for episode in episodes),default=None)}
            row['feet'].append(measurement)
        result['cases'].append(row)
    result.update(complete=len(result['missing'])==0,seconds=time.time()-started)
    (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); volume.commit()
    return {'output':str(root/'report.json'),'complete':result['complete'],'cases':len(result['cases']),'missing':result['missing'],'seconds':result['seconds']}

@app.local_entrypoint()
def main(label: str='final'):
    call=measure.spawn(label)
    print(json.dumps({'callId':call.object_id,'label':label}),flush=True)
