"""Compare authored legs/root with model nonleg motion, through unchanged source kernels."""
import json
import subprocess
import time
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
BUNDLE=Path('/tmp/tin88-lanes/unimate-escalation/source-tools-v2')
CONTROL=Path('/mnt/c/Users/bened/.codex/visualizations/2026/10/21/01a15d5e-b142-7272-acae-7b030533b12c/unimate-procedural-control')
image=(modal.Image.from_id('im-KhepHmwYkODnf6SDKdLAjZ')
    .add_local_file(BUNDLE/'dist/gait-kernel.mjs','/opt/readback/dist/gait-kernel.mjs',copy=True)
    .add_local_file(CONTROL/'synthetic-control-declaration.json','/opt/fox/control-declaration.json',copy=True)
    .add_local_file(HERE/'render_skin.py','/opt/escalation/render_skin.py',copy=True)
    .add_local_file(HERE/'make_gait_comparison.mjs','/opt/escalation/make_gait_comparison.mjs',copy=True))
volume=modal.Volume.from_name('tin88-unimate-eval',environment_name='main')
app=modal.App('tin88-unimate-escalation')

@app.function(image=image,volumes={'/evaluation':volume},cpu=(2,2),memory=(4096,4096),timeout=600,retries=0,max_containers=1,scaledown_window=2)
def compare(arm: str,image_id: str):
    if arm not in ('procedural','hybrid'): raise ValueError('Unknown comparator')
    source=Path('/evaluation/unimate-20261021-escalation-long-skin/v3-walk/1729/raw')
    if json.loads((source/'receipt.json').read_text())['status']!='passed': raise ValueError('Source readback unfinished')
    root=Path('/evaluation/unimate-20261021-escalation-gait')/arm
    root.mkdir(parents=True,exist_ok=False)
    receipt={'status':'running','arm':arm,'image':image_id,'startedUnix':time.time()}
    try:
        commands=[['/opt/escalation/node','/opt/escalation/make_gait_comparison.mjs',str(source/'decode/original-rig-410.json'),str(root/'solve'),arm],
          ['/opt/escalation/node','/opt/escalation/readback_long.mjs',str(root/'solve/original-rig-410.json'),str(root/'readback'),'raw']]
        for i,command in enumerate(commands):
            with (root/f'{i}.log').open('w') as log: subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            volume.commit()
        rig=json.loads((root/'readback/rig-context.json').read_text())
        with (root/'render.log').open('w') as log:
            subprocess.run(['/opt/conda/bin/python','/opt/escalation/render_skin.py','--input',str(root/'readback'),'--output',str(root/'renders'),
                '--floor',str(rig['restFullSkinMinimumY']),'--title',arm+' / authored legs and path'],check=True,stdout=log,stderr=subprocess.STDOUT)
        receipt.update(status='passed',output=str(root))
    except Exception as error:
        receipt.update(status='failed',error=repr(error)); raise
    finally:
        receipt['seconds']=time.time()-receipt['startedUnix']
        (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); volume.commit()
    return receipt

@app.local_entrypoint()
def main(arm: str):
    call=compare.spawn(arm,image.object_id)
    print(json.dumps({'callId':call.object_id,'arm':arm}),flush=True)
