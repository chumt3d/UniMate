"""Corrected diagnostic views from already-retained original weighted vertices."""
import json
import subprocess
import time
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
image=modal.Image.from_id('im-KhepHmwYkODnf6SDKdLAjZ').add_local_file(HERE/'render_skin.py','/opt/escalation/render_skin.py',copy=True)
volume=modal.Volume.from_name('tin88-unimate-eval',environment_name='main')
app=modal.App('tin88-unimate-escalation')

@app.function(image=image,volumes={'/evaluation':volume},cpu=(2,2),memory=(4096,4096),timeout=600,retries=0,max_containers=1,scaledown_window=2)
def render(case: str,seed: int,image_id: str):
    if case not in ('v3-walk','truebones-forward') or seed not in (1729,2718,31415): raise ValueError('Unknown retained case')
    source=Path('/evaluation/unimate-20261021-escalation-long-skin')/case/str(seed)/'raw'
    if json.loads((source/'receipt.json').read_text())['status']!='passed': raise ValueError('Incomplete readback')
    destination=source/'renders-clear-floor-v2'
    if destination.exists(): raise FileExistsError(destination)
    rig=json.loads((source/'readback/rig-context.json').read_text())
    started=time.time()
    command=['/opt/conda/bin/python','/opt/escalation/render_skin.py','--input',str(source/'readback'),'--output',str(destination),
       '--floor',str(rig['restFullSkinMinimumY']),'--title',f'{case} seed {seed} / unmodified model']
    with (source/'clear-floor-v2-render.log').open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    receipt={'status':'passed','image':image_id,'seconds':time.time()-started,'output':str(destination)}
    (source/'clear-floor-v2-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); volume.commit()
    return receipt
