"""Research-only long original-skin readback using the unchanged fixed60 source reader."""
import hashlib
import json
import subprocess
import time
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
BUNDLE = Path('/tmp/tin88-lanes/unimate-escalation/source-tools-v2')
image = (modal.Image.from_id('im-S1yr4GIUoelfQ0X3ghZjAi')
    .add_local_file(BUNDLE / 'native-readback-v2.mjs', '/opt/readback/native-readback-v2.mjs', copy=True)
    .add_local_file(BUNDLE / 'dist/e5-core.mjs', '/opt/readback/dist/e5-core.mjs', copy=True)
    .add_local_file(BUNDLE / 'compose-long-fox.mjs', '/opt/readback/compose-long-fox.mjs', copy=True)
    .add_local_file(HERE / 'readback_long.mjs', '/opt/escalation/readback_long.mjs', copy=True)
    .add_local_file(HERE / 'render_skin.py', '/opt/escalation/render_skin.py', copy=True))
volume = modal.Volume.from_name('tin88-unimate-eval', environment_name='main')
app = modal.App('tin88-unimate-escalation')

@app.function(image=image,volumes={'/evaluation':volume},cpu=(2,2),memory=(4096,4096),timeout=600,retries=0,max_containers=1,scaledown_window=2)
def process(case: str, seed: int, render: bool, mode: str, image_id: str):
    if case not in ('v3-walk','truebones-forward') or seed not in (1729,2718,31415) or mode not in ('raw','floor-only'):
        raise ValueError('Outside declared long comparison')
    source = (Path('/evaluation/unimate-20261021-escalation-c/artist-v3-walk') if case == 'v3-walk' else
              Path('/evaluation/unimate-20261021-escalation-e/artist-truebones-forward')) / str(seed)
    prerequisite = json.loads((source/'receipt.json').read_text())
    if prerequisite['status'] != 'passed': raise ValueError('Unfinished source generation')
    input_path = Path(prerequisite['motion']['path'])
    identity = prerequisite['motion']
    if input_path.stat().st_size != identity['bytes'] or hashlib.sha256(input_path.read_bytes()).hexdigest() != identity['sha256']:
        raise ValueError('Source identity differs')
    root = Path('/evaluation/unimate-20261021-escalation-long-skin') / case / str(seed) / mode
    root.mkdir(parents=True,exist_ok=False)
    receipt = {'status':'running','case':case,'seed':seed,'mode':mode,'image':image_id,'input':identity,'startedUnix':time.time()}
    try:
        commands = [
            ['/opt/escalation/node','/opt/readback/compose-long-fox.mjs','--input',str(input_path),'--sha256',identity['sha256'],
             '--bytes',str(identity['bytes']),'--asset','/opt/fox/fox-animations.glb','--geometry','/opt/fox/geometry.json',
             '--profile','/opt/fox/profile.json','--output',str(root/'decode')],
            ['/opt/escalation/node','/opt/escalation/readback_long.mjs',str(root/'decode/original-rig-410.json'),str(root/'readback'),mode],
        ]
        for index, command in enumerate(commands):
            with (root/f'{index}.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            volume.commit()
        if render:
            rig = json.loads((root/'readback/rig-context.json').read_text())
            command = ['/opt/conda/bin/python','/opt/escalation/render_skin.py','--input',str(root/'readback'),
                '--output',str(root/'renders'),'--floor',str(rig['restFullSkinMinimumY']),'--title',f'{case} seed {seed} / {mode}']
            with (root/'render.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        receipt.update(status='passed',output=str(root))
    except Exception as error:
        receipt.update(status='failed',error=repr(error))
        raise
    finally:
        receipt['seconds']=time.time()-receipt['startedUnix']
        (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        volume.commit()
    return receipt

@app.local_entrypoint()
def main(case: str, seed: int = 1729, render: bool = True, mode: str = 'raw'):
    call=process.spawn(case,seed,render,mode,image.object_id)
    print(json.dumps({'callId':call.object_id,'case':case,'seed':seed,'mode':mode}),flush=True)
