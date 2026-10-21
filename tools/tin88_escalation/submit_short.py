"""Submit once, retain call identity, and disconnect; never wait for remote completion."""
import json
import sys
from pathlib import Path

kind=sys.argv[1]
receipt=Path(sys.argv[2])
if receipt.exists(): raise FileExistsError(receipt)
if kind=='long-skin':
    import long_skin_packet as packet
    case,seed,render,mode=sys.argv[3:]
    args=(case,int(seed),render=='yes',mode)
    function=packet.process
elif kind=='gait':
    import gait_comparison_packet as packet
    args=(sys.argv[3],)
    function=packet.compare
else:
    raise ValueError('Unknown bounded submission')
with packet.app.run(detach=True):
    call=function.spawn(*args,packet.image.object_id)
    result={'kind':kind,'args':args,'image':packet.image.object_id,'callId':call.object_id}
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
