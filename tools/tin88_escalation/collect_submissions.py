"""One immediate lookup per supplied submission; no waiting or retry loop."""
import json
import sys
import time
from pathlib import Path
import modal

for name in sys.argv[1:]:
    source = Path(name)
    submission = json.loads(source.read_text())
    call_id = submission['callId']
    result = {'submission': str(source), 'callId': call_id, 'collectedUnix': time.time()}
    try:
        result.update(status='complete', result=modal.FunctionCall.from_id(call_id).get(timeout=0))
    except TimeoutError:
        result['status'] = 'running'
    except Exception as error:
        result.update(status='failed', errorType=type(error).__name__, error=str(error))
    source.with_suffix('.result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
