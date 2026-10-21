"""One immediate result lookup, without holding a client slot while remote work runs."""
import json
import sys
import modal

try:
    result = modal.FunctionCall.from_id(sys.argv[1]).get(timeout=0)
    print(json.dumps({"callId": sys.argv[1], "status": "complete", "result": result}), flush=True)
except TimeoutError:
    print(json.dumps({"callId": sys.argv[1], "status": "running"}), flush=True)
