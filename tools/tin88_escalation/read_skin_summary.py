"""Small read-only inspection of retained remote original-skin channels."""
import json
import modal

volume=modal.Volume.from_name('tin88-unimate-eval',environment_name='main')
for label,directory in [
    ('control','unimate-20261021-escalation-skin/initial/procedural-known-plant-control/readback/control.json-28704e4c094b'),
    ('v3long','unimate-20261021-escalation-long-skin/v3-walk/1729/raw/readback')]:
    data=json.loads(b''.join(volume.read_file(directory+'/sole-probes.json')))
    print(json.dumps({'case':label,'minima':[min(data['minimumSkinY']),max(data['minimumSkinY'])],
      'firstMinimum':data['minimumSkinY'][0], 'footKeys':list(data['feet'][0]),
      'firstFootPoints':[foot['contactWorldPositions'][0] for foot in data['feet']]}),flush=True)
