import os, json
path = os.path.join(os.path.expanduser('~'), '.emergent_named_routes.json')
print('path=', path)
if not os.path.exists(path):
    print('named routes file not found')
else:
    with open(path, 'r') as fh:
        data = json.load(fh)
    print('ports:', list(data.keys()))
    for port, bucket in data.items():
        print('port:', port)
        for name, entry in bucket.items():
            print('  -', name, 'agents=', len(entry.get('waypoints', [])), 'date=', entry.get('date'))
