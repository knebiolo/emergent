import emergent.ship_abm.ofs_loader as ofs
import traceback, datetime as dt, numpy as np, os

logdir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logdir, exist_ok=True)
logp = os.path.join(logdir, 'ofs_diag_seattle.out')

with open(logp, 'w', encoding='utf-8') as fh:
    fh.write('start diagnostic\n')
    try:
        wf = ofs.get_wind_fn('Seattle', start=dt.datetime.utcnow())
        fh.write('get_wind_fn returned sampler: %s\n' % (repr(wf),))
        try:
            sample = wf(np.array([-122.335167]), np.array([47.608013]), dt.datetime.utcnow())
            fh.write('sample shape=%s sample=%s\n' % (str(np.asarray(sample).shape), str(sample)))
        except Exception:
            fh.write('EXCEPTION during sampler call:\n')
            fh.write(traceback.format_exc())
    except Exception:
        fh.write('EXCEPTION during get_wind_fn:\n')
        fh.write(traceback.format_exc())

print('wrote', logp)
