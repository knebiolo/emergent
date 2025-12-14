import numpy as np
from emergent.fish_passage import fatigue
from emergent.salmon_abm.sockeye import _calc_battery_numba as legacy

def run():
    battery = np.array([1.0, 0.5, 0.2, 0.0])
    per_rec = np.array([0.1, 0.1, 0.1, 0.1])
    ttf = np.array([10.0, 5.0, 2.0, 1.0])
    mask_sustained = np.array([True, False, False, False])
    dt = 0.5

    print('inputs:')
    print('battery', battery)
    print('per_rec', per_rec)
    print('ttf', ttf)
    print('mask_sustained', mask_sustained)
    print('dt', dt)

    expected = legacy(battery.copy(), per_rec.copy(), ttf.copy(), mask_sustained.copy(), dt)
    print('\nlegacy returned', expected)

    got = fatigue.calc_battery(battery.copy(), per_rec.copy(), ttf.copy(), mask_sustained.copy(), dt)
    print('numpy calc returned', got)

    # reproduce internal steps from fatigue.calc_battery
    b = battery.copy()
    b[mask_sustained] += per_rec[mask_sustained]
    print('\nafter sustained add b=', b)
    mask_non = ~mask_sustained
    print('mask_non', mask_non)
    ttf0 = ttf[mask_non] * b[mask_non]
    print('ttf0', ttf0)
    ttf1 = ttf0 - dt
    print('ttf1', ttf1)
    safe = ttf0 != 0
    print('safe', safe)
    ratio = np.ones_like(ttf0)
    ratio[safe] = np.maximum(0.0, ttf1[safe] / ttf0[safe])
    print('ratio', ratio)
    b[mask_non] = b[mask_non] * ratio
    np.clip(b, 0.0, 1.0, out=b)
    print('final b', b)

if __name__ == '__main__':
    run()
