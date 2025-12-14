import importlib
import inspect

mod = importlib.import_module('emergent.fish_passage.tests.test_fatigue')
print('module', mod)
fn = getattr(mod, 'test_calc_battery_basic')
print('calling', fn)
try:
    fn()
    print('test function completed without assertion')
except AssertionError as e:
    print('AssertionError from test:', e)
except Exception as e:
    print('Exception from test:', e)
