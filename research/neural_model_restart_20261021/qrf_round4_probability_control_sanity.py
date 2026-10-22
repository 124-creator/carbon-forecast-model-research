# -*- coding: utf-8 -*-
"""Independent hand-calculated fallback checks; no model fit or real-data rerun."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
from qrf_round4_probability_control import replace_probability

P = Path(__file__).resolve().parent
TAG = 'qrf_round4_probability_control'

support = np.array([-1., 0., 2.], np.float32)
weights = np.array([[.2, 0., .8], [0., 1., 0.]], np.float32)
new_support, new, details = replace_probability(support, weights, np.array([.6, .4]))
assert np.array_equal(new_support, support)
assert np.allclose(new, [[.12, .4, .48], [.2, .6, .2]], atol=1e-7, rtol=0)
assert details['zero_mass_fallback_queries'] == 1
assert details['nonzero_mass_fallback_queries'] == 1

support2 = np.array([-1., 2.], np.float32)
new_support2, new2, details2 = replace_probability(support2, np.array([[.25, .75]], np.float32), np.array([.6]))
assert np.array_equal(new_support2, [-1., 0., 2.])
assert np.allclose(new2, [[.15, .4, .45]], atol=1e-7, rtol=0)
assert details2['dummy_zero_added']

result = dict(zero_mass_uniform_fallback=True, nonzero_mass_uniform_fallback=True,
              missing_zero_support_inserted=True, hand_calculated_probabilities_match=True,
              no_training=True)
(P/f'{TAG}_sanity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result))
