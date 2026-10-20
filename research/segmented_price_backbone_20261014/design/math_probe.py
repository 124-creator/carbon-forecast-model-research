"""Exact finite-state mathematical checks, not empirical market results."""
import itertools, json
from pathlib import Path
states = list(itertools.product((-1., 1.), repeat=2))
def moment(b, a=1.):
    return sum((a*u + b*z)*z for z,u in states)/len(states)
def loss(b, a=1.):
    return moment(b,a)**2
results = {}
results['honest_nonzero_news_revision_has_zero_moment'] = moment(0.) == 0.
results['predictable_old_information_revision_detected'] = moment(.5) == .5
results['raw_revision_penalty_would_punish_honest_news'] = sum(u*u for z,u in states)/len(states) == 1.
# Shared target at old origin t-1 and horizon h+1 equals new origin t and horizon h.
results['same_target_indexing'] = all((t-1)+(h+1)==t+h for t in range(4,9) for h in range(1,4))
# A common additive forecast bias disappears from the difference: identification is incomplete.
results['common_bias_is_invisible'] = all((3+u+7)-(3+7)==u for z,u in states)
eps=1e-6; b=.5
fd=(loss(b+eps)-loss(b-eps))/(2*eps)
results['analytic_gradient_matches_finite_difference'] = abs(fd-2*b)<1e-8
# A weighted within-origin separable SSE has no mixed second derivative;
# the squared revision of one paired observation is (new-old)^2, mixed derivative -2.
def paired(new,old): return (new-old)**2
mixed=(paired(1+eps,2+eps)-paired(1+eps,2-eps)-paired(1-eps,2+eps)+paired(1-eps,2-eps))/(4*eps**2)
results['cross_origin_coupling_not_separable_h_weighting'] = abs(mixed+2)<1e-3
payload={'scope':'finite-state algebra only; no market/backbone training','tests':results,'passed':sum(results.values()),'total':len(results),'finite_difference_gradient':fd,'mixed_derivative':mixed}
Path(__file__).with_name('math_probe_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(json.dumps(payload,indent=2))
assert all(results.values())
