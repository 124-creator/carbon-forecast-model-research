"""Fixed ordinary-model search; not a novel tuning algorithm."""
import itertools
import math

def choose(records):
    schedule = plan()
    grid = configurations()
    expected = {(family, config['id'], cut) for family in schedule['families']
                for config in grid for cut in schedule['search_cuts']}
    found = {}
    for record in records:
        key = (record['family'], record['config_id'], record['cut'])
        value = float(record['selection_score'])
        if key not in expected or key in found or not math.isfinite(value) or value < 0:
            raise ValueError('Duplicate, unexpected or nonfinite tuning trial')
        found[key] = value
    if set(found) != expected:
        raise ValueError('Incomplete tuning cannot select a config')
    result = {}
    for family in schedule['families']:
        scores = {config['id']: sum(found[(family, config['id'], cut)] for cut in schedule['search_cuts'])
                  / len(schedule['search_cuts']) for config in grid}
        selected = min(scores, key=lambda key: (scores[key], key))
        result[family] = dict(config_id=selected, selection_score=scores[selected], all_scores=scores)
    return result

def configurations() -> list[dict]:
    return [dict(id=f'c{i:02d}', hidden=h, lr=lr, weight_decay=wd)
            for i, (h, lr, wd) in enumerate(itertools.product((8, 16), (.001, .003), (0., .001)))]

def plan() -> dict:
    return dict(families=['gru_dense', 'gru_profile', 'gru_rank6', 'lstm_dense'],
                search_cuts=[359, 419], search_seed=17, search_span=20,
                market_origins=list(range(459, 930)), market_seeds=[17, 29],
                market_cuts=[459 + 40 * i for i in range(12)], default_id='c07',
                max_epochs=60, patience=12, max_total_fits=268,
                fit_budget_seconds=1800, audit_budget_seconds=300)
