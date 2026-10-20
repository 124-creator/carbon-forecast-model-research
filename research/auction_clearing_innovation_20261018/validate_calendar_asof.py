"""Check consequential calendar-vintage boundaries without reading price data."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from parse_calendar_vintages import select_asof_plan


def main():
    root = Path(__file__).resolve().parent
    path = root / 'local_only/calendar_vintages_development.csv'
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) > 0
    assert all(2017 <= int(r['calendar_year']) <= 2023 for r in rows)
    assert {r['platform'] for r in rows} == {'CAP', 'DE', 'PL', 'NI'}
    assert all(int(r['plannedvolume']) > 0 and int(r['plannedvolume']) % 500 == 0 for r in rows)
    assert all((datetime.fromisoformat(r['revision_date']) + timedelta(days=1)).date().isoformat()
               == r['revision_available_date'] for r in rows)
    assert len({(r['vintage_id'], r['scheduleddate'], r['platform']) for r in rows}) == len(rows)

    # These dates are tied to observed official history discontinuities, not to
    # model performance. This catches inappropriate forward filling or early use.
    expected = {
        (2017, '2016-11-25'): None,
        (2017, '2016-11-26'): '2016-11-25',
        (2019, '2019-06-25'): '2019-05-16',
        (2019, '2019-06-26'): None,
        (2019, '2019-07-11'): None,
        (2019, '2019-07-12'): '2019-07-11',
        (2020, '2020-03-17'): '2019-12-12',
        (2020, '2020-03-18'): None,
        (2020, '2020-07-01'): None,
        (2020, '2020-07-02'): '2020-07-01',
        (2023, '2022-07-28'): None,
        (2023, '2022-07-29'): '2022-07-28',
        (2023, '2023-06-21'): '2022-07-28',
        (2023, '2023-06-22'): '2023-06-21',
    }
    boundary_checks = []
    for (year, origin), revision in expected.items():
        selected = select_asof_plan(rows, origin, year)
        found = sorted({r['revision_date'] for r in selected})
        assert found == ([] if revision is None else [revision]), (year, origin, found)
        assert all(r['scheduleddate'] > origin and r['revision_available_date'] <= origin
                   and (not r['valid_until_exclusive'] or origin < r['valid_until_exclusive'])
                   for r in selected)
        boundary_checks.append({'calendar_year': year, 'origin': origin,
                                'expected_revision': revision, 'selected_rows': len(selected), 'passed': True})

    # The August 2018 revision deletes five provisional German auctions after
    # contract expiry. A latest-row-per-event join would incorrectly revive them.
    before = [r for r in select_asof_plan(rows, '2018-08-14', 2018)
              if r['platform'] == 'DE' and r['scheduleddate'] > '2018-11-14']
    after = [r for r in select_asof_plan(rows, '2018-08-15', 2018)
             if r['platform'] == 'DE' and r['scheduleddate'] > '2018-11-14']
    assert len(before) == 5 and not after

    result = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'rows': len(rows), 'vintages': len({r['vintage_id'] for r in rows}),
        'read_files': [str(path)], 'price_data_read': False,
        'boundary_checks': boundary_checks,
        'deleted_2018_german_auctions': {'before_revision': len(before), 'after_revision': len(after), 'passed': True},
        'all_checks_passed': True,
    }
    destination = root / 'local_only/CALENDAR_ASOF_VALIDATION.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'all_checks_passed': True, 'rows': len(rows), 'vintages': result['vintages'],
                      'boundary_cases': len(boundary_checks), 'output': str(destination)}))


if __name__ == '__main__':
    main()
