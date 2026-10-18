"""Parse official EEX auction schedule vintages without reading realized prices.

Only the first overview/calendar worksheet is used, never country-allocation
details. Publication dates come from each workbook's Document History. The
2017 initial workbook is dated by the later official history that explicitly
identifies its initial publication. Unknown dates/volumes/platforms are rejected.

As-of joins must select a complete annual vintage before selecting future events.
Joining the latest row separately for each auction date would resurrect deleted
auctions. Known material revisions without an archived workbook invalidate the
old plan from their announcement date + one natural day until the next archive.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import warnings

import openpyxl

DEFAULT_SOURCE = Path(__file__).resolve().parents[3] / 'research_planning/data/source_documents/calendar_vintages_2012_2023'
ARCHIVE_URL = 'https://www.eex.com/fileadmin/EEX/Downloads/Trading/Calendar/Auction_Calendar/20260217_History_EUA_Primary_Auction_Calendar_2011-2025.zip'
FIELDS = ['calendar_year', 'vintage_id', 'revision_date', 'revision_available_date',
          'valid_until_exclusive', 'scheduleddate', 'platform', 'platform_raw',
          'auction_name', 'plannedvolume', 'product_code', 'trading_period',
          'bidding_window', 'volume_is_explicit', 'volume_is_indicative',
          'publication_date_is_explicit', 'publication_evidence',
          'source_path', 'source_sha256', 'source_sheet', 'source_row', 'source_url']


def as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ('%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y'):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                pass
    return None


def platform_of(name):
    text = name.lower()
    if 'northern ireland' in text or 'northern irish' in text or ' nir' in text:
        return 'NI'
    if 'german' in text:
        return 'DE'
    if 'polish' in text or 'poland' in text:
        return 'PL'
    if 'eu primary' in text or 'eu auction' in text or 'common auction' in text:
        return 'CAP'
    raise ValueError(f'Unknown auction platform: {name}')


def parse_history(rows):
    in_history = False
    events = []
    for row_number, row in enumerate(rows, 1):
        text = ' '.join(str(v) for v in row if v is not None)
        if 'document history' in text.lower():
            in_history = True
            continue
        if not in_history:
            continue
        if 'notification' in text.lower() or any(str(v).strip().lower() == 'product code' for v in row):
            break
        event_date = as_date(row[0])
        if event_date:
            reason = ' '.join(str(v) for v in row[2:] if v not in (None, ''))
            version = str(row[1] or '')
            if reason and version:
                events.append({'date': event_date.isoformat(), 'version': version,
                               'reason': reason, 'source_row': row_number})
    return events


def volume_of(value):
    indicative = isinstance(value, str) and '*' in value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    elif isinstance(value, str):
        s = value.strip().replace('*', '').replace(' ', '')
        if not re.fullmatch(r'\d+(?:[.,]\d{3})*', s):
            raise ValueError(f'Unparsed planned volume: {value!r}')
        number = float(s.replace('.', '').replace(',', ''))
    else:
        raise ValueError(f'Missing planned volume: {value!r}')
    if number <= 0 or number != int(number) or int(number) % 500:
        raise ValueError(f'Planned volume is not a positive multiple of 500: {value!r}')
    return int(number), indicative


def load_workbook_metadata(path):
    year_match = re.search(r'/History_EUA_Primary_Auction_Calendar_(\d{4})/', str(path), re.I)
    if not year_match:
        raise ValueError('Calendar-year folder missing')
    calendar_year = int(year_match[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.worksheets[0]
    if 'detail' in sheet.title.lower():
        raise ValueError('First worksheet is a detail allocation sheet')
    # Some allocation sheets have spurious dimensions exceeding 1800 columns.
    # The overview uses at most 13; bound the read to its relevant first 20.
    rows = list(sheet.iter_rows(max_col=min(sheet.max_column, 20), values_only=True))
    workbook.close()
    return {'year': calendar_year, 'path': str(path.resolve()), 'sheet': sheet.title,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'history': parse_history(rows[:100]), 'rows': rows}


def select_asof_plan(rows, origin_date, calendar_year=None):
    """Return future events from one complete known vintage per calendar year.

An empty result may mean a missing-vintage interval. Consult coverage metadata;
do not fall back to a superseded schedule or future realized auction quantities.
"""
    origin = str(origin_date)[:10]
    eligible = [r for r in rows if r['revision_available_date'] <= origin
                and (calendar_year is None or int(r['calendar_year']) == calendar_year)]
    selected = {}
    for row in eligible:
        year = int(row['calendar_year'])
        key = row['revision_available_date']
        if year not in selected or key > selected[year][0]:
            selected[year] = key, row['vintage_id']
    return [r for r in eligible if selected[int(r['calendar_year'])][1] == r['vintage_id']
            and (not r['valid_until_exclusive'] or origin < r['valid_until_exclusive'])
            and r['scheduleddate'] > origin]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'local_only')
    parser.add_argument('--min-year', type=int, default=2017)
    parser.add_argument('--max-year', type=int, default=2023)
    args = parser.parse_args()
    if not 2012 <= args.min_year <= args.max_year <= 2023:
        raise ValueError('Only 2012–2023 development calendar years are authorized')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    books, rejected = [], []
    for path in sorted(args.source_root.rglob('*.xlsx')):
        match = re.search(r'/History_EUA_Primary_Auction_Calendar_(\d{4})/', str(path), re.I)
        if not match or not args.min_year <= int(match[1]) <= args.max_year:
            continue
        try:
            books.append(load_workbook_metadata(path))
        except Exception as error:
            rejected.append({'path': str(path), 'reason': str(error)})

    all_events = {}
    for book in books:
        for event in book['history']:
            key = book['year'], event['date'], event['version'], event['reason']
            all_events[key] = {**event, 'year': book['year'], 'evidence_path': book['path']}

    outputs, vintages = [], []
    for book in books:
        history = book['history']
        if history:
            current = max(history, key=lambda e: (e['date'], e['source_row']))
            pub_date = date.fromisoformat(current['date'])
            evidence = f"Workbook Document History, {book['sheet']}!A{current['source_row']}:C{current['source_row']}, version {current['version']}: {current['reason']}"
        elif book['year'] == 2017 and Path(book['path']).name == '20161125_Auction_Calendar_2017.xlsx':
            corroborating = [e for e in all_events.values() if e['year'] == 2017
                             and e['date'] == '2016-11-25' and 'initial publication' in e['reason'].lower()]
            if not corroborating:
                rejected.append({'path': book['path'], 'reason': 'Initial publication not corroborated'})
                continue
            current = corroborating[0]
            pub_date = date(2016, 11, 25)
            evidence = f"Official later Document History explicitly dates initial version 1.0 to 2016-11-25: {current['evidence_path']} row {current['source_row']}"
        else:
            rejected.append({'path': book['path'], 'reason': 'No explicit Document History; filename date is insufficient'})
            continue
        available = pub_date + timedelta(days=1)
        vintage_id = f"{book['year']}_{pub_date.isoformat()}_{book['sha256'][:12]}"
        parsed, errors = [], []
        header = None
        for index, row in enumerate(book['rows']):
            norm = [str(v).strip().lower() if v is not None else '' for v in row]
            if 'product code' in norm and 'volume' in norm:
                auction_columns = [i for i, v in enumerate(norm) if v in ('auction name', 'auctionname')]
                if auction_columns:
                    header = {'product': norm.index('product code'), 'volume': norm.index('volume'),
                              'name': auction_columns[0], 'period': norm.index('trading period'),
                              'window': norm.index('call trading period')}
                continue
            if not header or str(row[header['product']]).strip() != 'T3PA':
                continue
            try:
                scheduled = as_date(row[0])
                if scheduled is None and index + 1 < len(book['rows']):
                    scheduled = as_date(book['rows'][index + 1][0])
                if scheduled is None:
                    raise ValueError('Explicit scheduled date absent from row and next date row')
                if scheduled.year != book['year']:
                    continue
                auction_name = str(row[header['name']] or '').strip()
                platform = platform_of(auction_name)
                volume, footnote_marker = volume_of(row[header['volume']])
                indicative = False
                # Annual MSR preliminary schedules explicitly mark September–December.
                notification = ' '.join(str(v) for rr in book['rows'][:100] for v in rr if isinstance(v, str))
                if 'indicative' in notification.lower() and 'september to december' in notification.lower() and scheduled.month >= 9:
                    indicative = True
                parsed.append({'calendar_year': book['year'], 'vintage_id': vintage_id,
                               'revision_date': pub_date.isoformat(), 'revision_available_date': available.isoformat(),
                               'valid_until_exclusive': '', 'scheduleddate': scheduled.isoformat(),
                               'platform': platform, 'platform_raw': auction_name,
                               'auction_name': auction_name, 'plannedvolume': volume, 'product_code': 'T3PA',
                               'trading_period': str(row[header['period']] or ''),
                               'bidding_window': str(row[header['window']] or ''),
                               'volume_is_explicit': True, 'volume_is_indicative': indicative,
                               'publication_date_is_explicit': True, 'publication_evidence': evidence,
                               'source_path': book['path'], 'source_sha256': book['sha256'],
                               'source_sheet': book['sheet'], 'source_row': index + 1, 'source_url': ARCHIVE_URL})
            except Exception as error:
                errors.append({'row': index + 1, 'reason': str(error)})
        if not parsed or errors:
            rejected.append({'path': book['path'], 'reason': 'Incomplete vintage rejected', 'parsed_rows': len(parsed), 'errors': errors})
            continue
        unique_keys = {(r['scheduleddate'], r['platform']) for r in parsed}
        if len(unique_keys) != len(parsed):
            raise ValueError(f"Duplicate auction date/platform in {book['path']}")
        vintages.append({'calendar_year': book['year'], 'vintage_id': vintage_id,
                         'revision_date': pub_date.isoformat(), 'revision_available_date': available.isoformat(),
                         'row_count': len(parsed), 'source_path': book['path'], 'source_sha256': book['sha256'],
                         'publication_evidence': evidence})
        outputs.extend(parsed)

    # Conflicting source schedules sharing a stated revision date are not safely
    # dateable. Exclude the whole ambiguous group and retain its history event as
    # a gap boundary, instead of assigning a later file's contents to an old date.
    groups = {}
    for vintage in vintages:
        groups.setdefault((vintage['calendar_year'], vintage['revision_date']), []).append(vintage)
    ambiguous_ids = set()
    for key, group in groups.items():
        if len(group) > 1:
            for vintage in group:
                ambiguous_ids.add(vintage['vintage_id'])
                rejected.append({'path': vintage['source_path'],
                                 'reason': 'Ambiguous same-date source vintages excluded; no filename date inferred',
                                 'calendar_year': key[0], 'revision_date': key[1],
                                 'conflicting_sources': [v['source_path'] for v in group]})
    outputs = [r for r in outputs if r['vintage_id'] not in ambiguous_ids]
    vintages = [v for v in vintages if v['vintage_id'] not in ambiguous_ids]

    # A known missing substantive revision is a real gap, not forward-fill license.
    actual_dates = {(v['calendar_year'], v['revision_date']) for v in vintages}
    missing = []
    for event in all_events.values():
        if (event['year'], event['date']) in actual_dates:
            continue
        text = event['reason'].lower()
        documentation_only = 'correction of table header' in text or ('document history' in text and 'added' in text)
        missing.append({**event, 'documentation_only': documentation_only,
                        'available_date': (date.fromisoformat(event['date']) + timedelta(days=1)).isoformat()})
    material_events = [e for e in all_events.values() if not ('correction of table header' in e['reason'].lower()
                       or ('document history' in e['reason'].lower() and 'added' in e['reason'].lower()))]
    for row in outputs:
        future = [(date.fromisoformat(e['date']) + timedelta(days=1)).isoformat() for e in material_events
                  if e['year'] == row['calendar_year'] and e['date'] > row['revision_date']]
        future += [v['revision_available_date'] for v in vintages if v['calendar_year'] == row['calendar_year']
                   and v['revision_date'] > row['revision_date']]
        row['valid_until_exclusive'] = min(future) if future else ''
    for vintage in vintages:
        vintage['valid_until_exclusive'] = next(r['valid_until_exclusive'] for r in outputs if r['vintage_id'] == vintage['vintage_id'])

    outputs.sort(key=lambda r: (r['calendar_year'], r['revision_available_date'], r['scheduleddate'], r['platform']))
    destination = args.output_dir / 'calendar_vintages_development.csv'
    with destination.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(outputs)
    coverage = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'source_url': ARCHIVE_URL,
                'scope': [args.min_year, args.max_year], 'this_parser_read_confirmation_prices_or_calendar_contents': False,
                'availability_rule': 'Explicit publication date + 1 natural day; no same-day availability assumed',
                'publication_evidence_scope': 'Dates are explicit official calendar Document History assertions, including later-history corroboration for the initial 2017 vintage. They are not independently timestamped historical retrievals. Two 2023 vintages are additionally corroborated by dated official customer information. Current ZIP packaging date is never used as the historical publication date.',
                'selection_rule': 'Select full latest known annual vintage, then future dates; respect valid_until_exclusive and known gaps',
                'rows': len(outputs), 'vintages': sorted(vintages, key=lambda v: (v['calendar_year'], v['revision_date'])),
                'missing_history_versions': missing, 'rejected_files': rejected,
                'year_coverage': {str(year): {'vintages': sum(v['calendar_year'] == year for v in vintages),
                                  'rows_including_repeated_vintages': sum(r['calendar_year'] == year for r in outputs),
                                  'platforms': sorted({r['platform'] for r in outputs if r['calendar_year'] == year})}
                                  for year in range(args.min_year, args.max_year + 1)},
                'output_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
    (args.output_dir / 'CALENDAR_VINTAGE_COVERAGE.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(destination), 'rows': len(outputs), 'vintages': len(vintages),
                      'rejected_files': len(rejected), 'coverage': coverage['year_coverage'],
                      'missing_history_versions': [{'year': e['year'], 'date': e['date'], 'reason': e['reason'], 'documentation_only': e['documentation_only']} for e in missing]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
