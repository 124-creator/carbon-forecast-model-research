"""Read literal NBS article tables; do not infer a missing publication date."""
import datetime as dt
import html
from html.parser import HTMLParser
import re


SPECS = ('普通混煤（4500大卡）', '山西大混（5000大卡）', '山西优混（5500大卡）')


class Article(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta, self.rows, self.visible = {}, [], []
        self.cells, self.cell = None, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta':
            self.meta[(attrs.get('name') or '').lower()] = attrs.get('content') or ''
        if tag == 'tr':
            self.cells = []
        if tag in ('td', 'th') and self.cells is not None:
            self.cell = []

    def handle_data(self, data):
        self.visible.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None and self.cells is not None:
            self.cells.append(' '.join(''.join(self.cell).split()))
            self.cell = None
        if tag == 'tr' and self.cells is not None:
            self.rows.append(self.cells)
            self.cells = None


def parse_article(text):
    page = Article()
    page.feed(text)
    publication = page.meta.get('pubdate', '').strip()
    title = page.meta.get('articletitle', '')
    source = page.meta.get('contentsource', '').strip()
    if not publication:
        header = re.search(r'<div\b[^>]*class=[\'"]detail-title-des[\'"][^>]*>(.*?)</h2>', text, re.S)
        literal = re.search(r'20\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}', header[1]) if header else None
        if literal is not None:
            publication = literal[0]
            source = '国家统计局' if '国家统计局' in header[1] else ''
            h1 = re.search(r'<h1\b[^>]*>(.*?)</h1>', text, re.S)
            title = html.unescape(re.sub(r'<[^>]+>', '', h1[1])).strip() if h1 else ''
    if not publication:
        return None
    match = re.fullmatch(r'(20\d{2})[/\-](\d{2})[/\-](\d{2})\s+(\d{2}):(\d{2})(?::\d{2})?', publication)
    if not match:
        raise ValueError('Unsupported literal publication format')
    if source != '国家统计局':
        raise ValueError('Official source identity missing')
    if publication not in ' '.join(page.visible):
        raise ValueError('Publication meta has no exact visible counterpart')
    published = dt.date(*map(int, match.group(1, 2, 3)))
    if not re.search(r'201[89]年|202[01]年', title):
        raise ValueError('Article is outside the allowed observation years')
    selected = {}
    for cells in page.rows:
        if len(cells) < 3 or cells[1] != '吨':
            continue
        key = re.sub(r'\s+', '', cells[0])
        if key not in SPECS:
            continue
        value = float(cells[2].replace(',', ''))
        if not 0 < value < float('inf'):
            raise ValueError('Nonpositive or nonfinite source price')
        if key in selected and selected[key][1] != value:
            raise ValueError('Conflicting duplicated literal source row')
        selected[key] = (cells[0], value)
    if set(selected) != set(SPECS):
        raise ValueError('Incomplete literal thermal vector')
    return dict(title=title, published=published.isoformat(),
                available_date=(published + dt.timedelta(days=1)).isoformat(),
                labels=[selected[key][0] for key in SPECS],
                values=[selected[key][1] for key in SPECS])
