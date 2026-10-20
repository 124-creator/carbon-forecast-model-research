from pathlib import Path
import pymupdf
p = Path(__file__).resolve().parent / 'sources'
for f in p.glob('*.pdf'):
    text = '\n'.join(page.get_text() for page in pymupdf.open(f))
    f.with_suffix('.extracted.txt').write_text(text, encoding='utf-8')
    print('\nSOURCE', f.name, '\n', text[:1800])
    terms = ['revision', 'orthogon', 'parallel', 'interpolation', 'moment', 'github']
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if any(t in line.lower() for t in terms)]
    for i in hits[:14]: print(' | '.join(lines[max(0, i-1):i+3]))
