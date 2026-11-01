"""Read-only structural validation of the separately authored presentation package."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import fitz
from pptx import Presentation

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    'CV5570_Final_Presentation.pptx', 'CV5570_Final_Presentation.pdf',
    'speaker_notes.md', 'presentation_timing.md', 'LIVE_DEMO_GUIDE.md',
    'DEMO_VIDEO_SCRIPT.md', 'Q_AND_A_PREPARATION.md', 'REHEARSAL_CHECKLIST.md',
]


def verify():
    report = {'files': {}, 'checks': {}, 'limitations': [
        'Structural checks complement, but do not replace, per-slide visual review and browser interaction tests.',
        'Historical experiment test receipts are not a newly rerun research test suite.',
    ]}
    for name in REQUIRED:
        p = ROOT / name
        report['files'][name] = {'exists': p.is_file(), 'bytes': p.stat().st_size if p.is_file() else 0}
    report['checks']['required_files_present'] = all(v['bytes'] > 0 for v in report['files'].values())
    pptx = ROOT / REQUIRED[0]
    pdf = ROOT / REQUIRED[1]
    if pptx.is_file():
        prs = Presentation(pptx)
        report['slides'] = len(prs.slides)
        report['aspect_ratio'] = round(prs.slide_width / prs.slide_height, 5)
        report['checks']['widescreen'] = abs(prs.slide_width / prs.slide_height - 16 / 9) < .01
        report['checks']['slide_count_reasonable'] = 12 <= len(prs.slides) <= 18
        notes = []
        outside = []
        for i, slide in enumerate(prs.slides, 1):
            text = slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ''
            notes.append({'slide': i, 'notes_characters': len(text.strip())})
            for shape in slide.shapes:
                if shape.left < -9144 or shape.top < -9144 or shape.left + shape.width > prs.slide_width + 9144 or shape.top + shape.height > prs.slide_height + 9144:
                    outside.append({'slide': i, 'shape': shape.name})
        report['notes'] = notes
        report['checks']['every_slide_has_notes'] = all(n['notes_characters'] > 80 for n in notes)
        report['out_of_bounds_shapes'] = outside
        with zipfile.ZipFile(pptx) as z:
            bad = z.testzip()
            for name in z.namelist():
                if name.endswith('.xml') or name.endswith('.rels'):
                    ET.fromstring(z.read(name))
            report['checks']['pptx_zip_and_xml_valid'] = bad is None
    if pdf.is_file():
        with fitz.open(pdf) as doc:
            report['pdf_pages'] = len(doc)
            report['checks']['pdf_matches_deck'] = len(doc) == report.get('slides')
            report['checks']['every_pdf_page_has_text'] = all(len(page.get_text().strip()) > 40 for page in doc)
    q = ROOT / 'Q_AND_A_PREPARATION.md'
    if q.is_file():
        report['checks']['at_least_twenty_questions'] = sum(line.startswith('### ') for line in q.read_text(encoding='utf-8').splitlines()) >= 20
    report['checks']['root_launcher_exists'] = (ROOT.parents[1] / 'run_live_demo.ps1').is_file()
    report['checks']['root_demo_guide_exists'] = (ROOT.parents[1] / 'LIVE_DEMO_GUIDE.md').is_file()
    report['passed'] = all(report['checks'].values())
    (ROOT / 'delivery_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(verify())
