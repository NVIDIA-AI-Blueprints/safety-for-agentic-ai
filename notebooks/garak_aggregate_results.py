# pip install bs4
import argparse
import os
import csv
from bs4 import BeautifulSoup
from pathlib import Path

# REPORTS_ROOT = os.path.join(
#     os.path.dirname(__file__),
#     "reports/r1_llama_8b_test"
# )
# OUTPUT_CSV = os.path.join(
#     os.path.dirname(__file__),
#     "reports_aggregated/garak_aggregated.csv"
# )


def extract_probe_section(panel):
    # probe
    h3 = panel.find('h3', class_=lambda c: c and c.startswith('defcon'))
    probe = ""
    if h3 and 'probe:' in h3.text:
        probe = h3.text.split('probe:')[1].split('-')[0].strip()
    # detector and aggregate_defcon
    h4 = panel.find('h4', class_=lambda c: c and c.startswith('defcon'))
    detector = ""
    aggregate_defcon = ""
    if h4:
        p = h4.find('p', class_='left')
        if p and 'detector:' in p.text:
            detector = p.text.split('detector:')[1].strip()
        dc_span = h4.find('span', class_=lambda c: c and c.startswith('defcon') and 'dc' in c)
        if dc_span and 'DC:' in dc_span.text:
            aggregate_defcon = dc_span.text.split('DC:')[1].strip()
    # pass_rate and pass_rate_defcon
    pass_rate = ""
    pass_rate_defcon = ""
    abs_score_p = panel.find('span', string='absolute score:')
    if abs_score_p:
        b = abs_score_p.find_next('b')
        if b:
            pass_rate = b.text.split('%')[0].strip() + "%" if '%' in b.text else b.text.strip()
        dc = abs_score_p.find_next('span', class_=lambda c: c and c.startswith('defcon') and 'dc' in c)
        if dc and 'DC:' in dc.text:
            pass_rate_defcon = dc.text.split('DC:')[1].strip()
    # z_score, z_score_status, z_score_defcon
    z_score = ""
    z_score_status = ""
    z_score_defcon = ""
    rel_score_p = panel.find('span', string='relative score (Z):')
    if rel_score_p:
        b = rel_score_p.find_next('b')
        if b:
            zscore_text = b.text.strip()
            if ' ' in zscore_text:
                z_score = zscore_text.split(' ')[0]
            if '(' in zscore_text and ')' in zscore_text:
                z_score_status = zscore_text.split('(')[1].split(')')[0]
        dc = rel_score_p.find_next('span', class_=lambda c: c and c.startswith('defcon') and 'dc' in c)
        if dc and 'DC:' in dc.text:
            z_score_defcon = dc.text.split('DC:')[1].strip()
    return {
        "probe": probe,
        "detector": detector,
        "pass_rate": pass_rate,
        "z_score": z_score,
        "z_score_status": z_score_status,
        "z_score_defcon": z_score_defcon,
        "pass_rate_defcon": pass_rate_defcon,
        "aggregate_defcon": aggregate_defcon
    }

def extract_from_html(html_file):
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    # Find all probe panels
    results = []
    found = 0
    for button in soup.find_all('button', class_=lambda c: c and 'accordion' in c.split()):
        panel = button.find_next('div', class_='panel')
        if panel:
            h3 = panel.find('h3', class_=lambda c: c and c.startswith('defcon'))
            if h3:
                found += 1
                row = extract_probe_section(panel)
                if row['probe']:
                    results.append(row)
    return results

def find_all_reports(reports_root):
    html_files = []
    for root, dirs, files in os.walk(reports_root):
        for file in files:
            if file.startswith('garak.') and file.endswith('.report.html'):
                html_files.append(os.path.join(root, file))
    return html_files

def create_csv(reports_root: str,
               output_csv: str):
    COLUMNS = [
        "probe",
        "detector",
        "pass_rate",
        "z_score",
        "z_score_status",
        "z_score_defcon",
        "pass_rate_defcon",
        "aggregate_defcon"
    ]

    all_rows = []
    html_files = find_all_reports(reports_root)
    for html_file in html_files:
        rows = extract_from_html(html_file)
        all_rows.extend(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Aggregated {len(all_rows)} rows from {len(html_files)} reports into {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description here.")

    # Add arguments
    parser.add_argument("--reports_root", type=str, required=True, help="Path to reports root")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output file")
    args = parser.parse_args()
    
    create_csv(reports_root=args.reports_root, output_csv=args.output_csv)