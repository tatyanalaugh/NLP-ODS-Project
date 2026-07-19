import pandas as pd
import csv
import re
from typing import List, Dict, Tuple, Set


def remove_stress_marks(text: str) -> str:
    result = []
    for ch in text:
        if ord(ch) in (0x0301, 0x0300, 0x0302, 0x0308, 0x0306):
            continue
        result.append(ch)
    return ''.join(result)


def parse_narcoterms(filepath: str) -> List[Dict]:
    """
    Parses a file with drug slang.
    Returns a list of dictionaries with keys:
        normalized_term — one normalized form of the term
        description — definition (after the dash)
    """
    entries = []

    line_re = re.compile(
        r'^[•\-\s]*'
        r'(.+?)'
        r'\s*[—–]\s*'
        r'(.+)$',
        re.UNICODE
    )

    with open(filepath, encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            line = re.sub(r'\[\d+\]', '', line)
            line = line.strip()

            m = line_re.match(line)
            if not m:
                continue

            terms_raw, description_raw = m.group(1), m.group(2)

            terms_raw = remove_stress_marks(terms_raw)
            description_raw = remove_stress_marks(description_raw).strip()

            raw_terms = re.split(r',', terms_raw)

            for raw_term in raw_terms:
                term = raw_term.strip()

                if not term:
                    continue

                term = re.sub('[«»„"‘’•]', '', term).strip()
                term = re.sub(r'^[-–—]\s*', '', term).strip()

                if not term:
                    continue

                entries.append({
                    'normalized_term': term,
                    'description': description_raw,
                })

    return entries


def save_to_csv(entries: List[Dict], output_path: str) -> None:
    fieldnames = ['normalized_term', 'description']
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)


def load_csv(filepath: str) -> Tuple[List[Dict], List[str]]:
    rows = []
    fieldnames = []

    with open(filepath, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(dict(row))

    return rows, list(fieldnames)


def merge_csv_files(
    source_path: str,
    target_path: str,
    key_column: str = 'normalized_term',
) -> Dict:
    source_rows, source_fields = load_csv(source_path)
    target_rows, target_fields = load_csv(target_path)

    target_keys: Set[str] = {
        row[key_column].strip().lower()
        for row in target_rows
        if key_column in row
    }

    merged_fields = list(target_fields)
    for field in source_fields:
        if field not in merged_fields:
            merged_fields.append(field)

    new_rows: List[Dict] = []
    new_terms: List[str] = []

    for row in source_rows:
        term = row.get(key_column, '').strip()
        if term.lower() not in target_keys:
            full_row = {field: row.get(field, '') for field in merged_fields}
            new_rows.append(full_row)
            new_terms.append(term)
            target_keys.add(term.lower())

    stats = {
        'total_source': len(source_rows),
        'total_target_before': len(target_rows),
        'added': len(new_rows),
        'total_target_after': len(target_rows) + len(new_rows),
        'new_terms': new_terms,
    }

    if not new_rows:
        print("New terms are not found.")
        return stats

    with open(target_path, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writerows(new_rows)

    return stats


drug_seed_terms = [
    "adderall", "alprazolam", "amphetamine", "clonazepam", "cocaine", "concerta", "daytrana",
    "dilaudid", "ecstasy", "fentanyl", "flunitrazepam", "ghb", "heroin", "hydrocodone",
    "hydromorphone", "ketalar", "ketamine", "khat", "klonopin", "lorcet", "lsd",
    "marijuana", "mdma", "mescaline", "methamphetamine", "methylphenidate", "molly", "morphine",
    "norco", "opium", "oxaydo", "oxycodone", "oxycontin", "pcp", "percocet",
    "peyote", "phencyclidine", "promethazine", "ritalin", "rohypnol", "roxicodone", "steroids",
    "suboxone", "u-47700", "vicodin", "xanax"
]


if __name__ == '__main__':
    INPUT_TXT  = 'Narcoterms.txt'
    SOURCE_CSV = 'narcoterms.csv'
    TARGET_CSV = 'illegal_terms_dictionary_edit.csv'

    # PARSING 
    entries = parse_narcoterms(INPUT_TXT)
    save_to_csv(entries, SOURCE_CSV)

    # COMPARING AND EDITING 
    stats = merge_csv_files(
        source_path=SOURCE_CSV,
        target_path=TARGET_CSV,
        key_column='normalized_term',
    )

    print(f"\n{'='*50}")
    print(f"  Rows in source:              {stats['total_source']:>6}")
    print(f"  Rows in target file:     {stats['total_target_before']:>6}")
    print(f"  New terms added:         {stats['added']:>6}")
    print(f"{'='*50}")

    # A BIT MORE TERMS  
    new_rows = pd.DataFrame({
        'term': drug_seed_terms,
        'normalized_term': drug_seed_terms
    })
    new_rows.to_csv(TARGET_CSV, mode='a', header=False, index=False)
    print(f"\nДобавлено {len(drug_seed_terms)} seed terms в файл {TARGET_CSV}")


