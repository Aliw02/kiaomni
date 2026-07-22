import urllib.request
import zipfile
import json
import os
from pathlib import Path

def download_and_extract():
    OUT_DIR = Path('test_lb_download')
    OUT_DIR.mkdir(exist_ok=True)

    LB_TASK_FILES = {
        'qasper':          'qasper_e.jsonl',
        'hotpotqa':        'hotpotqa_e.jsonl',
        'multifieldqa_en': 'multifieldqa_en_e.jsonl',
    }
    LB_DATA_ZIP_URL = 'https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip'
    
    zpath = OUT_DIR / 'data.zip'
    
    if not zpath.exists():
        print(f'Downloading data.zip to {zpath}...')
        try:
            urllib.request.urlretrieve(LB_DATA_ZIP_URL, zpath)
            print('Download complete.')
        except Exception as e:
            print(f'Error downloading: {e}')
            return
    else:
        print('data.zip already exists, skipping download.')

    print('Extracting target files...')
    try:
        with zipfile.ZipFile(zpath) as zf:
            all_members = zf.namelist()
            for task, fname in LB_TASK_FILES.items():
                found = False
                for member in all_members:
                    if member.endswith(fname):
                        target = OUT_DIR / fname
                        with zf.open(member) as source, open(target, 'wb') as f:
                            f.write(source.read())
                        print(f'  Extracted: {fname}')
                        found = True
                        break
                if not found:
                    print(f'  Warning: {fname} not found in zip.')
    except Exception as e:
        print(f'Error extracting: {e}')
        return

    print('\nVerification:')
    for fname in LB_TASK_FILES.values():
        p = OUT_DIR / fname
        if p.exists():
            try:
                lines = [l for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
                if lines:
                    sample = json.loads(lines[0])
                    print(f'{fname}: {len(lines)} samples, keys={list(sample.keys())}')
                else:
                    print(f'{fname}: File is empty.')
            except Exception as e:
                print(f'{fname}: Error reading file - {e}')
        else:
            print(f'{fname}: File does not exist.')

if __name__ == '__main__':
    download_and_extract()
