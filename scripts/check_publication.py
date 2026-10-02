"""Read-only release checks. Reports locations and issue types, never matched values."""
from pathlib import Path
import ast
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.venv', '__pycache__', '.ipynb_checkpoints', 'outputs', 'build', 'dist'}
PATTERNS = {
    'private-key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY',
    'access-key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'service-token': r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,})',
    'cloud-resource': r'arn:aws:|s3://|\d{12}\.dkr\.ecr\.',
    'personal-path': r'/Users/[^\s]+|/home/[^\s]+|/Volumes/[^\s]+',
    'private-conversation': r'https?://(?:chatgpt\.com|chat\.openai\.com)/c/|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',
}

def texts(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from texts(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in ('image/png', 'image/jpeg'):
                yield from texts(item)

def main():
    issues=[]; checked=0
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if any(p in SKIP or p.endswith('.egg-info') for p in rel.parts):
            continue
        if path.is_symlink():
            issues.append((str(rel), 'symlink')); continue
        if not path.is_file():
            continue
        if path.name.startswith('.env') or path.suffix in ('.pem','.key'):
            issues.append((str(rel),'sensitive filename'))
        if path == Path(__file__).resolve():
            continue
        if path.suffix == '.png':
            continue  # Pixels require visual review; no OCR claim.
        try:
            content=path.read_text()
        except UnicodeError:
            issues.append((str(rel),'unexpected binary')); continue
        checked+=1
        if path.suffix=='.py':
            ast.parse(content, filename=str(rel))
        if path.suffix=='.ipynb':
            notebook=json.loads(content)
            content='\n'.join(texts(notebook))
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    if cell.get('execution_count') is None:
                        issues.append((str(rel),'unexecuted cell'))
                    if any(o.get('output_type')=='error' for o in cell.get('outputs',[])):
                        issues.append((str(rel),'saved execution error'))
        for label,pattern in PATTERNS.items():
            if re.search(pattern,content):
                issues.append((str(rel),label))
    for path,label in issues:
        print(path+': '+label)
    print(f'Checked {checked} text artifacts; {len(issues)} issues. Image pixels require separate review.')
    return bool(issues)
if __name__=='__main__':
    sys.exit(main())
