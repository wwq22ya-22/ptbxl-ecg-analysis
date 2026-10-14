"""Check local MD links/tables and preserve pre-study historical tables."""
from pathlib import Path
from urllib.parse import unquote
from collections import Counter
import re, json, zipfile, hashlib

ROOT = Path('D:/MyProject/college/ecg1')
OUT = ROOT/'outputs/unet_teacher_20261010'
files = [ROOT/'README.md', ROOT/'data/README.md', *sorted((ROOT/'md').glob('*.md'))]
bad, links, anchors, tables = [], 0, 0, 0
for file in files:
    text = file.read_text(encoding='utf-8')
    ids = re.findall(r'<a id="([^"]+)">', text)
    assert len(ids)==len(set(ids)) and text.count('```')%2==0, file
    for label, raw in re.findall(r'\[([^\]]+)\]\((<?[^\n)]+>?)\)',text):
        target=raw.strip('<>'); links+=1
        if re.match(r'(https?://|mailto:)',target): continue
        path, sep, fragment = target.partition('#')
        resolved=file if not path else (file.parent/unquote(path)).resolve()
        if not resolved.exists(): bad.append([str(file),label,'missing target',target]); continue
        if fragment:
            anchors+=1
            if f'id="{fragment}"' not in resolved.read_text(encoding='utf-8'):
                bad.append([str(file),label,'missing anchor',target])
    lines=text.splitlines()
    for i,line in enumerate(lines):
        if re.match(r'^\|[- :|]+\|$',line):
            tables+=1; j=i+1
            while j<len(lines) and lines[j].startswith('|'):
                if lines[j].count('|')!=line.count('|'): bad.append([str(file),j+1,'column count'])
                j+=1
preserved={}
with zipfile.ZipFile(OUT/'markdown_before_unet.zip') as z:
    for name in ['md/各模型整体与逐类指标总表.md','md/STM32部署与串口验证.md','md/Chapman_冻结模型外部验证总结.md']:
        old=Counter(x for x in z.read(name).decode('utf-8').splitlines() if x.startswith('|'))
        new=Counter(x for x in (ROOT/name).read_text(encoding='utf-8').splitlines() if x.startswith('|'))
        assert not old-new, name
        preserved[name]=sum(old.values())
report={'passed':not bad,'links_checked':links,'explicit_anchor_links_checked':anchors,'tables_checked':tables,
        'historical_table_rows_preserved':preserved,'problems':bad,
        'documents':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(OUT/'markdown_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
assert not bad
