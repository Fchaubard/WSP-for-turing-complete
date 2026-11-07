"""Build a self-contained release and a SHA-256 receipt, excluding caches and prior ZIPs."""
import hashlib,json,sys,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'output';z=out/'research-bundle.zip'
allowed_output={'output/pdf/weight-space-programming.pdf','output/optimizer-comparison.html','output/optimizer-runs.csv','output/preprint-source.zip','output/ten-task-exact-and-rmse.csv','output/all-exact-and-rmse.zip','output/nonconvex-endpoints.csv','output/table-4-extrapolation.png','output/gradient-descent-and-lbfgs.csv'}
allowed_output.update({'output/pdf/weight-space-programming-2026-11-06-iterative.pdf','output/pdf/weight-space-programming-2026-11-06-one-shot.pdf','output/table-4-one-shot.png','output/table-4-iterative-extrapolation.png','output/iterative-wsp-endpoints.csv','output/iterative-wsp-by-size.csv','output/iterative-wsp-rounds.csv','output/iterative-wsp-symmetry.csv'})
allowed_output.update({'output/pdf/weight-space-programming-2026-11-06-charts.pdf',
                       'output/figure-1-extrapolation-error.png','output/table-3-task-charts.pdf',
                       'output/table-3-task-charts.html','output/table-4-charts.png'})
allowed_output.update({'output/pdf/weight-space-programming-2026-11-06-architecture.pdf',
                       'output/all-optimizer-raw-rmse-ranking.csv','output/optimizer-rmse-task-contributions.csv',
                       'output/figure-1-all-optimizers.png','output/architecture-choice-models.csv',
                       'output/architecture-choice-errors.csv','output/architecture-choice-costs.csv'})
allowed_output.update({
    'output/pdf/weight-space-programming-2026-11-06-learning-curves.pdf',
    'output/iterative-wsp-training-curves.png','output/iterative-wsp-task-training-curves.png',
    'output/iterative-wsp-proposal-curves.png','output/iterative-wsp-learning-curves.csv',
    'output/iterative-wsp-learning-summary.csv','output/iterative-wsp-proposal-curves.csv',
    'output/function-prediction-curves.png','output/function-prediction-curves.csv'})
allowed_output.add('output/pdf/weight-space-programming-2026-11-07.pdf')
allowed_output.add('output/pdf/weight-space-programming-public.pdf')
files=[]
for p in sorted(root.rglob('*')):
    rel=p.relative_to(root)
    if not p.is_file() or any(n.startswith('.') or n=='__pycache__' for n in rel.parts):continue
    if rel.parts[0]=='output' and rel.as_posix() not in allowed_output:continue
    files.append(p)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest={'files':{p.relative_to(root).as_posix():digest(p) for p in files}}
with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as arc:
    for p in files:arc.write(p,'weight-space-programming/'+p.relative_to(root).as_posix())
    arc.writestr('weight-space-programming/FILE_HASHES.json',json.dumps(manifest,indent=1)+'\n')
with zipfile.ZipFile(z) as arc:
    assert arc.testzip() is None
    for key,v in manifest['files'].items():
        assert hashlib.sha256(arc.read('weight-space-programming/'+key)).hexdigest()==v
receipt={'sha256':{k:(digest(root/k)) for k in sorted(allowed_output|{'output/research-bundle.zip'})},'archived_artifacts':len(files)}
(out/'checksums.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(len(files),'files verified;',round(z.stat().st_size/1048576,2),'MiB')
