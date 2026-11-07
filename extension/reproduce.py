"""Replay the extended study offline. Use --refit for optimizer experiments in a fresh extraction."""
from pathlib import Path
import argparse,os,subprocess,sys
root=Path(__file__).resolve().parent
def job(name,*args):
    f=root/name
    print('Running',name,*args,flush=True)
    subprocess.run([sys.executable,str(f),*args],cwd=f.parent,check=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--refit',action='store_true');ap.add_argument('--hard-cases',action='store_true',help='fetch pinned public sine and tanh input corpora')
    a=ap.parse_args()
    os.environ.update(OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1')
    if a.refit:
        job('writer/compare.py','fit')
        job('optimizer/run.py');job('optimizer/report.py');job('convex/run.py')
        job('counters/train.py');job('smooth_optimizer/bench.py','selfpower');job('smooth_optimizer/bench.py','tanh');job('recurrent/run.py','--jobs','5')
        job('recurrent/run.py','--jobs','5','--steps','5000','--polish','200','--output','longer')
        job('recurrent/calibrated.py');job('recurrent/summarize.py')
    job('writer/compare.py','score');job('writer/report.py')
    job('counters/run.py');job('counters/writer.py','score');job('audit/lattice.py')
    job('synthesis/experiment.py','score');job('synthesis/paper_table.py')
    job('synthesis/repair/score.py');job('synthesis/repair/report.py')
    job('synthesis/literal_square_certificate.py');job('recurrent/audit.py');job('scalar/replay.py')
    if a.hard_cases:
        job('scalar/hard_sin.py','--fetch');job('synthesis/challenge.py');job('tanh/verify.py')
    job('square_full/verify.py');job('audit/square_certificate_review.py');job('smooth_optimizer/replay.py')
    job('check_release.py');job('optimizer_brief.py')
