"""Build the preprint in a disposable directory and add primary-source links."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

paper = Path(__file__).resolve().parent
root = paper.parent
out = root/'output/pdf/weight-space-programming.pdf'
endpoint_dir = root / 'focus/endpoints'
sources = [*sorted(paper.glob('*.tex')), paper / 'neurips_2026.sty', *sorted(paper.glob('*.pdf'))]
sources.extend(endpoint_dir / name for name in ('figure1.pdf', 'figure1.tex', 'paired_table.tex', 'endpoint_tables.tex'))
# Preserve the original figure and data files alongside this edition's generated tables.
iteration = root / "focus/iterative_tables"
iter_sources = {f"iterative_{name}.tex": iteration / f"{name}.tex" for name in
                ("figure1", "paired_table", "representative", "endpoint_tables")}
iter_sources["iterative_figure.pdf"] = iteration / "iterative_figure.pdf"
iter_sources["task_panels.tex"] = iteration / "task_panels.tex"
for index in range(1, 6):
    name = f"task_panels_{index}.pdf"
    iter_sources[name] = iteration / name
for folder_name in ("holistic_plots", "prompt_audit", "learning_plots", "main_visuals"):
    folder = root / "focus" / folder_name
    for source in [*sorted(folder.glob("*.tex")), *sorted(folder.glob("*.pdf"))]:
        iter_sources[source.name] = source
architecture_folder = root / "focus/architecture_search"
for source in [*sorted(architecture_folder.glob("*.tex")), *sorted(architecture_folder.glob("*.pdf")), *sorted(architecture_folder.glob("architecture_*.txt"))]:
    iter_sources[source.name] = source
with tempfile.TemporaryDirectory(prefix='weight-space-paper-') as temp:
    build = Path(temp)
    for source in sources:
        shutil.copy(source, build / source.name)
    for name, source in iter_sources.items():
        shutil.copy(source, build / name)
    bib = (root/'literature/references.bib').read_text()
    (build/'references.bib').write_text(bib)
    for args in (['pdflatex','-interaction=nonstopmode','-halt-on-error','paper.tex'],['bibtex','paper']):
        proc=subprocess.run(args,cwd=build,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if proc.returncode: raise RuntimeError(proc.stdout[-6000:])
    # plain.bst omits URLs. Link each reference label to its recorded source.
    links = {}
    for entry in re.split(r'\n(?=@)', bib):
        if not entry.strip(): continue
        key = re.search(r'@\w+\{([^,]+),',entry)
        url = re.search(r'\burl\s*=\s*\{([^}]+)\}',entry)
        if key and url: links[key[1]]=url[1]
    bbl = build/'paper.bbl'
    def link_entry(m):
        key,body=m.group(1),m.group(2)
        if key not in links: return m.group(0)
        return '\\bibitem{'+key+'}'+body.rstrip()+' '+r'\href{'+links[key]+r'}{\textnormal{[source]}}'+'\n\n'
    bbl.write_text(re.sub(r'\\bibitem\{([^}]+)\}([\s\S]*?)(?=\\bibitem|\\end\{thebibliography\})',link_entry,bbl.read_text()))
    for _ in range(2):
        proc=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','paper.tex'],cwd=build,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if proc.returncode: raise RuntimeError(proc.stdout[-6000:])
    log=(build/'paper.log').read_text(errors='replace')
    for line in log.splitlines():
        if 'Overfull' in line or 'undefined' in line: print(line)
    if 'There were undefined' in log: raise RuntimeError('Unresolved reference')
    out.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(build/'paper.pdf',out)
    source_archive = root/'output/preprint-source.zip'
    with zipfile.ZipFile(source_archive, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [source.name for source in sources] + list(iter_sources) + ['paper.bbl','references.bib']:
            archive.write(build/name, name)
    print(out)
    print(source_archive)
