"""Export the outer-results table from its actual page of the finished paper."""
import math
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pdf = root / "output/pdf/weight-space-programming.pdf"
layout = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
pages = [i for i, text in enumerate(layout.split("\f"), 1) if re.search(r"Table 4:\s*Outer extrapolation", text)]
if len(pages) != 1:
    raise ValueError("Expected exactly one outer-result table.")
page = str(pages[0])
bbox = subprocess.check_output(["pdftotext","-bbox-layout","-f",page,"-l",page,str(pdf),"-"],text=True)
entries = re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', bbox)
caption = next(float(y) for _, y, _, _, word in entries if word=="4:")
words = [(float(x),float(y),float(r),float(b)) for x,y,r,b,_ in entries if caption-.5 <= float(y) < 720]
left = min(w[0] for w in words)-9
top = min(w[1] for w in words)-8
right = max(w[2] for w in words)+9
bottom = max(w[3] for w in words)+11
scale = 4
subprocess.run(["pdftoppm","-f",page,"-l",page,"-r","288","-x",str(math.floor(left*scale)),
                "-y",str(math.floor(top*scale)),"-W",str(math.ceil((right-left)*scale)),
                "-H",str(math.ceil((bottom-top)*scale)),"-png","-singlefile",str(pdf),
                str(root/"output/table-4-extrapolation")],check=True)
