"""Audit recorded prompt instructions and the geometry of selected square networks."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "focus/iterative_eval"))
from measure import datasets, measures, frozen_input

EXTRAPOLATION = "Among similarly fitting solutions, aim for exact outputs on unseen inputs across the stated domain."
REVISION = "Seek a rule that also generalizes to examples not shown."
TRAIN = "Your primary objective is to minimize the training loss on the supplied observations"
DEV = "Smaller development loss improves the candidate; among identical development losses smaller training loss improves it."

def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def geometry(weights):
    width = (len(weights) - 2) // 3
    values = list(map(lambda x: Fraction.from_float(float(x)), weights))
    a, b = values[:2]
    slopes = values[2:2+width]
    offsets = values[2+width:2+2*width]
    readout = values[2+2*width:]
    knots = sorted(-offset/slope for slope, offset, output in zip(slopes, offsets, readout)
                   if slope and output)
    left_slope = a + sum(slope * output for slope, output in zip(slopes, readout) if slope < 0)
    right_slope = a + sum(slope * output for slope, output in zip(slopes, readout) if slope > 0)
    left_intercept = b + sum(offset * output for slope, offset, output in zip(slopes, offsets, readout) if slope < 0)
    right_intercept = b + sum(offset * output for slope, offset, output in zip(slopes, offsets, readout) if slope > 0)
    result = dict(knot_min=float(min(knots)), knot_max=float(max(knots)), active_hinges=len(knots),
                  hinges_outside_training_interval=sum(abs(knot)>3 for knot in knots),
                  left_tail_slope=float(left_slope), right_tail_slope=float(right_slope),
                  left_tail_intercept=float(left_intercept), right_tail_intercept=float(right_intercept))
    for x in [-10, 10]:
        value = a*x+b+sum(output*max(Fraction(0), slope*x+offset)
                          for slope, offset, output in zip(slopes, offsets, readout))
        result["real_prediction_at_"+("minus10" if x<0 else "plus10")] = float(value)
    return result

def tex_literal(text):
    replacement = {"\\":r"\textbackslash{}", "{":r"\{", "}":r"\}", "_":r"\_",
                   "&":r"\&", "%":r"\%", "#":r"\#", "$":r"\$",
                   "^":r"\textasciicircum{}", "~":r"\textasciitilde{}"}
    out = []
    for character in text:
        if character == "\n":
            out.append("\\par\n")
        elif character in replacement:
            out.append(replacement[character])
        else:
            out.append(character)
        if character in ",;[]":
            out.append(r"\allowbreak{}")
    return "".join(out)

def main():
    frozen_input()
    prompts = []
    cases = []
    original_events = Counter()
    iterative_events = Counter()
    for source in ["objective", "arithmetic"]:
        for case in json.loads((ROOT/"focus"/source/"manifest.json").read_text()):
            case = dict(case, source=source)
            cases.append(case)
            stem = ROOT/"focus"/source/"trials"/case["id"]
            path = stem.with_suffix(".prompt.txt")
            text = path.read_text()
            extrapolation = EXTRAPOLATION if source=="objective" else "Among equally good training fits, seek exact generalization to new inputs throughout the stated domain."
            train = TRAIN if source=="objective" else "The primary objective for each variant is to minimize TRAINING LOSS"
            assert extrapolation in text and train in text
            meta = json.loads(stem.with_suffix(".meta.json").read_text())
            original_events.update(meta["completed_items"])
            prompts.append(dict(kind="one_shot",task=case["task"],seed=case["seed"],condition=case["arm"],
                                round=0,path=str(path.relative_to(ROOT)),sha256=digest(path),
                                contains_extrapolation=True,primary_objective="training loss",
                                requested_output="numeric arrays only"))
            for number in range(1,4):
                revision=ROOT/"focus/iterative/trials"/(case["id"]+f"-r{number}.prompt.txt")
                if not revision.exists():
                    continue
                contents=revision.read_text()
                assert text in contents and extrapolation in contents and REVISION in contents and DEV in contents
                meta = json.loads(revision.with_name(revision.name.replace(".prompt.txt",".meta.json")).read_text())
                iterative_events.update(meta["completed_items"])
                prompts.append(dict(kind="iterative",task=case["task"],seed=case["seed"],condition=case["arm"],
                                    round=number,path=str(revision.relative_to(ROOT)),sha256=digest(revision),
                                    contains_extrapolation=True,primary_objective="development loss, then training loss",
                                    requested_output="numeric arrays only; explanations forbidden"))
    assert len(cases)==60 and len(prompts)==221
    assert original_events==Counter(agent_message=60) and iterative_events==Counter(agent_message=161)
    write_csv(HERE/"prompts.csv",prompts)
    rows = []
    for case in cases:
        if case["task"] != "square":
            continue
        saved=json.loads((ROOT/"focus/iterative/trajectories"/(case["id"]+".json")).read_text())
        state=saved["states"]["16"]
        for method, key in [("one_shot","initial"),("iterative","weights")]:
            p=np.asarray(state[key])
            row=dict(task="square",size=16,seed=case["seed"],condition=case["arm"],method=method,
                     trajectory=str(Path("focus/iterative/trajectories")/(case["id"]+".json")),
                     **geometry(p))
            for split, data in datasets("square",case["seed"],None).items():
                metrics=measures("square",16,p,*data)
                row[split+"_rmse"]=metrics["raw_rmse"]
                row[split+"_scaled_loss"]=metrics["split_loss"]
            rows.append(row)
    write_csv(HERE/"square_geometry.csv",rows)
    quotes=dict(original_primary=TRAIN,original_extrapolation=EXTRAPOLATION,iterative_generalization=REVISION,
                iterative_acceptance=DEV,counts=dict(original=60,iterative=161),
                original_completed_items=dict(original_events),iterative_completed_items=dict(iterative_events))
    (HERE/"summary.json").write_text(json.dumps(quotes,indent=2)+"\n")

    # The shortest complete numeric-only sequence example exposes all supplied information.
    example = next(c for c in cases if c["id"]=="347817495f1c")
    assert example["seed"]==164 and example["arm"]=="examples"
    items = [("Original prompt",ROOT/"focus/objective/trials/347817495f1c.prompt.txt"),
             ("Original final response",ROOT/"focus/objective/trials/347817495f1c.reply.txt"),
             ("First revision prompt",ROOT/"focus/iterative/trials/347817495f1c-r1.prompt.txt"),
             ("First revision final response",ROOT/"focus/iterative/trials/347817495f1c-r1.reply.txt")]
    sections=[]
    for title, source in items:
        shutil.copy(source,HERE/source.name)
        sections += [r"\paragraph{"+title+r".}",
                     r"\begingroup\footnotesize\ttfamily\raggedright\setlength{\parindent}{0pt}",
                     tex_literal(source.read_text()),r"\par\endgroup"]
    (HERE/"literal_trace.tex").write_text("\n".join(sections)+"\n")
    print(json.dumps(quotes))
    for r in rows:
        if r["seed"]==164 and r["condition"]=="examples":
            print(r)

if __name__=="__main__":
    main()
