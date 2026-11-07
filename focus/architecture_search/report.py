"""Produce publication tables and verbatim public construction explanations from frozen replies."""
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from evaluate import parse
TASKS={'square':'Square','sin':'Sine','selfpower':'Positive self-power','tanh':'Hyperbolic tangent',
       'add':'Addition','subtract':'Subtraction','sum':'Running sum','sort':'Sort four','fib':'Fibonacci','tri':'Triangular'}

def read(name):
    with (HERE/name).open(newline='') as f:return list(csv.DictReader(f))
def latex(text):
    result=[]
    symbols={'−':'-','²':r'\textsuperscript{2}','³':r'\textsuperscript{3}','¹':r'\textsuperscript{1}',
             '⁴':r'\textsuperscript{4}','⁵':r'\textsuperscript{5}','⁷':r'\textsuperscript{7}','⁹':r'\textsuperscript{9}',
             '₀':r'\textsubscript{0}','₁':r'\textsubscript{1}','₂':r'\textsubscript{2}',
             '≤':r'\ensuremath{\leq}','±':r'\ensuremath{\pm}','·':r'\ensuremath{\cdot}',
             '×':r'\ensuremath{\times}','→':r'\ensuremath{\to}','–':'--'}
    escaped={'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}',
             '\\':r'\textbackslash{}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    for ch in str(text):result.append(symbols.get(ch,escaped.get(ch,ch)))
    return ''.join(result)
def num(value):
    x=float(value)
    if x==0:return '0'
    if abs(x)>=.001 and abs(x)<10000:return f'{x:.3g}'
    a,b=f'{x:.2e}'.split('e');return '$'+a.rstrip('0').rstrip('.')+r'\times10^{'+str(int(b))+'}$'
def span(values):
    a,b=min(values),max(values)
    return str(a) if a==b else str(a)+'--'+str(b)

def main():
    audit=json.loads((HERE/'audit.json').read_text());assert audit['models']==30
    for name,h in audit['result_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h
    models=read('models.csv');scores=read('scores.csv');costs=read('costs.csv')
    out={(r['task'],r['seed']):r for r in scores if r['split']=='outer'}
    work={(r['task'],r['seed']):r for r in costs if r['split']=='outer'}
    cases=json.loads((HERE/'manifest.json').read_text())
    summaries=[]
    for t in TASKS:
        group=[r for r in models if r['task']==t];valid=[r for r in group if r['status']=='valid']
        rows=[out[t,r['seed']] for r in valid];cost=[work[t,r['seed']] for r in valid]
        summaries.append(dict(task=t,attempted=3,valid=len(valid),fully_exact=sum(int(r['exact_count'])==int(r['n_examples']) for r in rows),
                              median_raw_rmse_valid=median(float(r['raw_rmse']) for r in rows),
                              coefficients_range=span([int(r['coefficients']) for r in valid]),
                              flops_max_range=span([int(r['flops_max']) for r in cost]),
                              workspace_max_range=span([int(r['workspace_bytes_max']) for r in cost])))
    with (HERE/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows(summaries)
    main_text=(r'Across 30 independent architecture proposals from input/output pairs, 28 were executable and 21 matched every original extrapolation output. '
               r'All three proposals were exact for square, hyperbolic tangent, addition, subtraction, running sum, Fibonacci and triangular numbers. '
               r'The sine proposals had median outer RMSE $7.79\times10^{-16}$; sorting errors were about $5.2\times10^{-16}$. '
               r'Positive self-power had one executable proposal with RMSE $1.68\times10^4$ and two invalid proposals. '
               r'Astra chose feedforward arithmetic or activation graphs and recurrent networks; no proposal chose convolution or attention. '
               r'Square used one allowed multiplication, zero stored coefficients and 16 bytes of live activation storage. Its graph specification still occupied 109 serialized bytes. '
               r'Operator choice explains this reduction relative to a ReLU-only network. The trials establish observed implementations and resource costs, with no global architecture optimum claimed.\n')
    (HERE/'architecture_main_results.tex').write_text(main_text.replace('\\n','\n'))
    table=[r'\begin{table}[htbp]\centering\small',r'\caption{Architecture selection from examples alone. Each task has three first-response trials. RMSE is the median over executable proposals, with invalid trials retained in the valid-count column. Resources give ranges across executable proposals at the longest outer input. The arithmetic count excludes comparisons and transcendental calls, which are reported separately in the appendix.}\label{tab:architecture_search}',r'\setlength{\tabcolsep}{2.5pt}',r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',r'Target & Valid & Exact sets & RMSE & Coefficients & FLOPs & Workspace bytes\\\midrule']
    for r in summaries:
        table.append(' & '.join([TASKS[r['task']],str(r['valid'])+'/3',str(r['fully_exact'])+'/3',num(r['median_raw_rmse_valid']),r['coefficients_range'],r['flops_max_range'],r['workspace_max_range']])+r'\\')
    table += [r'\bottomrule\end{tabular}\end{table}']
    (HERE/'architecture_main_table.tex').write_text('\n'.join(table)+'\n')
    appendix=[r'\subsection{Architecture discovery protocol and resources}\label{app:architecture_search}',
              r'Each request supplies the original twelve labeled examples or nine integer indices, input/output shapes and the deployment domain. It supplies no task name, formula, initial weights or prescribed architecture. Extrapolation is the primary objective. Resource minimization is secondary. There are three independent first responses per task, and no response is selected using test data. Every graph is frozen before evaluation. The two invalid responses remain failed trials.',
              r'Allowed operators are affine maps, rectified linear units (ReLU), hyperbolic tangent, sigmoid, elementwise addition and multiplication, static tensor indexing and concatenation, reshape, transpose, matrix multiplication, softmax and one-dimensional convolution. Optional recurrence repeats a shared cell. This library permits multilayer perceptrons, recurrent neural networks, convolutional neural networks and attention networks, as well as multiplicatively gated arithmetic graphs. It bounds every proposal at 64 nodes and 4,096 scalar coefficients. A tanh node directly implements hyperbolic tangent, and a multiplication node directly implements square. Both operations therefore affect capacity and the appropriate interpretation of comparisons with the fixed ReLU architectures.',
              r'Each floating-point operation (FLOP) is a scalar addition, multiplication or division. Dot products count every multiply and accumulation, including fixed zeros and ones. Comparisons and transcendental evaluations are listed separately. The coefficient count includes every stored scalar constant, bias, matrix entry and initial-state value. Each coefficient takes eight bytes. Peak workspace counts live binary64 input, state, intermediate and output buffers under the interpreter\textquotesingle s dependency-based lifetime rule. Graph outputs are materialized. Python, allocator and matrix-library workspaces are excluded. Serialized executable JavaScript Object Notation (JSON) reports the full graph-file size, including topology and numeric literals. This serialization metric is separate from runtime memory. For recurrence, costs use one example starting from the initial state; the listed worst case is index 64 or a six-element stream. A comparison of costs therefore requires the same operations, precision, input length and memory accounting.',
              r'Zero stored coefficients is the elementary lower bound on a nonnegative coefficient count. The square, tanh and addition proposals attain that bound in this operator library. The square graph also uses one arithmetic multiplication. A nonconstant scalar output requires at least one stored output scalar under the materialized-output convention, or eight bytes in binary64. The observed 16-byte square workspace includes input and output. These simple bounds provide no certificate that the other proposed architectures minimize coefficients, workspace or arithmetic.',
              r'The three sine constructions illustrate competing resource objectives. Their coefficient counts were 19, 14 and 10, while peak live workspaces were 32, 40 and 88 bytes and arithmetic counts were 32, 27 and 27. All three produced the same measured outer RMSE. Reducing coefficient storage therefore did not minimize workspace in this sample.',
              r'The independently drawn confirmation set uses a post-freeze random seed. It contains 2,048 new central and 2,048 new outer inputs per stochastic task. The integer tasks have fixed held-out indices 17--32 and 33--64. All fifteen exact stochastic-task proposals also matched every independently drawn confirmation output. The six exact integer-sequence proposals concern fixed indices. All three running-sum proposals also matched all 2,048 length-32 central streams. The complete scores are in the data archive.',
              r'\begin{table}[p]\centering\footnotesize',r'\caption{Every architecture proposal on the original outer test. Exact is the number of whole examples matching the reference. FLOPs, comparisons, transcendental calls and workspace use the largest input in the split. Dashes indicate a rejected executable graph. Weight bytes count stored coefficients once, including recurrent initial state. Graph bytes measure the serialized graph file, including numeric literals. Live bytes measure peak live tensor storage; all three sizes are in bytes.}\label{tab:architecture_all}',r'\setlength{\tabcolsep}{2.3pt}',r'\begin{tabular}{@{}lr r r r r r r r r@{}}\toprule',r'Target & Seed & Exact & RMSE & \shortstack{Weight\\bytes} & \shortstack{Graph\\bytes} & FLOPs & \shortstack{Compar-\\isons} & \shortstack{Function\\calls} & \shortstack{Live\\bytes}\\\midrule']
    for t in TASKS:
        for m in [r for r in models if r['task']==t]:
            r=out[t,m['seed']];c=work.get((t,m['seed']))
            if c:
                fields=[TASKS[t],m['seed'],r['exact_count']+'/'+r['n_examples'],num(r['raw_rmse']),m['coefficient_bytes'],m['serialized_model_bytes'],c['flops_max'],c['comparisons_max'],c['transcendental_max'],c['workspace_bytes_max']]
            else:fields=[TASKS[t],m['seed'],'invalid',r'$\infty$','--','--','--','--','--','--']
            appendix.append(' & '.join(fields)+r'\\')
        appendix.append(r'\addlinespace[2pt]')
    appendix += [r'\bottomrule\end{tabular}\end{table}',r'\clearpage',r'\subsection{Complete architecture-selection prompts and final answers}\label{app:architecture_prompts}',
                 r'The following two complete prompts use the first data seed, 164, for square and sine. They are literal request texts, with line wrapping added for printing. The archive supplies all 30 prompts and replies. The network language and objective were fixed before dispatch. Each response contained a public construction summary and a proposed mathematical certificate. These are model-supplied final-answer explanations; no hidden internal reasoning trace was recorded. Numerical outcomes and interpreter resource counts above are independently measured. Model-supplied certificates are reproduced as claims and are not automatically promoted to proved results.']
    appendix.append(r'''\lstset{literate={−}{{$-$}}1 {²}{{$^2$}}1 {³}{{$^3$}}1 {¹}{{$^1$}}1 {⁴}{{$^4$}}1 {⁵}{{$^5$}}1 {⁷}{{$^7$}}1 {⁹}{{$^9$}}1 {₀}{{$_0$}}1 {₁}{{$_1$}}1 {₂}{{$_2$}}1 {≤}{{$\leq$}}1 {±}{{$\pm$}}1 {·}{{$\cdot$}}1 {×}{{$\times$}}1 {→}{{$\to$}}1 {–}{{--}}1}''')
    for t in ('square','sin'):
        c=next(r for r in cases if r['task']==t and r['seed']==164)
        for suffix in ('prompt','reply'):
            dest=f'architecture_{t}_164_{suffix}.txt'
            text=(HERE/'trials'/f"{c['id']}.{suffix}.txt").read_text()
            if suffix=='prompt':
                text='\n'.join(line.replace('],[','],\n[') if line.startswith('[[') else line for line in text.split('\n'))
            (HERE/dest).write_text(text)
            appendix += [r'\paragraph{'+TASKS[t]+' '+('request' if suffix=='prompt' else 'complete final answer')+r'.}',
                         r'\lstinputlisting[basicstyle=\ttfamily\scriptsize,breaklines=true,breakatwhitespace=false,columns=fullflexible,keepspaces=true]{'+dest+'}']
    appendix += [r'\subsection{All public construction summaries}\label{app:architecture_summaries}',
                 r'Each paragraph reproduces the inferred rule, implementation explanation and certificate fields returned by one request. Unicode mathematical characters are typeset equivalently. Failed executable graphs are labeled below. Their explanations remain part of the response record.']
    for t in TASKS:
        for c in [r for r in cases if r['task']==t]:
            reply=parse(HERE/'trials'/f"{c['id']}.reply.txt")
            status=next(r['status'] for r in models if r['id']==c['id'])
            appendix.append(r'\paragraph{'+TASKS[t]+', seed '+str(c['seed'])+', '+status+r'.}')
            for field,label in [('family','Family'),('inferred_rule','Inferred rule'),('construction_summary','Construction summary'),('certificate','Model-supplied certificate')]:
                appendix.append(r'\textbf{'+label+r'.} '+latex(reply.get(field,''))+'\n')
    (HERE/'architecture_appendix.tex').write_text('\n\n'.join(appendix)+'\n')
    meta=[]
    for c in cases:
        p=HERE/'trials'/f"{c['id']}.meta.json"
        if p.exists():meta.append(json.loads(p.read_text()))
    counts=dict(requests=len(meta),recorded_tool_events=sum(sum(v for k,v in r['completed_items'].items() if k!='agent_message') for r in meta),
                summed_request_seconds=sum(r['elapsed_seconds'] for r in meta),output_tokens=sum(sum(u.get('output_tokens',0) for u in r.get('usage',[])) for r in meta))
    (HERE/'report.json').write_text(json.dumps(dict(summary=summaries,collection=counts),indent=1)+'\n')
    print(json.dumps(counts))

if __name__=='__main__':main()
