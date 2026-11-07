"""Closed-form coefficients. MPFR is used only while writing constants and scoring."""
from math import factorial
from fractions import Fraction
import gmpy2 as mp
from graph import Graph

# Compilers only use correctly rounded binary64 arithmetic in the resulting graph.
def two_sum(a,b):
    s = a+b
    v = s-a
    return s,(a-(s-v))+(b-v)

def split(a):
    t = a*(2.**27+1)
    hi = t-(t-a)
    return hi,a-hi

def two_product(a,b):
    ah,al = split(a)
    bh,bl = split(b)
    p = a*b
    return p,(((ah*bh-p)+ah*bl)+al*bh)+al*bl

class Precision:
    def __init__(self,g,double):
        self.g,self.double=g,double
        self.zero=g.literal(0)

    def from_wire(self,x):
        return (x,self.zero) if self.double else x

    def const(self,c):
        if isinstance(c,Fraction):
            c=mp.mpfr(mp.mpq(c.numerator,c.denominator))
        hi=float(c)
        if not self.double:
            return self.g.param(hi)
        return self.g.param(hi), self.g.param(float(c-mp.mpfr(hi)))

    def add(self,a,b):
        if not self.double: return a+b
        h,e=two_sum(a[0],b[0])
        return two_sum(h,(a[1]+b[1])+e)

    def multiply(self,a,b):
        if not self.double:return a*b
        h,e=two_product(a[0],b[0])
        return two_sum(h,(e+a[0]*b[1])+a[1]*b[0])

    def final(self,a):
        return a[0]+a[1] if self.double else a

    def horner(self,z,coefficients):
        nodes=[self.const(c) for c in coefficients]
        y=nodes[-1]
        for a in reversed(nodes[:-1]):
            y=self.add(self.multiply(y,z),a)
        return y

def sin_graph(degree,double=True,coefs=None):
    # Odd global polynomial, no phase information or envelope in the weights.
    g=Graph()
    ops=Precision(g,double)
    if coefs is None:
        coefs=[Fraction((-1)**j,factorial(2*j+1)) for j in range((degree+1)//2)]
    x=ops.from_wire(g.x)
    with mp.context(precision=220):
        poly=ops.horner(ops.multiply(x,x),coefs)
        ops.final(ops.multiply(x,poly))
    return g

def selfpower_graph(log_degree,exp_degree,double=True):
    g=Graph()
    ops=Precision(g,double)
    one=g.literal(1)
    # Indicators H_i equal 1 when x exceeds the dyadic boundary 2^i for all
    # binary64 x in [0,128]. Each above-boundary step is at least 2^(i-52).
    exponents=list(range(-64,7))
    hs=[]
    for k in exponents:
        above=((g.x-g.param(2.**k))*g.param(2.**(52-k))).relu()
        hs.append(one-(one-above).relu())
    masks=[a-b for a,b in zip(hs,hs[1:])]+[hs[-1]]
    scale=g.literal(0)
    exponent=g.literal(0)
    for k,mask in zip(exponents,masks):
        scale=scale+mask*g.param(2.**(-k))
        exponent=exponent+mask*g.param(k)
    with mp.context(precision=220):
        # Center mantissa in [1,2] about 3/2. The centering subtraction is exact
        # by Sterbenz. Decoding is shared across every polynomial order.
        z=ops.multiply(ops.from_wire(g.x*scale-g.param(1.5)),ops.const(Fraction(2,3)))
        logcoeff=[Fraction((-1)**j,j+1) for j in range(log_degree)]
        log1p=ops.multiply(z,ops.horner(z,logcoeff))
        offset=ops.add(ops.const(mp.log(mp.mpfr(1.5))),
                       ops.multiply(ops.from_wire(exponent),ops.const(mp.log(mp.mpfr(2)))))
        logx=ops.add(offset,log1p)
        t=ops.multiply(ops.from_wire(g.x*g.param(2.**-10)),logx)
        exp=ops.horner(t,[Fraction(1,factorial(j)) for j in range(exp_degree+1)])
        for _ in range(10):
            exp=ops.multiply(exp,exp)
        # On [0,2^-64], x^x rounds to 1 by a margin of more than tenfold.
        result=ops.final(exp)
        hs[0]*result+(one-hs[0])
    return g

def gate(g,z,disabled):
    # Conditional copy of a signed value of magnitude <= 32.
    block=disabled*32
    return (z-block).relu()-(g.literal(0)-z-block).relu()

def static_input_check(g):
    """Every multiplication has an input independent of x. Thus all are affine maps of x-values."""
    static=[]
    for op,*args in g.nodes:
        if op=="x": flag=False
        elif op in ("weight","literal"): flag=True
        else:
            if op=="mul" and not any(static[i] for i in args):
                raise ValueError("Dynamic product in claimed affine/ReLU graph")
            flag=all(static[i] for i in args)
        static.append(flag)
    return dict(dynamic_nodes=sum(not s for s in static)-1,
                relus=sum(op[0]=="relu" and not s for op,s in zip(g.nodes,static)))

def rot_coefficients(k):
    """Use only 0<=a<=1 Taylor coefficients, then double the angle in weight space."""
    a=mp.mpfr(2)**min(0,k)
    s=a
    c=mp.mpfr(1)
    ts,tc=s,c
    for j in range(1,55):
        ts*=-a*a/((2*j)*(2*j+1))
        tc*=-a*a/((2*j-1)*(2*j))
        s+=ts
        c+=tc
    for _ in range(max(0,k)):
        s,c=2*s*c,c*c-s*s
    return s,c

def relu_sin_graph(last,double=True):
    """Binary-angle rotations, conditional copies, and compensated affine arithmetic."""
    g=Graph()
    ops=Precision(g,double)
    one,zero=g.literal(1),g.literal(0)
    def select(flag,new,old):
        if double:
            return tuple(gate(g,a,one-flag)+gate(g,b,flag) for a,b in zip(new,old))
        return gate(g,new,one-flag)+gate(g,old,flag)
    def minus(v):
        if double:return tuple(zero-w for w in v)
        return zero-v

    rem=g.x.relu()+(zero-g.x).relu()
    s,c=ops.from_wire(zero),ops.from_wire(one)
    with mp.context(precision=220):
        for k in range(4,last-1,-1):
            pivot=g.param(2.**k)
            # predecessor of 2^k is 2^k-2^(k-53).
            take=(one-((pivot-rem)*g.param(2.**(53-k))).relu()).relu()
            rem=rem-take*pivot
            sk,ck=map(ops.const,rot_coefficients(k))
            sr=ops.add(ops.multiply(s,ck),ops.multiply(c,sk))
            cr=ops.add(ops.multiply(c,ck),minus(ops.multiply(s,sk)))
            s,c=select(take,sr,s),select(take,cr,c)
        out=ops.final(s)
    cutoff=g.param(2.**-26)
    gain=g.param(2.**78)
    pos=one-(one-((g.x-cutoff)*gain).relu()).relu()
    neg=one-(one-((zero-g.x-cutoff)*gain).relu()).relu()
    # For |x|<=2^-26 the correctly rounded binary64 sin(x) equals x.
    gate(g,out,one-pos)-gate(g,out,one-neg)+gate(g,g.x,pos+neg)
    static_input_check(g)
    return g
