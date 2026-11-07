"""Static-weight ReLU tanh via hyperbolic angle addition and digit division."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]/'scalar'))
import gmpy2 as mp
from compile import Graph, Precision, static_input_check, two_sum, two_product


class ThreeWord:
    def __init__(self,g):
        self.g,self.z=g,g.literal(0)
    def from_wire(self,v):
        return (v,self.z,self.z)
    def const(self,v):
        hi=float(v)
        rem=v-mp.mpfr(hi)
        mid=float(rem)
        return self.g.param(hi),self.g.param(mid),self.g.param(float(rem-mp.mpfr(mid)))
    def add(self,a,b):
        hi,e=two_sum(a[0],b[0])
        mid,f=two_sum(a[1],b[1])
        mid,g=two_sum(mid,e)
        lo=(a[2]+b[2])+(f+g)
        mid,lo=two_sum(mid,lo)
        hi,e=two_sum(hi,mid)
        mid,lo=two_sum(e,lo)
        return hi,mid,lo
    def multiply(self,a,c):
        h0,e0=two_product(a[0],c[0])
        h1,e1=two_product(a[0],c[1])
        h2,e2=two_product(a[1],c[0])
        low=(a[2]*c[0]+a[1]*c[1])+a[0]*c[2]
        return self.add(self.add((h0,e0,self.z),(h1,e1,self.z)),(h2,e2,low))
    def final(self,a):
        return a[0]+(a[1]+a[2])


def tanh_graph(bits, double, guard_round=False, triple=False):
    g=Graph()
    op=ThreeWord(g) if triple else Precision(g,double)
    zero,one=g.literal(0),g.literal(1)
    def parts(fn,*v):
        return tuple(fn(*words) for words in zip(*v)) if double else fn(*v)
    def minus(v):
        return parts(lambda a:zero-a,v)
    def scale(v,c):
        return parts(lambda a:a*c,v)
    def gated(flag,v):
        off=(one-flag)*128.
        return parts(lambda a:(a-off).relu()-(zero-a-off).relu(),v)
    def high(v):
        return v[0] if double else v
    rem=g.x.relu()+(zero-g.x).relu()
    n,d=op.from_wire(zero),op.from_wire(one)
    with mp.context(precision=256):
        for k in range(3,-bits-1,-1):
            delta=g.param(2.**k)
            take=(one-((delta-rem)*g.param(2.**(53-k))).relu()).relu()
            rem=rem-take*delta
            u=mp.expm1(mp.mpfr(2)**(k+1))
            t=op.const(u/(2+u))
            change_n=gated(take,op.multiply(d,t))
            change_d=gated(take,op.multiply(n,t))
            n,d=op.add(n,change_n),op.add(d,change_d)
    # At each unit bit compare the remainder to the shifted denominator.
    # The comparison gain uses a fixed constant, never a data product.
    rem,quot=n,op.from_wire(zero)
    large=g.param(2.**500)
    digit_sum=zero
    seen=[zero]
    first=[zero]
    for k in range(1,(min(bits,82) if guard_round else bits)+1):
        increment=g.param(2.**(-k))
        divisor=scale(d,increment)
        trial=op.add(rem,minus(divisor))
        take=(one-(zero-high(trial)).relu()*large).relu()
        rem=op.add(rem,minus(gated(take,divisor)))
        if guard_round:
            # seen[k] records whether any of the first k binary digits was set.
            # Accumulate the 53 significant digits exactly. A set guard digit
            # adds one full unit of the final place. Nonzero tanh(binary64 x)
            # is transcendental, so the mathematical output cannot be a midpoint.
            previously=seen[max(0,k-53)]
            at_guard=first[max(0,k-53)]
            significant=(take-previously).relu()
            round_up=(take+at_guard-one).relu()
            digit_sum=(digit_sum+significant*increment)+round_up*(increment*2.)
            new_seen=one-(one-(seen[-1]+take)).relu()
            first.append(new_seen-seen[-1])
            seen.append(new_seen)
        else:
            quot=op.add(quot,op.from_wire(take*increment))
    out=digit_sum if guard_round else op.final(quot)
    # |x|-tanh(|x|) <= |x|^3/3. The resulting correctly rounded answer for
    # |x| <= 2^-27 is x, also for subnormal inputs.
    cutoff=g.param(2.**-27)
    gain=g.param(2.**79)
    pos=one-(one-((g.x-cutoff)*gain).relu()).relu()
    neg=one-(one-((zero-g.x-cutoff)*gain).relu()).relu()
    def keep(a,on):
        block=(one-on)*128.
        return (a-block).relu()-(zero-a-block).relu()
    keep(out,pos)-keep(out,neg)+keep(g.x,one-pos-neg)
    static_input_check(g)
    return g
