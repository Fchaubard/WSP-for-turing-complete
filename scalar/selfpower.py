"""Density-adapted 1D ReLU splines for the positive real self-power function."""
from scipy.optimize import brentq
from benchmark import SEEDS, WIDTHS, data, hinge, initial_hinge, metrics, np, pack, write_records

def grid(h, mode):
    if mode == "uniform":
        return np.linspace(0,10,h+2)
    # x = t^2 regularizes sqrt(1/x) at the origin. The limiting transformed
    # density equals two for either local absolute or relative error.
    t = np.linspace(0,np.sqrt(10),65537)
    x = t*t
    jac_density = np.full_like(t,2)
    jac_density[1:] = 2*np.sqrt(1+x[1:]*(1+np.log(x[1:]))**2)
    if mode == "absolute":
        jac_density *= np.sqrt(np.power(x,x))
    cumulative = np.r_[0,np.cumsum((jac_density[:-1]+jac_density[1:])/2*np.diff(t))]
    knots = np.interp(np.linspace(0,cumulative[-1],h+2),cumulative,t)**2
    knots[[0,-1]] = [0,10]
    return knots

def center_relative(knots, values, slopes):
    # A positive convex function lies below its secants. Every secant-to-target
    # ratio has a unique interior maximum: the derivative numerator of l/f,
    # l' f-l f', has derivative -l f'' < 0. Balance this maximum with the
    # minimum ratio 1 at the knots by scaling the entire readout.
    maximum = 1.
    for left,right,y,slope in zip(knots[:-1],knots[1:],values[:-1],slopes):
        def derivative_sign(x):
            if x == 0:
                return np.inf
            return slope-(y+slope*(x-left))*(1+np.log(x))
        x = brentq(derivative_sign,left,right)
        ratio = (y+slope*(x-left))/x**x
        maximum = max(maximum,ratio)
    return 2/(1+maximum)

def run():
    rows = []
    sets = data("selfpower")
    for h in WIDTHS:
        for mode in ("uniform","relative","absolute","uniform_centered","relative_centered"):
            k = grid(h,mode.removesuffix("_centered"))
            y = np.power(k,k)
            slope = np.diff(y)/np.diff(k)
            p = initial_hinge(h,SEEDS[0])
            p[:2] = [slope[0],y[0]]
            if h:
                p[2:] = pack([np.ones(h),-k[1:-1],np.diff(slope)])
            if mode.endswith("_centered"):
                scale = center_relative(k,y,slope)
                p[:2] *= scale
                if h:
                    p[-h:] *= scale
            for split,x in sets.items():
                target = np.power(x,x)
                pred = hinge(x,p)
                rel = np.abs(pred-target)/target
                rows.append(dict(task="selfpower",architecture="hinge",size=h,params=p.size,
                                 nonzero=np.count_nonzero(p),method=mode,seed=SEEDS[0],split=split,
                                 **metrics(pred,target), mean_relative=rel.mean(), max_relative=rel.max()))
    write_records(rows,"selfpower.csv")

if __name__ == "__main__":
    run()
