from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple
import random
import math

@dataclass
class MHResult:
    best_x: float
    best_y: float
    best_value: float
    steps: int

def hill_climb(f: Callable[[float,float], float], x0: float, y0: float, step: float = 0.5, iters: int = 500) -> MHResult:
    x,y = x0,y0
    best = f(x,y)
    for i in range(iters):
        candidates = [(x+step,y),(x-step,y),(x,y+step),(x,y-step)]
        improved = False
        for cx,cy in candidates:
            v = f(cx,cy)
            if v < best:
                x,y,best = cx,cy,v
                improved = True
                break
        if not improved:
            step *= 0.9
            if step < 1e-3:
                return MHResult(x,y,best,i+1)
    return MHResult(x,y,best,iters)

def simulated_annealing(f: Callable[[float,float], float], x0: float, y0: float, t0: float = 5.0, alpha: float = 0.99, iters: int = 1000) -> MHResult:
    x,y = x0,y0
    bestx,besty = x,y
    best = f(x,y)
    t = t0
    for i in range(iters):
        nx = x + random.uniform(-1,1)
        ny = y + random.uniform(-1,1)
        cur = f(x,y)
        nv = f(nx,ny)
        if nv < cur or random.random() < math.exp(-(nv-cur)/max(t,1e-9)):
            x,y = nx,ny
        v = f(x,y)
        if v < best:
            best = v
            bestx,besty = x,y
        t *= alpha
    return MHResult(bestx,besty,best,iters)
