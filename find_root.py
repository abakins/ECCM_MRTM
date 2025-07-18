
import numpy as np 
from numba import njit 

# Taken directly from scipy, and modified for compilation w/ numba 

@njit
def bracket(func, xa=0.0, xb=1.0, args=(), grow_limit=110.0, maxiter=1000):
    """
    Bracket the minimum of a function.

    Given a function and distinct initial points, search in the
    downhill direction (as defined by the initial points) and return
    three points that bracket the minimum of the function.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to minimize.
    xa, xb : float, optional
        Initial points. Defaults `xa` to 0.0, and `xb` to 1.0.
        A local minimum need not be contained within this interval.
    args : tuple, optional
        Additional arguments (if present), passed to `func`.
    grow_limit : float, optional
        Maximum grow limit.  Defaults to 110.0
    maxiter : int, optional
        Maximum number of iterations to perform. Defaults to 1000.

    Returns
    -------
    xa, xb, xc : float
        Final points of the bracket.
    fa, fb, fc : float
        Objective function values at the bracket points.
    funcalls : int
        Number of function evaluations made.

    Raises
    ------
    BracketError
        If no valid bracket is found before the algorithm terminates.
        See notes for conditions of a valid bracket.

    Notes
    -----
    The algorithm attempts to find three strictly ordered points (i.e.
    :math:`x_a < x_b < x_c` or :math:`x_c < x_b < x_a`) satisfying
    :math:`f(x_b) ≤ f(x_a)` and :math:`f(x_b) ≤ f(x_c)`, where one of the
    inequalities must be satisfied strictly and all :math:`x_i` must be
    finite.

    """
    _gold = 1.618034  # golden ratio: (1.0+sqrt(5.0))/2.0
    _verysmall_num = 1e-21
    # convert to numpy floats if not already
    xa, xb = np.asarray([xa, xb])
    fa = func(*(xa,) + args)
    fb = func(*(xb,) + args)
    if (fa < fb):                      # Switch so fa > fb
        xa, xb = xb, xa
        fa, fb = fb, fa
    xc = xb + _gold * (xb - xa)
    fc = func(*((xc,) + args))
    funcalls = 3
    iter = 0
    while (fc < fb):
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        if np.abs(val) < _verysmall_num:
            denom = 2.0 * _verysmall_num
        else:
            denom = 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        msg = ("No valid bracket was found before the iteration limit was "
               "reached. Consider trying different initial points or "
               "increasing `maxiter`.")
        if iter > maxiter:
            raise RuntimeError(msg)
        iter += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = func(*((w,) + args))
            funcalls += 1
            if (fw < fc):
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            elif (fw > fb):
                xc = w
                fc = fw
                break
            w = xc + _gold * (xc - xb)
            fw = func(*((w,) + args))
            funcalls += 1
        elif (w - wlim)*(wlim - xc) >= 0.0:
            w = wlim
            fw = func(*((w,) + args))
            funcalls += 1
        elif (w - wlim)*(xc - w) > 0.0:
            fw = func(*((w,) + args))
            funcalls += 1
            if (fw < fc):
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw = func(*((w,) + args))
                funcalls += 1
        else:
            w = xc + _gold * (xc - xb)
            fw = func(*((w,) + args))
            funcalls += 1
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw

    # three conditions for a valid bracket
    cond1 = (fb < fc and fb <= fa) or (fb < fa and fb <= fc)
    cond2 = (xa < xb < xc or xc < xb < xa)
    cond3 = np.isfinite(xa) and np.isfinite(xb) and np.isfinite(xc)
    
    if not (cond1 and cond2 and cond3):
        msg = ("The algorithm terminated without finding a valid bracket. "
           "Consider trying different initial points.")
        raise RuntimeError(msg)
        # e = BracketError(msg)
        # e.data = (xa, xb, xc, fa, fb, fc, funcalls)
        # raise e

    return xa, xb, xc, fa, fb, fc, funcalls

@njit
def brent(func, ixa=0.0, ixb=1.0, args=()):
     
    # set up for optimization
    xa, xb, xc, fa, fb, fc, funcalls = bracket(func, xa=ixa, xb=ixb, args=args)
    _mintol = 1.0e-11
    _cg = 0.3819660
    maxiter = 1000 
    tol = 1e-6
    #################################
    #BEGIN CORE ALGORITHM
    #################################
    x = w = v = xb
    fw = fv = fx = fb
    if (xa < xc):
        a = xa
        b = xc
    else:
        a = xc
        b = xa
    deltax = 0.0
    iter = 0

    while (iter < maxiter):
        tol1 = tol * np.abs(x) + _mintol
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        # check for convergence
        if np.abs(x - xmid) < (tol2 - 0.5 * (b - a)):
            break

        if (np.abs(deltax) <= tol1):
            if (x >= xmid):
                deltax = a - x       # do a golden section step
            else:
                deltax = b - x
            rat = _cg * deltax
        else:                              # do a parabolic step
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if (tmp2 > 0.0):
                p = -p
            tmp2 = np.abs(tmp2)
            dx_temp = deltax
            deltax = rat
            # check parabolic fit
            if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and
                    (np.abs(p) < np.abs(0.5 * tmp2 * dx_temp))):
                rat = p * 1.0 / tmp2        # if parabolic step is useful.
                u = x + rat
                if ((u - a) < tol2 or (b - u) < tol2):
                    if xmid - x >= 0:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                if (x >= xmid):
                    deltax = a - x  # if it's not do a golden section step
                else:
                    deltax = b - x
                rat = _cg * deltax

        if (np.abs(rat) < tol1):            # update by at least tol1
            if rat >= 0:
                u = x + tol1
            else:
                u = x - tol1
        else:
            u = x + rat
        fu = func(*((u,) + args))      # calculate new output value
        funcalls += 1

        if (fu > fx):                 # if it's bigger than current
            if (u < x):
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v = w
                w = u
                fv = fw
                fw = fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
        else:
            if (u >= x):
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        iter += 1
    #################################
    #END CORE ALGORITHM
    #################################

    xmin = x
    fval = fx
    iter = iter
    funcalls = funcalls

    # ENFORCE BOUNDS! 
    if xmin < ixa: 
        xmin = ixa
    elif xmin > ixb: 
        xmin = ixb

    return xmin, fval, iter, funcalls

@njit 
def solve_quadratic(a, b, c): 
    s1 = (-b + np.sqrt(b**2 - 4 * a * c)) / 2 / a
    s2 = (-b - np.sqrt(b**2 - 4 * a * c)) / 2 / a
    return s1, s2