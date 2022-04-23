from copy import copy
from sympy import * 
import math
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from sympy.stats.crv_types import rv, SingleContinuousDistribution, _value_check
from sympy import stats

'''
ASSUMPTIONS:
1. All invariance functions are monotonic 
'''

def opt_swap_2(delta_f, U_f, omega, vars, relational=False):
    ''' Optimal routing between two tokens
    delta_f: relative invariant function. sympy expression f(a)
    U_f: sympy expression g(b,c)
    vars: [a,b,c]
    relational: whether to ignore bounds and process optimization symbolically
    '''

    a, b, c = vars 
    x = symbols('x_')
    b_prime = x + omega[0]
    c_prime = -delta_f.subs(a, -x) + omega[1]
    U_f2 = U_f.subs([(b, b_prime), (c, c_prime)])
    upper_bound = solve(c_prime, x)
    if relational:
        x_max = solve(diff(U_f2, x), x)[0]
    else:
        x_max = np.clip([y.evalf() for y in solve(diff(U_f2, x), x)], sympify(-omega[0]), upper_bound)
        U_max = [U_f2.subs(x, x_prime).evalf() for x_prime in x_max]
        x_max = x_max[U_max.index(max(U_max))].evalf()
    sol_max = (b_prime.subs(x, x_max), c_prime.subs(x, x_max))
    return sol_max, U_f.subs([(b, sol_max[0]), (c, sol_max[1])])


#region Pool Operations
def invariant_to_delta(invariant, initial, x):
    '''Convert an invariance function into a delta function
    invariant: sympy expression \psi(x,y) 
    x: the symbol for the parameter of delta function 
    '''
    C = invariant.subs(initial)
    q = symbols('q_')
    return solve(invariant.subs([(initial[0][0], initial[0][1] + x), (initial[1][0], initial[1][1] + q)]) - C,q)[0]

def delta_to_invariant(delta, initial, x):
    '''Convert a delta function to an invariant function (offset by initial value)
    delta: sympy expression f(x)
    '''
    psi = delta.subs(x, initial[0][0] - initial[0][1]) - (initial[1][0] - initial[1][1])
    constants = [x for x in psi.expand().args if x.is_constant() is True]
    return psi - sum(constants)


def chain_pools(invariant1, invariant2, initial1, initial2):
    '''
    invariant1: f(x,y)
    invariant2: g(p,q)
    '''
    a = symbols('a_')

    chained_initial = [initial1[0], initial2[1]]

    return delta_to_invariant(chain_pools_delta(invariant1, invariant2, initial1, initial2, a), chained_initial, a), chained_initial 

def chain_pools_delta(invariant1, invariant2, initial1, initial2, a):
    '''
    invariant1: f(x,y)
    invariant2: g(p,q)
    '''
    C1 = invariant1.subs(initial1)
    C2 = invariant2.subs(initial2)
    k = symbols('k_')
    j = symbols('j_')
    k = solve(invariant1.subs([(initial1[0][0], initial1[0][1] + a), (initial1[1][0], initial1[1][1] + k)]) - C1, k)[0]
    j = solve(invariant2.subs([(initial2[0][0], initial2[0][1] - k), (initial2[1][0], initial2[1][1] + j)]) - C2, j)[0]

    return j


def stack_pools_parametric(invariant1, invariant2, initial1, initial2):
    '''
    For computational tractability, parametric equation is used for stacked liquidity 
    invariant1: f(x,y)
    invariant2: g(p,q)
    returns a parametric function x,y = q(t)
    '''
    
    spot_p = spot_price(invariant1, initial1) 

    # Must have the same spot price
    assert(abs(spot_p - \
        spot_price(invariant2, initial2)) < 1e-8)
    

    combined_initial = [initial1[0][1] + initial2[0][1], initial1[1][1] + initial2[1][1]]

    t = symbols('t') # parameter
    deltas = [0,0]

    for curve in [(invariant1, initial1), (invariant2, initial2)]:
        invariant, init = curve 
        x, y = init[0][0], init[1][0]
        dp_dx, dp_dy = dp_dxdy(invariant, [x,y])
        C = invariant.subs(init)
        y_x = solve(invariant - C, y, manual=True)[0]
        p_expr = y_x/x 
        p = symbols('p')
        x_p = solve(p_expr - p, x, manual=True)[0]
        dp_dx = dp_dx.subs(y, y_x).subs(x, x_p)
        dp_dy = dp_dy.subs(y, y_x).subs(x, x_p)
        delta_x = integrate(1/dp_dx, (p, spot_p, t))
        delta_y = integrate(1/dp_dy, (p, spot_p, t))

        deltas[0] += delta_x 
        deltas[1] += delta_y

    print(delta_x, delta_y)
    return lambda t_v: (combined_initial[0] + deltas[0].subs(t, t_v).evalf(), combined_initial[1] + deltas[1].subs(t, t_v).evalf())

def spot_price(invariant, liq):
    x, y = liq[0][0], liq[1][0]
    return (diff(invariant, x) / diff(invariant, y)).subs(liq).evalf()

def dp_dxdy(invariant, vars):
    '''Get dp/dx and dp/dy, where p = y/x 
    invariant: sympy expression f(x,y)
    vars = [x,y]
    '''
    x, y = vars 
    dp_dx = -y * x ** -2 - x ** -1 * diff(invariant, x) / diff(invariant, y)
    dp_dy = x ** -1 + y * x ** -2 * diff(invariant, y) / diff(invariant, x)

    return (dp_dx, dp_dy)

#endregion 

#region Curve Generators

def budget_curve(delta_f, omega, a, resolution=100):
    '''Get the budget curve of a trader given the liquidity curve that he can trade with 
    delta_f: relative invariant function. sympy expression f(a)
    omega: initial endowment
    '''
    x = symbols('x_')
    c_prime = -delta_f.subs(a, -x) + omega[1]
    upper_bound = solve(c_prime, x)[0].evalf()
    delta_range = np.linspace(-omega[0], float(upper_bound), num=resolution)
    x_range = delta_range + omega[0]
    y_range = np.array([-delta_f.subs(a, -x).evalf() + omega[1] for x in delta_range])
    return (x_range, y_range)

def indifference_curve(U_f, U, a_interval, vars, resolution=100):
    '''Get indifference curve from utility function 
    U_f: Utility/consumption function. sympy expression f(a,b)
    Us: Contour value. float
    '''
    a, b = vars 
    a_range = np.linspace(a_interval[0], a_interval[1], resolution)
    y_range = np.array([solve(U_f.subs(a, x) - U, b) for x in a_range])
    return (a_range, y_range)

def invariant_curve(invariant, a_interval, initial, vars, resolution=100):
    '''Draw invariant curve 
    invariant: Invariance function. sympy expression f(a,b)
    '''
    a,b = vars 
    a_range = np.linspace(a_interval[0], a_interval[1], resolution)
    C = invariant.subs(initial).evalf() 
    b_range = np.array([solve(invariant.subs(a, x) - C, b)[0] for x in a_range])
    valid_indices = [i for i in range(resolution) if not b_range[i].has(oo, -oo, zoo, nan)]

    return (a_range[valid_indices], b_range[valid_indices])

def invariant_curve_parametric(invariant, t_interval, resolution=100):
    '''Draw invariant curve 
    invariant: Invariance function. sympy expression f(a,b)
    '''
    t_range = np.linspace(t_interval[0], t_interval[1], resolution)
    a_range = []
    b_range = []
    for t in t_range:
        a, b = invariant(t) 
        a_range += [a] 
        b_range += [b] 
    a_range = np.array(a_range) 
    b_range = np.array(b_range)
    valid_indices = [i for i in range(resolution) if not b_range[i].has(oo, -oo, zoo, nan)]

    return (a_range[valid_indices], b_range[valid_indices])


def expected_consumption_2T1S(con_f, portfolio, invariant1, invariant2, initial1, initial2, cov, vars, mc_samples=10):
    '''Construct expected utility function over effective market liquidity (2 items)
    con_f: consumption function defined over portfolio. sympy expression: f(a,b)
    cov: 2x2 matrix
    vars: [a,b]
    '''
    
    x, y, p, q = initial1[0][0], initial1[1][0], initial2[0][0], initial2[1][0]
    a, b = vars 
    C1, C2 = invariant1.subs(initial1), invariant2.subs(initial2)
    d, e, f = symbols('d e f')
    

    final_r23 = initial2[0][1] + portfolio[1]
    d = solve(invariant2.subs([(p,final_r23), (q, initial2[1][1] - d)]) - C2, d)[0]
    max_t3 = (portfolio[2] + d) 
    final_r32 = initial2[1][1] - d 
    final_r31 = initial1[1][1] + max_t3
    e = solve(invariant1.subs([(x, initial1[0][1] - e), \
            (y, final_r31 )]) - C1, e)[0]
    final_r13 = initial1[0][1] - e
    max_t1 = portfolio[0] + e 
    final_deltac = chain_pools_delta(invariant1, invariant2, [(x, final_r13), (y, final_r31)], [(q, final_r32), (p, final_r23)], f)
    # Run monte carlo sample over perturbations
    perturbations = np.random.multivariate_normal([0,0], cov, mc_samples)
    max_consump = 0
    for i in range(mc_samples):
        _, max_consump_ = opt_swap_2(final_deltac, con_f.subs([(a, a*math.e**perturbations[i][0]), (b, b*math.e**perturbations[i][1])]), [max_t1, 0], [f, a, b], relational = True)
        max_consump += max_consump_
    max_consump = max_consump / mc_samples
    # _, max_consump = opt_swap_2(final_deltac, con_f, [max_t1, 0], [f, a, b], relational = True)

    return max_consump


def infinite_liq_expected_payoff_analytic():
    t1, t2, x, y, w1, w2 = symbols('t1 t2 x y w1 w2')
    covariance, sig1, sig2 = symbols('cov sig1 sig2')
    corr = covariance / (sig1 * sig2)
    xy = Matrix([x, y])
    initials = [(t1, 10), (t2, 10), (w1, 3), (w2, 5)]
    expected_util = []
    x_range = np.linspace(-0.2, 0.2, 10)
    for covariance in x_range:
        random_xy = stats.Normal('test', Matrix([0,0]), Matrix([[0.3,covariance], [covariance,0.3]]))
        f_xy = stats.density(random_xy)(xy)
        max_t1 = (w1 + x) * (t1 + t2 * w2 / w1)
        max_t2 = (w2 + y) * (t2 + t1 * w1 / w2)
        t2_lim = w2 / w1 * x
        max_t1, max_t2, t2_lim = max_t1.subs(initials), max_t2.subs(initials), t2_lim.subs(initials)
        print(max_t1, max_t2, t2_lim)
        inner_int = Integral(f_xy * max_t2, (y, -oo, t2_lim)).doit() + Integral(f_xy * max_t1, (y, t2_lim, oo)).doit()
        complete_int = Integral(inner_int, (x, -oo, oo)).doit()
        expected_util += [complete_int.evalf()]
        print(expected_util[-1])
    plt.plot(x_range, expected_util)

def liq_expected_payoff_mc(cov, f, w, samples=1000): 

        w1, w2 = w
        perturbation = np.random.multivariate_normal([-cov[0][0]/2,-cov[1][1]/2], cov, samples)
        w1_prime = w1 * np.exp(perturbation[:, 0])
        w2_prime = w2 * np.exp(perturbation[:, 1])

        sample_mean = 0 
        sample_variance = 0 

        for i in range(samples):
                sample_mean += f(w1_prime[i], w2_prime[i])

        sample_mean /= samples 
        for i in range(samples):
                sample_variance += (f(w1_prime[i], w2_prime[i]) - sample_mean) ** 2 
        sample_variance /= samples 
 
        
        return (sample_mean, math.sqrt(sample_variance))


#endregion 

#region Helper
def get_cartesian_from_barycentric(b, t):
    return t.dot(b)

class MultivariateNormalDistribution(SingleContinuousDistribution):
        _argnames = ('mean', 'std')
        @staticmethod
        def check(mean, std):
                pass
                # _value_check(std > 0, "Standard deviation must be positive")
        def pdf(self, x):
                return exp(-S.Half * (x - self.mean).T * (self.std.inv()) * (x - self.mean)) / (sqrt(2*pi)**(self.std.shape[0])*self.std.det())
        def sample(self):
                pass
                # define sampling function here

def MultivariateNormal(name, mean, std):
        return rv(name, MultivariateNormalDistribution, (mean, std))

#endregion

