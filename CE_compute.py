# Competitive Equilibrium
from scipy.optimize import minimize, Bounds
from sympy import * 
import numpy as np 
import math 

m = 2
n = 10 
F = lambda : np.exp(np.random.normal([0,0], [0.1,0.1]))
θ_1, θ_2 = symbols('θ_1 θ_2')
γ_1, γ_2 = symbols('γ_1 γ_2')
Ω = np.random.random((n,m))
Ω /= np.sum(Ω)
b = np.ones(m)
U = [ Matrix([θ_1 * γ_1, θ_2 * γ_2]).dot(F()) for k in range(n)]

# Marshallian demand
ξ = lambda U_θ, p, θ: (minimize(lambda X: -U_θ.subs({
    θ_1: X[0], θ_2: X[1]
}).evalf() , [0,0], constraints=[
    {"type":"ineq", "fun": lambda X: sum(p * θ - p * X)}
], bounds=Bounds([0,0], [np.inf, np.inf]))).x 

# Price movements 
γ_t = np.array([1.0, 1.0])

# Define sufficient KKT conditions 
parse_args = lambda x: (x[:m*n].reshape(n,m), \
    x[m*n:m*n+n].reshape(n),\
        x[m*n+n:m*n+n+m].reshape(m), \
            x[m*n+n+m:m*n+n+m+n*m].reshape(n,m))
lagrangian = lambda y, α, p, μ: sum([α[k] * U[k].subs({
    θ_1: y[k][0], θ_2: y[k][1],
    γ_1: γ_t[0], γ_2: γ_t[1],
}) for k in range(n)]) \
    - sum(p * (np.sum(y) - b)) \
        - np.sum(μ * y)

optimum_vector_constraints = [{"type":"ineq", "fun": lambda X: (lambda y, α, p, μ: α[k])(*parse_args(X)) } for k in range(n) ] # element-wise coefficients must be >= 0
# complementary slack conditions
slack_conditions1 = [{"type":"ineq", "fun": lambda X: (lambda y, α, p, μ: p[i])(*parse_args(X)) } for i in range(2) ] 
slack_conditions2 = [{"type":"eq", "fun": lambda X: (lambda y, α, p, μ: p[i] * (np.sum(y[:,i] - b[i])))(*parse_args(X)) } for i in range(2) ] 

competitive_equilibrium = {"type":"eq", "fun": lambda X: (lambda y, α, p, μ: sum([ξ(U[k].subs({
    γ_1: γ_t[0], γ_2: γ_t[1],
}), p, y[k]) for k in range(n)]) - b)(*parse_args(X))} 

init = np.random.uniform(size=(m*n+n+m+n*m)) 
solution = minimize(lambda X: -lagrangian(*parse_args(X)), init, method='SLSQP', constraints = optimum_vector_constraints + slack_conditions1 + slack_conditions2 + [competitive_equilibrium]) 
print(solution)
