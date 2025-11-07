import sympy as sp
phi = (1 + sp.sqrt(5))/2
eta = 1 - 1/phi  # ≈0.381966
psi = eta * sp.ln(phi)  # ≈0.183807
print(f"η = {eta.evalf(6)}, ψ = {psi.evalf(6)} ψ-bits/mode")