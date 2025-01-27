import numpy as np
import sympy as sp
from scipy.optimize import minimize
from scipy.integrate import quad


def sustainable_tourism_model():
    # Define constants and assumptions
    alpha_WD = 0.001  # ton/person Waste growth rate
    alpha_WS = 1 # TODO
    alpha_HD = 100 * 7.2 # gallon Water demand growth rate
    alpha_HS = 1 # TODO
    alpha_FD = 1  # TODO
    alpha_FS = 1 # TODO
    alpha_n = 1  # Visitor satisfaction coefficient
    
    C_W = 1000  # TODO
    C_T = 70000   # Traffic capacity (assumed constant)
    initial_water_supply = 1000000 * 100  # 1 billion gallons/year

    B_r = 0.4   # Weight of infrastructure in resident satisfaction
    B_E = 0.3   # Weight of environment in resident satisfaction
    B_Pi = 0.3  # Weight of economy in resident satisfaction

    V_0 = 100  # Reference glacier volume (arbitrary scale, normalized)
    Pi_0 = 1_000_000  # Reference business revenue (normalized)

    k = 0.02  # Glacier melting rate coefficient
    T_m = 10  # Mean temperature for glacier equation (°C)
    C_0 = 1  # Constant in temperature equation

    # Visitor satisfaction weight placeholder
    visitor_satisfaction_weight = 0.7

    # Assume government spending on waste, water, and traffic
    I_W = 0
    I_H = 0
    I_T = 0

    # Define variables
    n = sp.Symbol('n')  # Number of visitors
    T = sp.Symbol('T')  # Temperature (°C)

    e_S = 0.5
    e_D = -0.5

    # Price level
    P = sp.exp(C_0 * n**(1 / (e_S - e_D)))

    # Business revenue
    Pi = n*P

    # Government revenue
    R = 0.2 * Pi

    
    # Waste demand
    W_D = alpha_WD * n 

    # Waste supply
    W_S = alpha_WS * sp.ln(I_W) + C_W

    # Water demand
    H_D = alpha_HD * n

    # Water supply
    H_S = initial_water_supply + alpha_HS * sp.ln(I_H)

    # Traffic demand
    F_D = alpha_FD * n
    # Traffic supply
    F_S = alpha_FS * sp.ln(I_T) + C_T

    # Infrastructure metric
    r = ((H_D / H_S)**100 + (W_D / W_S)**100 + (F_D / F_S)**100)**(1 / 100)

    # Glacier volume as a function of temperature

    Tm = 0 
    v = 1000
    t = -6
    alpha = v / sp.log(-((t-Tm) - 31))
    #V_g = sp.sqrt(1 / beta * (C* sp.exp(-2 * beta * (0.5 * T**2 - Tm * T)) - alpha))
    V_g = alpha * sp.log(-((T-Tm)-31))

    # Resident satisfaction
    Omega = B_r * r + B_E * V_g / V_0 + B_Pi * Pi / Pi_0

    # Visitor satisfaction
    omega = sp.sqrt(r * V_g) / P

    # Visitor satisfaction influencing number of visitors
    n_omega = alpha_n * omega

    # Substitute constants for demonstration
    temp_example = 15  # Example temperature in °C
    visitors_example = 16000

    # Numerical evaluations
    Omega_func = sp.lambdify([n, T], Omega, 'numpy')
    Omega_val = Omega_func(visitors_example, temp_example)

    return {
        "Resident Satisfaction (Omega)": Omega_val,
        "Business Revenue (Pi)": float(Pi.subs(n, visitors_example)),
        "Infrastructure Metric (r)": float(r.subs(n, visitors_example)),
    }

#results = sustainable_tourism_model()
#for key, value in results.items():
#    print(f"{key}: {value}")

from sympy import symbols, Function, Eq, dsolve, exp
import matplotlib.pyplot as plt
# Glacier volume as a function of temperature
# beta = 0.2 # TODO
# Tm = 0 + 200
# v = 1000
# t = -6 + 200
# alpha = 0.2
# #C = (beta*(v**2) + alpha)/ (np.exp(beta*(Tm)**2 - beta*((t - Tm))))
# C = v**2 + alpha*(t**2 - 2*Tm*t)
# alpha = 0.2
# beta = 0.2
# T = -5 + 200
# #V_g = np.sqrt((1 / beta) * (C* np.exp(beta * ((Tm **2) - (T - Tm)**2)) - alpha))
# # Define the range of temperatures (in Celsius)
# T_range = np.linspace(-10, 10, 100)  # Temperature range from -10°C to 10°C
# print(C)
# # Calculate glacier volume (V_g) for each temperature
# #V_g_values = np.sqrt((1 / beta) * (C * np.exp(beta * (Tm**2 - (T_range - Tm)**2)) - alpha))
# V = np.sqrt(C - alpha*(T_range**2 - 2*Tm*T_range))

# # Plot the results
# plt.plot(T_range, V, label='Glacier Volume')
# plt.xlabel("Temperature (°C)")
# plt.ylabel("Glacier Volume (V)")
# plt.title("Glacier Volume as a Function of Temperature")
# plt.grid(True)
# plt.legend()
# plt.show()

beta = 0.1 # TODO
Tm = 0 
v = 1000
t = -6
alpha = v / sp.log(-(t - 30))
print(alpha)
#V_g = sp.sqrt(1 / beta * (C* sp.exp(-2 * beta * (0.5 * T**2 - Tm * T)) - alpha))
T = 0
V_g = alpha * sp.log(-(T-31))
print(V_g)