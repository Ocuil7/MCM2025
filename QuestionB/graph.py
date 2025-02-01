import sympy as sp
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


# Function to apply moving average smoothing
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

visitors_example = 1_700_000  # Initial number of visitors (example value)
temperature_value = 1  # Example temperature value

# Define constants and assumptions
alpha_WD = 0.0065  # ton/person per year #
alpha_WS = 0.02  # Ton/dollar Waste supply growth factor (assumed) #
alpha_HD = 100 * 4  # gallon/person/year Water demand growth rate (gallons per tourist per year) #
alpha_HS = 200  # gallon / dollar Water supply growth factor (assumed) #
alpha_FD = 25  # Traffic demand growth rate
alpha_FS = 500  # Traffic supply growth factor

N = 30000  # Maximum number of residents in the region

# Traffic and Waste Infrastructure Constants (initial values)
initial_C_W = 300000  # Waste infrastructure capacity (e.g., tons)
initial_C_T = 100000000  # Traffic capacity (e.g., 7 million vehicles/year)
initial_C_H = 300_000_000 * 7  # 

# Define symbolic variables
n = sp.Symbol('n')  # Number of visitors
T = sp.Symbol('T')  # Temperature (°C)

# Price level equation parameters
epsilon_S = 0.5  # elasticity of supply
epsilon_D = -0.3  # elasticity of demand
c_0 = sp.ln(232/(1_700_000 ** (1 / (epsilon_S - epsilon_D))))  # Example constant value

# Updated Price equation
P = sp.exp(c_0) * (n ** (1 / (epsilon_S - epsilon_D)))

# Business revenue
Pi = n * P
tau = 0.5
R = tau * Pi

# Assume government spending on waste, water, and traffic
I = R - 30*n
I_W = 0.2*I
I_H = 0.1*I
I_T = 0.7*I

# Waste demand
W_D = alpha_WD * n 
W_S = alpha_WS * I_W + initial_C_W

# Water demand
H_D = alpha_HD * n
H_S = initial_C_H + alpha_HS * I_H 

# Traffic demand
F_D = alpha_FD * n
F_S = alpha_FS * I_T + initial_C_T

# Infrastructure metric (Infrastructure Load Index)
ILI = ((H_D / H_S) ** 2 + (W_D / W_S) ** 2 + (F_D / F_S) ** 2) ** (1 / 2)

# Glacier volume as a function of temperature (simplified model)
Tm = 0  # Reference temperature
v = 1050  # Glacier volume scaling factor (example)
t = 1  # Temperature in the model (example)
alpha = v / sp.ln(-((t - Tm) - 20))  # Simplified formula for glacier volume as a function of temperature
V_g = alpha * sp.ln(-((T - Tm) - 20))

# Resident satisfaction based on revenue, infrastructure, and environment
r_Pi = 1 / (1 + sp.exp(-0.0001 * ((Pi) / N - 4)))
r_E = 1 / (1 + sp.exp(20 * (ILI - 0.8)))
r_V = 1 / (1 + sp.exp(-0.01 * (V_g - 900)))

W_Pi = 0.4  # Weight of revenue in resident satisfaction
W_E = 0.4  # Weight of infrastructure in resident satisfaction
W_V = 0.2  # Weight of environment in resident satisfaction

Omega = W_Pi * r_Pi + W_E * r_E + W_V * r_V

# Set temperature value (e.g., T = -6°C)
temperature_value = 1  # Example temperature value


# Lambda function for Omega to evaluate it efficiently
Omega_func = sp.lambdify([n, T], Omega, 'numpy')  # Use numpy for lambdify
# Lists to store values for plotting
visitors_list = []
revenue_list = []
ILI_list = []
glacier_volume_list = []
price_list = []
H_ratio_list = []
W_ratio_list = []
T_ratio_list = []
temp_list = []
W_D_list = []
W_S_list = []
H_D_list = []
H_S_list = []
F_D_list = []
F_S_list = []
omega_list = []  # Small omega
Omega_list = []  # Big Omega

for _ in range(200):  # Max iterations to avoid infinite loops
    # Calculate Omega based on the current number of visitors
    Omega_val = Omega_func(visitors_example, temperature_value)

    # Calculate visitor satisfaction omega
    W_P_omega = 0.1
    W_E_omega = 0.2
    W_V_omega = 0.7
    omega = W_P_omega * (1 / (1 + sp.exp(0.03*(P - 250)))) + W_E_omega * r_E + W_V_omega * r_V

    # Store omega and Omega values
    omega_list.append(omega.subs(n, visitors_example).subs(T, temperature_value).evalf())
    Omega_list.append(Omega_val)

    # Update the number of visitors (visitors model)
    new_visitors = n * sp.exp((omega - 0.83)*0.1)

    # Update temperature based on business revenue
    G = Pi * 0.623

    C_temp =1 - (0.8 / sp.ln(2) * sp.ln(394400000 * 0.623))

    try:
        temperature_value = C_temp + 0.8 / sp.ln(2) * sp.ln(G) + random.normalvariate(0, 0.0001)
    except ValueError:
        print("Error in calculating temperature. Check value of G.")
        break
    temperature_value = temperature_value.subs({n: visitors_example}).evalf()

    # Ensure temperature is real before proceeding
    if temperature_value.is_real:
        temperature_value = float(temperature_value)
    else:
        print(f"Warning: Non-real temperature value encountered. Skipping.")
        continue

    # Evaluate new_visitors expression numerically
    new_visitors_num = new_visitors.subs({n: visitors_example, T: temperature_value}).evalf() + random.normalvariate(0, 1000)
    print(f"New Visitors: {new_visitors_num}")

    # Store values for plotting
    visitors_list.append(visitors_example)
    revenue_list.append(Pi.subs(n, visitors_example).subs(T, temperature_value).evalf())
    ILI_list.append(ILI.subs(n, visitors_example).subs(T, temperature_value).evalf())
    glacier_volume_list.append(V_g.subs(T, temperature_value).evalf())
    temp_list.append(temperature_value)

    # Track the price (P) over time
    price_list.append(P.subs(n, visitors_example).evalf())

    # Store demands and supplies for water, housing, and traffic
    W_D_list.append(W_D.subs(n, visitors_example).evalf())
    W_S_list.append(W_S.subs(n, visitors_example).evalf())
    H_D_list.append(H_D.subs(n, visitors_example).evalf())
    H_S_list.append(H_S.subs(n, visitors_example).evalf())
    F_D_list.append(F_D.subs(n, visitors_example).evalf())
    F_S_list.append(F_S.subs(n, visitors_example).evalf())

    # Ensure real-valued ratios
    if H_S.subs(n, visitors_example).evalf() != 0:
        H_ratio_list.append(H_D.subs(n, visitors_example).evalf() / H_S.subs(n, visitors_example).evalf())
    else:
        H_ratio_list.append(np.nan)  # Append NaN if division by zero

    if W_S.subs(n, visitors_example).evalf() != 0:
        W_ratio_list.append(W_D.subs(n, visitors_example).evalf() / W_S.subs(n, visitors_example).evalf())
    else:
        W_ratio_list.append(np.nan)  # Append NaN if division by zero

    if F_S.subs(n, visitors_example).evalf() != 0:
        T_ratio_list.append(F_D.subs(n, visitors_example).evalf() / F_S.subs(n, visitors_example).evalf())
    else:
        T_ratio_list.append(np.nan)  # Append NaN if division by zero

    # Check for convergence
    if abs(new_visitors_num - visitors_example) < 0.0001:
        print(f"Converged to stable number of visitors: {new_visitors_num}")
        break

    # Update number of visitors for next iteration
    visitors_example = int(new_visitors_num)

# Smooth the visitors list (optional)
smoothed_visitors_list = moving_average(visitors_list, window_size=10)

print(revenue_list)
print(price_list)
print(visitors_list)

# Plot all variables individually
plt.figure(figsize=(16, 20))

# Plot Number of Visitors (with smoothing)
plt.subplot(3, 2, 1)
plt.plot(smoothed_visitors_list, label='Visitors', color='tab:blue', linewidth=2, marker='o', markersize=2, markerfacecolor='darkblue', markeredgewidth=2)
plt.title('Number of Visitors', fontsize=18, fontweight='bold', color='darkblue')
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Visitors', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

# Plot Glacier Volume
plt.subplot(3, 2, 2)
plt.plot(glacier_volume_list, label='Glacier Volume (km^3)', color='tab:red', linewidth=2, marker='o', markersize=2, markerfacecolor='darkred', markeredgewidth=2)
plt.title('Glacier Volume (km^3)', fontsize=18, fontweight='bold', color='darkred')
plt.xlabel('year', fontsize=14, fontweight='bold')
plt.ylabel('Volume', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
