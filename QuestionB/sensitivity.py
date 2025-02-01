import sympy as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

# Define symbolic variables
n = sp.Symbol('n')  # Number of visitors
I_E = sp.Symbol('I_E')  # Infrastructure investment

# Define constants and assumptions
alpha_WD = 0.0065  # ton/person per year
alpha_WS = 0.02  # Ton/dollar Waste supply growth factor
alpha_HD = 100 * 4  # gallon/person/year Water demand growth rate
alpha_HS = 200  # gallon / dollar Water supply growth factor
alpha_FD = 25  # Traffic demand growth rate
alpha_FS = 500  # Traffic supply growth factor

# Initial capacities for infrastructure
initial_C_W = 350000  # Waste infrastructure capacity (e.g., tons)
initial_C_T = 120000000  # Traffic capacity (e.g., 7 million vehicles/year)
initial_C_H = 350_000_000 * 7  # Water infrastructure capacity

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
I = R - 30 * n  # Total investment amount
I_W = 0.2 * I + I_E  # Waste infrastructure investment (now depends on I_E)
I_H = 0.1 * I + I_E  # Water infrastructure investment (now depends on I_E)
I_T = 0.7 * I + I_E  # Traffic infrastructure investment (now depends on I_E)

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

# Resident satisfaction based on revenue, infrastructure, and environment
r_Pi = 1 / (1 + sp.exp(-0.0001 * ((Pi) / 30000 - 45000)))
r_E = 1 / (1 + sp.exp(20 * (ILI - 0.8)))

W_Pi = 0.4  # Weight of revenue in resident satisfaction
W_E = 0.4  # Weight of infrastructure in resident satisfaction
W_V = 0.2  # Weight of environment in resident satisfaction

Omega = W_Pi * r_Pi + W_E * r_E + W_V * r_E  # Note: Only revenue and infrastructure contribute to satisfaction

# Lambda function for Omega to evaluate it efficiently
Omega_func = sp.lambdify([n, I_E], Omega, 'numpy')

# Glacier Volume Model: V_g = V_initial - decay_factor * T
V_initial = 1000  # Example initial volume of glacier (in km^3)
decay_factor = 0.05  # Decay factor for glacier volume (km^3 per degree Celsius increase)

# Function to calculate Glacier Volume based on temperature
def glacier_volume(T_value):
    return V_initial - decay_factor * T_value

# Function to perform Sobol sensitivity analysis
def perform_sobol_analysis(problem, model_output):
    sobol_results = sobol.analyze(problem, np.array(model_output))
    return sobol_results

# Function to analyze and plot Sobol sensitivity for given parameters
def analyze_sensitivity(problem, model_output, title):
    # Perform Sobol sensitivity analysis
    sobol_results = perform_sobol_analysis(problem, model_output)

    # Print the Sobol indices
    print(f"First-order Sobol indices for {title}: {sobol_results['S1']}")
    print(f"Total Sobol indices for {title}: {sobol_results['ST']}")

    # Plot Sobol indices using Seaborn for a polished look
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(x=problem['names'], y=sobol_results['S1'], palette="Blues_d", ax=axes[0])
    axes[0].set_title(f'First-order Sobol Indices ({title})', fontsize=16)
    axes[0].set_ylabel('Sensitivity Index', fontsize=12)
    axes[0].set_xlabel('Input Parameters', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(x=problem['names'], y=sobol_results['ST'], palette="Oranges_d", ax=axes[1])
    axes[1].set_title(f'Total Sobol Indices ({title})', fontsize=16)
    axes[1].set_ylabel('Sensitivity Index', fontsize=12)
    axes[1].set_xlabel('Input Parameters', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# Function to analyze sensitivity of glacier volume to carbon emissions
def analyze_glacier_volume_sensitivity():
    problem = {
        'num_vars': 1,
        'names': ['carbon_emissions'],
        'bounds': [
            [0, 1000000]  # Range of carbon emissions
        ]
    }

    param_values = saltelli.sample(problem, 1000)
    model_output = []

    for params in param_values:
        emission_value = params[0]
        
        # Calculate Pi from emission (assuming Pi is related to emission via some formula)
        Pi_value = emission_value / 0.623
        
        # Calculate temperature change caused by Pi
        T_value = Pi_value / (epsilon_S - epsilon_D)
        
        # Calculate glacier volume (V_g) for the given temperature
        V_g_value = glacier_volume(T_value)  # Use the new glacier volume function
        model_output.append(V_g_value)

    analyze_sensitivity(problem, model_output, "Glacier Volume")

# Run the glacier volume sensitivity analysis
analyze_glacier_volume_sensitivity()

# Function to analyze sensitivity of government revenue to visitor numbers and tax rates
def analyze_revenue_sensitivity():
    problem = {
        'num_vars': 2,
        'names': ['visitor_numbers', 'tax_rate'],
        'bounds': [
            [1000000, 2000000],  # Visitor numbers range
            [0.1, 0.5]  # Tax rate range (10% to 50%)
        ]
    }

    param_values = saltelli.sample(problem, 1000)
    model_output = []

    for params in param_values:
        visitors_example = params[0]
        tau_value = params[1]
        
        # Calculate Pi (business revenue)
        Pi_value = visitors_example * P.subs(n, visitors_example)
        
        # Calculate government revenue
        R_value = tau_value * Pi_value
        model_output.append(R_value)

    analyze_sensitivity(problem, model_output, "Revenue")

# Run the revenue sensitivity analysis
analyze_revenue_sensitivity()
