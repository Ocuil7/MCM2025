import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def sustainable_tourism_model():
    # Define constants and assumptions
    alpha_WD = 5  # ton/person Waste growth rate
    alpha_WS = 1  # TODO
    alpha_HD = 100 * 7.2  # gallon Water demand growth rate
    alpha_HS = 0.000000001  # TODO
    alpha_FD = 1  # TODO
    alpha_FS = 0.000000001  # TODO
    
    C_W = 1000  # TODO
    C_T = 70000  # Traffic capacity (assumed constant)
    initial_water_supply = 1000000  # 1 billion gallons/year

    W_Pi = 0.4  # Weight of revenue in resident satisfaction
    W_E = 0.4  # Weight of infrastructure in resident satisfaction
    W_V = 0.2  # Weight of environment in resident satisfaction

    # Define symbolic variables
    n = sp.Symbol('n')  # Number of visitors
    T = sp.Symbol('T')  # Temperature (°C)

    # Price level equation parameters
    epsilon_S = 1.5  # elasticity of supply
    epsilon_D = 1.2  # elasticity of demand
    c_0 = sp.ln(232/(16000 ** (1 / (epsilon_S - epsilon_D))))  # example constant value

    # Updated Price equation
    P = sp.exp(c_0) * n ** (1 / (epsilon_S - epsilon_D))
    
    # Business revenue
    Pi = n * P

    # Assume government spending on waste, water, and traffic
    I_W = 100
    I_H = 100
    I_T = 100

    # Waste demand
    W_D = alpha_WD * n 

    # Waste supply
    W_S = alpha_WS * sp.ln(I_W + 1) + C_W

    # Water demand
    H_D = alpha_HD * n

    # Water supply
    H_S = initial_water_supply + alpha_HS * sp.ln(I_H + 1)

    # Traffic demand
    F_D = alpha_FD * n
    F_S = alpha_FS * sp.ln(I_T + 1) + C_T

    # Infrastructure metric
    ILI = ((H_D / H_S) ** 2 + (W_D / W_S) ** 2 + (F_D / F_S) ** 2) ** (1 / 2)

    # Glacier volume as a function of temperature
    Tm = 0 
    v = 1050
    t = -6
    alpha = v / sp.ln(-((t - Tm) - 31))
    V_g = alpha * sp.ln(-((T - Tm) - 31))

    # Resident satisfaction
    r_Pi = 1 / (1 + sp.exp(-0.0001 * (Pi - 45000)))
    r_E = 1 / (1 + sp.exp(20 * (ILI - 0.8)))
    r_V = 1 / (1 + sp.exp(-0.01 * (V_g - 900)))

    Omega = W_Pi * r_Pi + W_E * r_E + W_V * r_V

    # Initialize number of visitors
    visitors_example = 16000  # Initial estimate of visitors

    # Set temperature value (e.g., T = 1)
    temperature_value = -6  # Example temperature value (1°C)

    # Create Omega function (once, outside the loop)
    Omega_func = sp.lambdify([n, T], Omega, 'numpy')  # Use numpy for lambdify

    # Lists to store values for plotting
    visitors_list = [16000]
    revenue_list = []
    ILI_list = []
    glacier_volume_list = [1050]
    price_list = []  # List to store price values
    H_ratio_list = []
    W_ratio_list = []
    T_ratio_list = []
    W_D_list = []
    W_S_list = []
    H_D_list = []
    H_S_list = []
    F_D_list = []
    F_S_list = []
    omega_list = []  # List to track small omega values
    Omega_list = []  # List to track Big Omega values

    for _ in range(1000):  # Max iterations to avoid infinite loops
        # Calculate Omega based on current number of visitors
        Omega_val = Omega_func(visitors_example, temperature_value)

        # Calculate visitor satisfaction omega
        W_P_omega = 0.1
        W_E_omega = 0.1
        W_V_omega = 0.9
        omega = W_P_omega * (1 / (1 + sp.exp(0.03 * (P - 250)))) + W_E_omega * r_E + W_V_omega * r_V
        
        # Store the omega and Omega values
        omega_list.append(omega.subs(n, visitors_example).subs(T, temperature_value).evalf())
        Omega_list.append(Omega_val)

        # Update number of visitors
        new_visitors = n * sp.exp(omega - 0.8)
        
        # Update temperature based on business revenue
        G = Pi * 0.623
        try:
            temperature_value = 0.8 / sp.ln(2) * sp.ln(0.00001*G) + temperature_value
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
        new_visitors_num = new_visitors.subs({n: visitors_example, T: temperature_value}).evalf()
        print(f"New Visitors: {new_visitors_num}")

        # Store the values for plotting (check if real before appending)
        visitors_list.append(visitors_example)
        revenue_list.append(Pi.subs(n, visitors_example).subs(T, temperature_value).evalf())
        ILI_list.append(ILI.subs(n, visitors_example).subs(T, temperature_value).evalf())
        glacier_volume_list.append(V_g.subs(T, temperature_value).evalf())

        # Track the price (P) over time
        price_list.append(P.subs(n, visitors_example).evalf())

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

        # Tracking the individual variables for later plotting
        W_D_list.append(W_D.subs(n, visitors_example).evalf())
        W_S_list.append(W_S.subs(n, visitors_example).evalf())
        H_D_list.append(H_D.subs(n, visitors_example).evalf())
        H_S_list.append(H_S.subs(n, visitors_example).evalf())
        F_D_list.append(F_D.subs(n, visitors_example).evalf())
        F_S_list.append(F_S.subs(n, visitors_example).evalf())

        # Check for convergence
        if abs(new_visitors_num - visitors_example) < 0.0001:
            print(f"Converged to stable number of visitors: {new_visitors_num}")
            break

        # Update number of visitors for next iteration
        visitors_example = int(new_visitors_num)

    # Final calculations with numeric substitutions
    final_Omega = Omega_func(visitors_example, temperature_value)
    print(f"Final Omega: {final_Omega}")

    # Substitute for Pi and r_E with final visitors and temperature value
    final_Pi = Pi.subs(n, visitors_example).subs(T, temperature_value).evalf()
    final_r_E = r_E.subs(n, visitors_example).subs(T, temperature_value).evalf()

    # Plot all variables individually
    plt.figure(figsize=(12, 12))

    # Plot Number of Visitors over Iterations
    plt.subplot(4, 4, 1)
    plt.plot(visitors_list, label="Visitors")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Visitors")
    plt.title("Number of Visitors Over Time")
    plt.grid(True)

    # Plot Business Revenue (Pi) over Iterations
    plt.subplot(4, 4, 2)
    plt.plot(revenue_list, label="Business Revenue (Pi)", color='green')
    plt.xlabel("Iteration")
    plt.ylabel("Business Revenue (Pi)")
    plt.title("Business Revenue Over Time")
    plt.grid(True)

    # Plot Infrastructure Metric (ILI) over Iterations
    plt.subplot(4, 4, 3)
    plt.plot(ILI_list, label="Infrastructure Metric (ILI)", color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Infrastructure Metric (ILI)")
    plt.title("Infrastructure Metric Over Time")
    plt.grid(True)

    # Plot Glacier Volume (V_g) over Iterations
    plt.subplot(4, 4, 4)
    plt.plot(glacier_volume_list, label="Glacier Volume (V_g)", color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Glacier Volume (V_g)")
    plt.title("Glacier Volume Over Time")
    plt.grid(True)

    # Plot Price (P) over Iterations
    plt.subplot(4, 4, 5)
    plt.plot(price_list, label="Price (P)", color='purple')
    plt.xlabel("Iteration")
    plt.ylabel("Price (P)")
    plt.title("Price Over Time")
    plt.grid(True)

    # Plot Water Ratio (H_D / H_S) over Iterations
    plt.subplot(4, 4, 6)
    plt.plot(H_ratio_list, label="Water Ratio (H_D / H_S)", color='orange')
    plt.xlabel("Iteration")
    plt.ylabel("Water Ratio (H_D / H_S)")
    plt.title("Water Ratio Over Time")
    plt.grid(True)

    # Plot Waste Ratio (W_D / W_S) over Iterations
    plt.subplot(4, 4, 7)
    plt.plot(W_ratio_list, label="Waste Ratio (W_D / W_S)", color='cyan')
    plt.xlabel("Iteration")
    plt.ylabel("Waste Ratio (W_D / W_S)")
    plt.title("Waste Ratio Over Time")
    plt.grid(True)

    # Plot Traffic Ratio (F_D / F_S) over Iterations
    plt.subplot(4, 4, 8)
    plt.plot(T_ratio_list, label="Traffic Ratio (F_D / F_S)", color='brown')
    plt.xlabel("Iteration")
    plt.ylabel("Traffic Ratio (F_D / F_S)")
    plt.title("Traffic Ratio Over Time")
    plt.grid(True)

    # Plot Waste Demand (W_D) over Iterations
    plt.subplot(4, 4, 9)
    plt.plot(W_D_list, label="Waste Demand (W_D)", color='yellow')
    plt.xlabel("Iteration")
    plt.ylabel("Waste Demand (W_D)")
    plt.title("Waste Demand Over Time")
    plt.grid(True)

    # Plot Waste Supply (W_S) over Iterations
    plt.subplot(4, 4, 10)
    plt.plot(W_S_list, label="Waste Supply (W_S)", color='pink')
    plt.xlabel("Iteration")
    plt.ylabel("Waste Supply (W_S)")
    plt.title("Waste Supply Over Time")
    plt.grid(True)

    # Plot Omega over Iterations (Big Omega)
    plt.subplot(4, 4, 1)
    plt.plot(Omega_list, label="Big Omega (Resident Satisfaction)")
    plt.xlabel("Iteration")
    plt.ylabel("Big Omega")
    plt.title("Big Omega Over Time")
    plt.grid(True)

    # Plot omega over Iterations (small omega)
    plt.subplot(4, 4, 2)
    plt.plot(omega_list, label="Small omega (Visitor Satisfaction)", color='orange')
    plt.xlabel("Iteration")
    plt.ylabel("Small omega")
    plt.title("Small omega Over Time")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return {
        "Resident Satisfaction (Omega)": final_Omega,
        "Business Revenue (Pi)": final_Pi,
        "Infrastructure Metric (r)": final_r_E,
    }

# Example usage
result = sustainable_tourism_model()
print(result)
