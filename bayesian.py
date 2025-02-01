import pymc as pm
import numpy as np
import arviz as az

import matplotlib.pyplot as plt

import json

country = "United States"

# Load the data
data = json.load(open("countries_yearwise_states.json"))
country_data = data[country]
# Make the keys integers
country_data = {int(k): v for k, v in country_data.items()}


# Simulated Data (replace these with your actual data)
# Example for a single country over multiple years
n_years = len(country_data)
n_sports = len(country_data[1896][1][2])  # Subtracting the host information

# Past data 
# medals = np.random.poisson(lam=3, size=(n_years, 3))  # [G, S, B]
# host_next_year = np.random.choice([0, 1], size=n_years, p=[0.8, 0.2])
# participants = np.random.randint(5, 50, size=(n_years, n_sports))
# medals_by_sport = np.random.poisson(lam=2, size=(n_years, n_sports))

all_medals = np.array([country_data[year][0] for year in country_data] )
all_was_hosting = [country_data[year][1][0] for year in country_data]
all_participants = [country_data[year][1][2] for year in country_data]
all_medals_by_sport = [country_data[year][1][3] for year in country_data]

all_was_hosting = [0] + all_was_hosting  # Add a dummy value for the first year
all_participants = [[0] * n_sports, *all_participants]  # Add a dummy value for the first year
all_medals_by_sport = [[0] * n_sports, *all_medals_by_sport]  # Add a dummy value for the first year

# print(len(all_medals), len(all_was_hosting), len(all_participants), len(all_medals_by_sport))
# print([len(x) for x in all_participants])

all_medals = np.array(all_medals)
all_was_hosting = np.array(all_was_hosting)
all_participants = np.array(all_participants)
all_medals_by_sport = np.array(all_medals_by_sport)

# Past data
medals = all_medals[:-3 ]  # [G, S, B]
host_next_year = all_was_hosting[:-4]
participants = all_participants[:-4]
medals_by_sport = all_medals_by_sport[:-4]

assert medals.shape[0] == host_next_year.shape[0] == participants.shape[0] == medals_by_sport.shape[0]

# Data for the current year
current_year_participants = all_participants[-4]
current_year_host = all_was_hosting[-4]

current_year_true_medals = all_medals[-3]


# Bayesian Model
def create_bayesian_medal_model():
    with pm.Model() as model:
        # Priors for medal counts (gold, silver, bronze)
        gold_base = pm.Normal("gold_base", mu=3, sigma=2)
        silver_base = pm.Normal("silver_base", mu=3, sigma=2)
        bronze_base = pm.Normal("bronze_base", mu=3, sigma=2)

        # Effect of hosting
        host_effect_gold = pm.Normal("host_effect_gold", mu=1, sigma=0.5)
        host_effect_silver = pm.Normal("host_effect_silver", mu=1, sigma=0.5)
        host_effect_bronze = pm.Normal("host_effect_bronze", mu=1, sigma=0.5)

        # Effects of participants in different sports
        sport_effect_gold = pm.Normal("sport_effect_gold", mu=0, sigma=0.1, shape=n_sports)
        sport_effect_silver = pm.Normal("sport_effect_silver", mu=0, sigma=0.1, shape=n_sports)
        sport_effect_bronze = pm.Normal("sport_effect_bronze", mu=0, sigma=0.1, shape=n_sports)

        host_next_year_shared = pm.Data("host_next_year", host_next_year, mutable=True)
        participants_shared = pm.Data("participants", participants, mutable=True)
        
        # Expected values for medals
        gold_mu = (
            gold_base
            + host_effect_gold * host_next_year_shared
            + pm.math.dot(participants_shared, sport_effect_gold)
        )
        silver_mu = (
            silver_base
            + host_effect_silver * host_next_year_shared
            + pm.math.dot(participants_shared, sport_effect_silver)
        )
        bronze_mu = (
            bronze_base
            + host_effect_bronze * host_next_year_shared
            + pm.math.dot(participants_shared, sport_effect_bronze)
        )

        # Observations
        gold_obs = pm.Poisson("gold_obs", mu=pm.math.abs(gold_mu), observed=medals[:, 0])
        silver_obs = pm.Poisson("silver_obs", mu=pm.math.abs(silver_mu), observed=medals[:, 1])
        bronze_obs = pm.Poisson("bronze_obs", mu=pm.math.abs(bronze_mu), observed=medals[:, 2])

    return model

# Create the model
model = create_bayesian_medal_model()

# Sampling
with model:
    trace = pm.sample(5000, tune=1000, return_inferencedata=True, target_accept=0.99)

# Posterior Prediction
def predict_medals(trace, current_year_participants, current_year_host):
    with model:
        pm.set_data({
            "host_next_year": np.array([current_year_host]),
            "participants": current_year_participants.reshape(1, -1),
        })

        posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["gold_obs", "silver_obs", "bronze_obs"])
        
    return posterior_predictive

posterior_predictive = predict_medals(trace, current_year_participants, current_year_host)

# Analyze Results
# az.plot_posterior(posterior_predictive, kind="hist")
az.plot_ppc(posterior_predictive, figsize=(12, 6))
# save the plot

plt.savefig("posterior_predictive.png")
print("Predicted Medals Distribution:")
# print(az.summary(posterior_predictive))
print(f"True Medals: {current_year_true_medals}")
# print(f"{posterior_predictive['posterior_predictive']['gold_obs']}")
print(f"Predicted Medals: {np.mean(posterior_predictive['posterior_predictive']['gold_obs'])}, {np.mean(posterior_predictive['posterior_predictive']['silver_obs'])}, {np.mean(posterior_predictive['posterior_predictive']['bronze_obs'])}")