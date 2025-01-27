import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
import torch

from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from datetime import date

import json

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_country_medal_counts(country, year, medal_counts_df):
    # medal_counts_df = pd.read_csv("2025_Problem_C_Data/summerOly_medal_counts_utf8.csv")

    year_medal_table = medal_counts_df[medal_counts_df["Year"] == year]
    country_medal_counts = year_medal_table[year_medal_table["NOC"] == country]

    if not country_medal_counts.empty:
        # Return the medal counts as a 4D list
        gold_counts = int(country_medal_counts["Gold"].values[0])
        silver_counts = int(country_medal_counts["Silver"].values[0])
        bronze_counts = int(country_medal_counts["Bronze"].values[0])
        total_counts = int(country_medal_counts["Total"].values[0])
        return [gold_counts, silver_counts, bronze_counts, total_counts]

    logging.warning(f"No medal counts found for {country} in {year}.")
    return [0, 0, 0, 0]


def get_host_country(year, hosts_df):
    host_country = hosts_df[hosts_df["Year"] == year]["Host"].values
    if len(host_country) > 0:
        host_country = host_country[0].split("(")[0].strip()
        if len(host_country.split(",")) > 1:
            return host_country.split(",")[1]
        else:
            return host_country
    logging.warning(f"No host country found for {year}.")
    return "Not Found"


def get_programs_vector(year):
    programs = pd.read_csv("2025_Problem_C_Data/summerOly_programs_utf8.csv")

    # Fill NA values with 0
    programs = programs.fillna(0)

    if str(year) not in programs.columns:
        logging.warning(f"No programs found for {year}.")
        return [[], [], []]
    year_programs_table = programs[str(year)]

    # Get the column except last 3 rows
    # year_programs_table = [int(i) for i in year_programs_table[:-3].tolist() if ('[s3]' not in i) else int(i.split("[s3]")[0])]
    ret = []
    for i in year_programs_table[:-3]:

        if isinstance(i, str) and "[" in i:
            if "Included" in i:
                i = 0
            else:
                i = int(i.split("[s3]")[0])
        else:
            i = int(i)
        ret.append(i)

    return ret


def get_country_sports_participation_vector(country, year, athletes_df):
    """
    Function that reads the atheletes data and returns a vector with the sports that a country is participating in a given year.
    """
    # chunk_size = 10000
    # chunks = pd.read_csv('2025_Problem_C_Data/summerOly_athletes_utf8.csv', chunksize=chunk_size)
    # athletes = pd.concat(chunks, ignore_index=True)
    
    all_sports = athletes_df["Sport"].unique()
    sports_vector = [0] * len(all_sports)

    year_athletes = athletes_df[athletes_df["Year"] == year]
    country_athletes = year_athletes[year_athletes["Team"] == country]

    if not country_athletes.empty:
        sports = country_athletes["Sport"].unique()

        # Count the number of athletes in each sport
        for sport in sports:
            sports_vector[list(all_sports).index(sport)] = int(len(
                country_athletes[country_athletes["Sport"] == sport]
            ))

        return sports_vector, all_sports

    logging.warning(f"No athletes found for {country} in {year}.")
    return [], []


def get_country_sports_medals_vector(country, year, athletes_df):
    """
    Function that reads the medal counts data and returns a vector with the sports that a country has won medals in a given year.
    """
    # athletes_df = pd.read_csv("2025_Problem_C_Data/summerOly_athletes_utf8.csv")
    all_sports = athletes_df["Sport"].unique()
    sports_vector = [[0,0,0]] * len(all_sports)

    year_athletes = athletes_df[athletes_df["Year"] == year]
    country_athletes = year_athletes[year_athletes["Team"] == country]

    if not country_athletes.empty:
        sports = country_athletes["Sport"].unique()

        for sport in sports:
            # Count the entries in Medal column for each sport with Gold, Silver, Bronze
            sports_vector[list(all_sports).index(sport)] = [
                len(
                    country_athletes[
                        (country_athletes["Sport"] == sport)
                        & (country_athletes["Medal"] == "Gold")
                    ]
                ),
                len(
                    country_athletes[
                        (country_athletes["Sport"] == sport)
                        & (country_athletes["Medal"] == "Silver")
                    ]
                ),
                len(
                    country_athletes[
                        (country_athletes["Sport"] == sport)
                        & (country_athletes["Medal"] == "Bronze")
                    ]
                ),
            ]

        return sports_vector, all_sports

    logging.warning(f"No athletes found for {country} in {year}.")
    return [], []

def get_countries_years():
    medal_counts = pd.read_csv('2025_Problem_C_Data/summerOly_medal_counts_utf8.csv')
    countries = medal_counts["NOC"].unique()
    years = medal_counts["Year"].unique()
    return countries, years

def get_country_year_data(country, year):
    athletes_df = pd.read_csv("2025_Problem_C_Data/summerOly_athletes_utf8.csv")
    medal_counts_df = pd.read_csv('2025_Problem_C_Data/summerOly_medal_counts_utf8.csv')
    hosts_df = pd.read_csv("2025_Problem_C_Data/summerOly_hosts_utf8.csv")
    
    country_medal_counts = get_country_medal_counts(country, year, medal_counts_df)
    was_host = int(get_host_country(year, hosts_df) == country)
    will_host = int(get_host_country(year + 4, hosts_df) == country)
    
    country_sports_part, _ = get_country_sports_participation_vector(country, year, athletes_df)
    country_sports_medals, _ = get_country_sports_medals_vector(country, year, athletes_df)
    
    # Sum the medals for each sport
    country_sports_medals = [ sum(sport) for sport in country_sports_medals]
    
    additional_info = [was_host, will_host, *country_sports_part, *country_sports_medals]
    
    return country_medal_counts, additional_info

def get_data():
    athletes_df = pd.read_csv("2025_Problem_C_Data/summerOly_athletes_utf8.csv")
    medal_counts_df = pd.read_csv('2025_Problem_C_Data/summerOly_medal_counts_utf8.csv')
    hosts_df = pd.read_csv("2025_Problem_C_Data/summerOly_hosts_utf8.csv")
    
    countries = medal_counts_df["NOC"].unique()
    years = medal_counts_df["Year"].unique()
        # Get the medal counts for each country and year
    data = {}
    for country in countries:
        country_dict = {}

        for year in years:
            # Check if there is a entry with NOC and Year
            if not medal_counts_df[(medal_counts_df["NOC"] == country) & (medal_counts_df["Year"] == year)].empty:
                was_host = int(get_host_country(year, hosts_df) == country)
                will_host = int(get_host_country(year + 4, hosts_df) == country)

                country_sports_part, _ = get_country_sports_participation_vector(country, year, athletes_df)
                
                if len(country_sports_part) == 0:
                    continue
                
                country_sports_medals, _ = get_country_sports_medals_vector(country, year, athletes_df) 
                # Sum the medals for each sport
                country_sports_medals = [ int(sum(sport)) for sport in country_sports_medals]
                
                additional_info = [was_host, will_host, country_sports_part, country_sports_medals]
                country_dict[int(year)] = [get_country_medal_counts(country, year, medal_counts_df), additional_info]

        data[str(country)]= country_dict
    
    json.dump(data, open("countries_yearwise_states.json", "w"))
    return data

def get_data_list(data):
    # Data List: Go through every country and year, if year + 4 exists then x is year and y is year + 4
    data_list = []
    for country in data:
        for year in data[country]:
            if year + 4 in data[country]:
                x = data[country][year]
                y = data[country][year + 4]
                # additional_info = data[country][year][1]
                data_list.append([x, y])
    
    json.dump(data_list, open("data_list.json", "w"))

    return data_list

get_data_list(get_data())

# Create a DataLoader for the data
class MedalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, additional_info = self.data[idx]
        # print(f"x: {x}, y: {y}, additional_info: {additional_info}")
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "additional_info": torch.tensor(additional_info, dtype=torch.float32),
        }


# dataset = MedalDataset(data_list)


# # Create train, val, and test splits
# train_size = int(0.7 * len(dataset))
# val_size = int(0.15 * len(dataset))
# test_size = len(dataset) - train_size - val_size

# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
#     dataset, [train_size, val_size, test_size]
# )
# # Create dataloaders
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
