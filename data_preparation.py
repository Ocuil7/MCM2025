import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_country_medal_counts(
    country, year
):
    medal_counts = pd.read_csv('2025_Problem_C_Data/summerOly_medal_counts_utf8.csv')

    year_medal_table = medal_counts[medal_counts["Year"] == year]
    country_medal_counts = year_medal_table[year_medal_table["NOC"] == country]

    if not country_medal_counts.empty:
        # Return the medal counts as a 4D list
        gold_counts = (
            country_medal_counts["Gold"].values[0] / year_medal_table["Gold"].sum()
        )
        silver_counts = (
            country_medal_counts["Silver"].values[0] / year_medal_table["Silver"].sum()
        )
        bronze_counts = (
            country_medal_counts["Bronze"].values[0] / year_medal_table["Bronze"].sum()
        )
        total_counts = (
            country_medal_counts["Total"].values[0] / year_medal_table["Total"].sum()
        )
        return [gold_counts, silver_counts, bronze_counts, total_counts]

    logging.warning(f"No medal counts found for {country} in {year}.")
    return [0, 0, 0, 0]

def get_host_country(year):
    hosts = pd.read_csv("2025_Problem_C_Data/summerOly_hosts_utf8.csv")
    host_country = hosts[hosts["Year"] == year]["Host"].values
    if len(host_country) > 0:
        host_country = host_country[0].split("(")[0].strip()
        if len(host_country.split(',')) > 1:
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
    ret =[]
    for i in year_programs_table[:-3]:
        
        if isinstance(i, str) and "[" in i :
            if "Included" in i:
                i= 0
            else:
                i = int(i.split("[s3]")[0])
        else:
            i = int(i)
        ret.append(i)
    
    return ret


medal_counts = pd.read_csv('2025_Problem_C_Data/summerOly_medal_counts_utf8.csv')
countries = medal_counts["NOC"].unique()
years = medal_counts["Year"].unique()
    # Get the medal counts for each country and year
data = {}
for country in countries:
    country_dict = {}
    
    for year in years:
        # Check if there is a entry with NOC and Year
        if not medal_counts[(medal_counts["NOC"] == country) & (medal_counts["Year"] == year)].empty:
            was_host = int(get_host_country(year) == country)
            will_host = int(get_host_country(year + 4) == country)
            
            additional_info = [was_host, will_host, *get_programs_vector(year)]
            country_dict[year] = [get_country_medal_counts(country, year), additional_info]
            
    data[country]= country_dict

# Data List: Go through every country and year, if year + 4 exists then x is year and y is year + 4
data_list = []
for country in data:
    for year in data[country]:
        if year + 4 in data[country]:
            x = data[country][year][0]
            y = data[country][year + 4][0]
            additional_info = data[country][year][1]
            data_list.append([x, y, additional_info])

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

# Create the dataset and dataloader
dataset = MedalDataset(data_list)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create train, val, and test splits
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)
# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
