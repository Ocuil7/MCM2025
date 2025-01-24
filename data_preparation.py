import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader

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
    return None

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
        
        if isinstance(i, str) and "[s3]" in i:
            i = i.split("[s3]")[0]
        else:
            i = int(i)
        ret.append(i)
    
    return ret


medal_counts = pd.read_csv('2025_Problem_C_Data/summerOly_medal_counts_utf8.csv')
countries = medal_counts["NOC"].unique()
years = medal_counts["Year"].unique()
    # Get the medal counts for each country and year
data = []
for country in countries:
    country_dict = {}
    
    for year in years:
        # Check if there is a entry with NOC and Year
        if not medal_counts[(medal_counts["NOC"] == country) & (medal_counts["Year"] == year)].empty:
            country_dict[str(year)] = get_country_medal_counts(country, year)
    
    data.append({country: country_dict})


# Create a DataLoader