import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter 
import numpy as np
import stumpy

# Adjust figure size
plt.rcParams["figure.figsize"] = [20, 12]
plt.rcParams['xtick.direction'] = 'out'

def plot_data_and_MP(df, approx_P):

    # Adjust lengths of the approximation (approx_P) to match the original data
    # We'll pad the approximation with NaNs to align it with the original data's length.
    approx_P_padded = np.full_like(df['Consumi GF [kWh]'].values, np.nan)
    approx_P_padded[m-1:m-1+len(approx_P)] = approx_P  # Align the approximation with the correct index

    # Create separate subplots for the original data and the approximation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

    # Plot the original data in the first subplot
    ax1.plot(df['Data e Ora'], df['Consumi GF [kWh]'], color='blue', linewidth=1)
    ax1.set_ylabel('Consumi GF [kWh]')
    ax1.set_title('Consumi Gruppi Frigo (Original Data)')
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(DateFormatter("%m-%Y"))
    ax1.tick_params(axis='x', rotation=45)

    # Plot the approximation data in the second subplot
    ax2.plot(df['Data e Ora'], approx_P_padded, color='red', linewidth=1)
    ax2.set_ylabel('Approximation (P)')
    ax2.set_title('Approximation of Consumi Gruppi Frigo (SCRUMP)')
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(DateFormatter("%m-%Y"))
    ax2.tick_params(axis='x', rotation=45)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

def find_k_biggest(distance_profile, k):

    idxs = np.argpartition(distance_profile, -k)[-k:]
    idxs = idxs[np.argsort(distance_profile[idxs])]
    return idxs

#main

# Read and prepare the data
df = pd.read_csv("data/Input_Consumi.csv", sep=';')
df['Data e Ora'] = pd.to_datetime(df['Data e Ora'], format='%d/%m/%Y %H:%M')
df = df.rename(columns={'RM016_01_QEGF_S128137_Mult01_AI011 (kWh)': 'Consumi GF [kWh]'})

# Sort the data by date to ensure proper line plotting
df = df.sort_values('Data e Ora')

# Remove any duplicate timestamps if they exist
df = df.drop_duplicates('Data e Ora')

# Replace any 0 values with NaN in 'Consumi GF [kWh]'
df['Consumi GF [kWh]'] = df['Consumi GF [kWh]'].apply(lambda x: np.nan if x < 3000 else x)

# Define the 'm' value for SCRUMP (based on a 1-week sliding window of hourly data)
m = 24 * 7

# Set a random seed for reproducibility of results (optional)
seed = np.random.randint(100000)  
np.random.seed(seed)

# Perform SCRUMP (Similarity-based Time Series Matching) to get the approximation
approx = stumpy.scrump(df['Consumi GF [kWh]'], m, percentage=0.01, pre_scrump=True)
approx.update()  # Update SCRUMP approximation
approx_P = approx.P_  # Extract the approximation result
approx_I = approx.I_  # Extract the index result

# plot_data_and_MP(df, approx_P)
k=10

smallest_indices = find_k_biggest(approx_I , k)

print(smallest_indices)

