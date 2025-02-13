import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter 
import numpy as np
import stumpy

# Adjust figure size
plt.rcParams["figure.figsize"] = [20, 12]
plt.rcParams['xtick.direction'] = 'out'

def clean_data_trend(df, column, lower_percentile=0.01, upper_percentile=0.99):
    """
    Pulisce i dati identificando come anomale le variazioni relative eccessive,
    sostituendo con NaN i valori anomali e rimuovendo le righe con NaN.

    Parametri:
    - df: DataFrame contenente i dati
    - column: Nome della colonna da pulire
    - lower_percentile: Percentile minimo per le variazioni accettabili (default: 1° percentile)
    - upper_percentile: Percentile massimo per le variazioni accettabili (default: 99° percentile)

    Ritorna:
    - DataFrame pulito
    """
    # Calcola le variazioni relative in percentuale rispetto al valore precedente
    df['Variazione'] = df[column].pct_change()

    # Calcola i limiti delle variazioni accettabili
    lower_limit = df['Variazione'].quantile(lower_percentile)
    upper_limit = df['Variazione'].quantile(upper_percentile)
    print(f"Limiti variazioni percentili: {lower_limit} - {upper_limit}")

    # Sostituisci con NaN i valori con variazioni fuori scala
    df[column] = df.apply(
        lambda row: np.nan if (row['Variazione'] < lower_limit or row['Variazione'] > upper_limit) 
                    else row[column], axis=1
    )

    # Rimuovi la colonna temporanea delle variazioni
    df = df.drop(columns=['Variazione'])
    
    # Rimuovi le righe contenenti NaN
    df = df.dropna(subset=[column])

    return df


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

def find_k_highest_positions(matrix_profile, k):
    # Trova le posizioni dei k valori più alti nel matrix profile
    idxs = np.argpartition(-matrix_profile, k)[:k]
    idxs = idxs[np.argsort(-matrix_profile[idxs])]
    return idxs

def find_anomalous_subsequences(df, indices, m):
    anomalous_subsequences = []
    for idx in indices:
        start = idx
        end = idx + m
        anomalous_subsequences.append(df['Consumi GF [kWh]'].values[start:end])
    return anomalous_subsequences

import matplotlib.pyplot as plt

def plot_anomalies_with_highlighting(T_df, idxs, Q_df, title='Time Series', xlabel='Time', ylabel='Value'):
    """
    Plot the time series with highlighted subsequences using different colors.

    Parameters:
    - T_df (pandas.Series or np.array): The time series data to plot.
    - idxs (list): List of starting indices for the subsequences.
    - Q_df (np.array): The subsequence that needs to be compared (e.g., a query subsequence).
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(12, 6))
    plt.suptitle(title, fontsize='30')
    plt.xlabel(xlabel, fontsize='20')
    plt.ylabel(ylabel, fontsize='20')

    # Plot the entire time series
    plt.plot(T_df, label='Time Series', color='blue')

    # Highlight subsequences with different colors
    for idx in idxs:
        # Extract the subsequence from the time series based on the index
        subseq = T_df.values[idx:idx+len(Q_df)]
        
        # Generate a random color for each subsequence
        color = plt.cm.jet(np.random.rand())  # Generate a random color from the 'jet' colormap
        plt.plot(range(idx, idx + len(subseq)), subseq, lw=2, color=color)

    # Show the plot
    plt.show()

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
# Pulizia dei dati basata sulle variazioni relative
df = clean_data_trend(df, 'Consumi GF [kWh]')


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

# Trova le posizioni dei k valori più alti
highest_positions = find_k_highest_positions(approx_P, k)

print("Posizioni dei k valori più alti nel matrix profile:", highest_positions)

# Individua le sottosequenze anomale usando le posizioni trovate
anomalous_subsequences = find_anomalous_subsequences(df, highest_positions, m)

print("Sottosequenze anomale:")
for i, seq in enumerate(anomalous_subsequences, 1):
    print(f"Sottosequenza {i}: {seq}")

plot_anomalies_with_highlighting(df['Consumi GF [kWh]'], highest_positions, df['Consumi GF [kWh]'].values[:m])