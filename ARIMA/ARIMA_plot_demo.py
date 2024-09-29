import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# Load the Monthly Sunspots dataset


def fetch_and_save_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
    data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
    data.to_csv("github_user_content.csv")

def load_data():
    data = pd.read_csv("github_user_content.csv")
    return data


data = load_data()

# Time plot
plt.figure(figsize=(7, 5))
plt.plot(data.index, data['Sunspots'], marker='o', linestyle='-', markersize=5)
plt.xlabel('Date')
plt.ylabel('Number of Sunspots')
plt.title('Monthly Sunspots Time Plot')
plt.grid(True)
plt.show()

# Histogram and Density Plot
plt.figure(figsize=(7, 5))
sns.histplot(data['Sunspots'], kde=True)
plt.xlabel('Number of Sunspots')
plt.ylabel('Frequency')
plt.title('Histogram and Density Plot')
plt.grid(True)
plt.show()

# Autocorrelation Plot
plt.figure(figsize=(7,5))
plot_acf(data['Sunspots'], lags=50)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Plot')
plt.grid(True)
plt.show()


