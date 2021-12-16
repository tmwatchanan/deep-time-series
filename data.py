import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader.yahoo.daily import YahooDailyReader


def fetch_data():
    df = YahooDailyReader('NFLX', start='05/23/02', end='12/01/21').read()
    df = df.reset_index()
    df = df[['Date', 'High', 'Open', 'Low', 'Close', 'Adj Close', 'Volume']]
    df.to_csv('data/netflix/NFLX.csv', index=False)
    return df

def format_date_axis():
    ax = plt.gca()
    ax.set_xlim([datetime.datetime(2002, 5, 23), datetime.datetime(2021, 12, 1)])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()  # Rotation

def plot_data(df):
    plt.figure(figsize=(16, 8))
    plt.plot(df['Date'], df['Close'], color='black')

    format_date_axis()

    plt.xlabel('Date')
    plt.ylabel('Closing Price ($)')
    plt.title('NFLX Closing Price from 05/23/2002 to 12/01/2021')

    plt.savefig('results/netflix/NFLX.pdf', dpi=300, bbox_inches='tight')

    print(df.head(10))
    print(df.tail(10))


def plot_attributes(df):
    attribute_names = ['Open', 'Close', 'Low', 'High', 'Volume']
    fig, axs = plt.subplots(len(attribute_names), 1, figsize=(12, 8), sharex=True)
    for i, attribute_name in enumerate(attribute_names):
        axs[i].plot(df['Date'], df[attribute_name], color='black')
        format_date_axis()
        axs[i].set_xlabel('Date')
        axs[i].set_ylabel(attribute_name)
    plt.savefig('results/netflix/NFLX-attributes.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    df = fetch_data()
    # plot_data(df)
    plot_attributes(df)
