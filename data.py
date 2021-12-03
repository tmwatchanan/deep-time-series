import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader.yahoo.daily import YahooDailyReader

df = YahooDailyReader('NFLX', start='05/23/02', end='12/01/21').read()
df = df.reset_index()
df = df[['Date', 'High', 'Open', 'Low', 'Close', 'Adj Close', 'Volume']]
df.to_csv('data/netflix/NFLX.csv', index=False)

plt.figure(figsize=(16, 8))
plt.plot(df['Date'], df['Close'], color='black')

ax = plt.gca()
ax.set_xlim([datetime.datetime(2002, 5, 23), datetime.datetime(2021, 12, 1)])
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()  # Rotation

plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.title('NFLX Closing Price from 05/23/2002 to 12/01/2021')

plt.savefig('results/netflix/NFLX.png', dpi=300, bbox_inches='tight')

print(df.head(10))
print(df.tail(10))
