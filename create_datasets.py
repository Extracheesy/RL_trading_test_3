#!/usr/bin/env python
# coding: utf-8

# # Download and store data

# This notebook contains information on downloading the Quandl Wiki stock prices and a few other sources that we use throughout the book. 

# ## Imports & Settings

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


from pathlib import Path
import requests
from io import BytesIO
from zipfile import ZipFile, BadZipFile

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.datasets import fetch_openml

pd.set_option('display.expand_frame_repr', False)


# ## Set Data Store path

# Modify path if you would like to store the data elsewhere and change the notebooks accordingly

# In[93]:


DATA_STORE = Path('assets.h5')


# ## Quandl Wiki Prices

# [Quandl](https://www.quandl.com/) makes available a [dataset](https://www.quandl.com/databases/WIKIP/documentation) with stock prices, dividends and splits for 3000 US publicly-traded companies. Quandl decided to discontinue support in favor of its commercial offerings but the historical data are still useful to demonstrate the application of the machine learning solutions in the book, just ensure you implement your own algorithms on current data.
# 
# > As of April 11, 2018 this data feed is no longer actively supported by the Quandl community. We will continue to host this data feed on Quandl, but we do not recommend using it for investment or analysis.

# 1. Follow the instructions to create a free [Quandl]([Quandl](https://www.quandl.com/)) account
# 2. [Download](https://www.quandl.com/databases/WIKIP/usage/export) the entire WIKI/PRICES data
# 3. Extract the .zip file,
# 4. Move to this directory and rename to wiki_prices.csv
# 5. Run the below code to store in fast HDF format (see [Chapter 02 on Market & Fundamental Data](../02_market_and_fundamental_data) for details).

# In[8]:


df = (pd.read_csv('wiki_prices.csv',
                 parse_dates=['date'],
                 index_col=['date', 'ticker'],
                 infer_datetime_format=True)
     .sort_index())

print(df.info(null_counts=True))
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/prices', df)


# ### Wiki Prices Metadata

# As of writing, the following instructions no longer work because Quandl changed its API:
# 
# > 1. Follow the instructions to create a free [Quandl]([Quandl](https://www.quandl.com/)) account if you haven't done so yet
# > 2. Find link to download wiki metadata under Companies](https://www.quandl.com/databases/WIKIP/documentation) or use the download link with your API_KEY: https://www.quandl.com/api/v3/databases/WIKI/metadata?api_key=<API_KEY>
# > 3. Extract the .zip file,
# > 4. Move to this directory and rename to wiki_stocks.csv
# > 5. Run the following code to store in fast HDF format

# Instead, load the file `wiki_stocks.csv` as described and store in HDF5 format.

# In[5]:


df = pd.read_csv('wiki_stocks.csv')
# no longer needed
# df = pd.concat([df.loc[:, 'code'].str.strip(),
#                 df.loc[:, 'name'].str.split('(', expand=True)[0].str.strip().to_frame('name')], axis=1)

print(df.info(null_counts=True))
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/stocks', df)


# ## S&P 500 Prices

# The following code downloads historical S&P 500 prices from FRED (only last 10 years of daily data is freely available)

# In[6]:


df = web.DataReader(name='SP500', data_source='fred', start=2009).squeeze().to_frame('close')
print(df.info())
with pd.HDFStore(DATA_STORE) as store:
    store.put('sp500/fred', df)


# Alternatively, download S&P500 data from [stooq.com](https://stooq.com/q/?s=%5Espx&c=1d&t=l&a=lg&b=0); at the time of writing the data was available since 1789. You can switch from Polish to English on the lower right-hand side.
# 
# We store the data from 1950-2020:

# In[9]:


# sp500_stooq = (pd.read_csv('^spx_d.csv', index_col=0,
#                      parse_dates=True).loc['1950':'2019'].rename(columns=str.lower))
# print(sp500_stooq.info())


# In[10]:


# with pd.HDFStore(DATA_STORE) as store:
#     store.put('sp500/stooq', sp500_stooq)


# ### S&P 500 Constituents

# The following code downloads the current S&P 500 constituents from [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).

# In[13]:


url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
df = pd.read_html(url, header=0)[0]


# In[14]:


df.head()


# In[15]:


df.columns = ['ticker', 'name', 'sec_filings', 'gics_sector', 'gics_sub_industry',
              'location', 'first_added', 'cik', 'founded']
df = df.drop('sec_filings', axis=1).set_index('ticker')


# In[16]:


print(df.info())


# In[17]:


with pd.HDFStore(DATA_STORE) as store:
    store.put('sp500/stocks', df)


# ## Metadata on US-traded companies

# The following downloads several attributes for [companies](https://www.nasdaq.com/screening/companies-by-name.aspx) traded on NASDAQ, AMEX and NYSE
# 
# > Update: unfortunately, NASDAQ has disabled automatic downloads. However, you can still access and manually download the files at the below URL when you fill in the exchange names. So for AMEX, URL becomes `https://www.nasdaq.com/market-activity/stocks/screener?exchange=AMEX&letter=0&render=download`.
# >

# In[12]:


# no longer works!
url = 'https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange={}&render=download'
exchanges = ['NASDAQ', 'AMEX', 'NYSE']
df = pd.concat([pd.read_csv(url.format(ex)) for ex in exchanges]).dropna(how='all', axis=1)
df = df.rename(columns=str.lower).set_index('symbol').drop('summary quote', axis=1)
df = df[~df.index.duplicated()]
print(df.info()) 


# In[13]:


df.head()


# ### Convert market cap information to numerical format

# Market cap is provided as strings so we need to convert it to numerical format.

# In[14]:


mcap = df[['marketcap']].dropna()
mcap['suffix'] = mcap.marketcap.str[-1]
mcap.suffix.value_counts()


# Keep only values with value units:

# In[15]:


mcap = mcap[mcap.suffix.str.endswith(('B', 'M'))]
mcap.marketcap = pd.to_numeric(mcap.marketcap.str[1:-1])
mcaps = {'M': 1e6, 'B': 1e9}
for symbol, factor in mcaps.items():
    mcap.loc[mcap.suffix == symbol, 'marketcap'] *= factor
mcap.info()


# In[16]:


df['marketcap'] = mcap.marketcap
df.marketcap.describe(percentiles=np.arange(.1, 1, .1).round(1)).apply(lambda x: f'{int(x):,d}')


# ### Store result

# The file `us_equities_meta_data.csv` contains a version of the data used for many of the examples. Load using 
# ```
# df = pd.read_csv('us_equities_meta_data.csv')
# ```
# and proceed to store in HDF5 format.

# In[5]:


df = pd.read_csv('us_equities_meta_data.csv')
df.info()


# In[7]:


with pd.HDFStore(DATA_STORE) as store:
    store.put('us_equities/stocks', df.set_index('ticker'))


# ## MNIST Data

# In[36]:


mnist = fetch_openml('mnist_784', version=1)


# In[37]:


print(mnist.DESCR)


# In[38]:


mnist.keys()


# In[39]:


mnist_path = Path('mnist')
if not mnist_path.exists():
    mnist_path.mkdir()


# In[40]:


np.save(mnist_path / 'data', mnist.data.astype(np.uint8))
np.save(mnist_path / 'labels', mnist.target.astype(np.uint8))


# ## Fashion MNIST Image Data

# We will use the Fashion MNIST image data created by [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) for some demonstrations.

# In[12]:


fashion_mnist = fetch_openml(name='Fashion-MNIST')


# In[13]:


print(fashion_mnist.DESCR)


# In[33]:


label_dict = {0: 'T-shirt/top',
              1: 'Trouser',
              2: 'Pullover',
              3: 'Dress',
              4: 'Coat',
              5: 'Sandal',
              6: 'Shirt',
              7: 'Sneaker',
              8: 'Bag',
              9: 'Ankle boot'}


# In[34]:


fashion_path = Path('fashion_mnist')
if not fashion_path.exists():
    fashion_path.mkdir()


# In[35]:


pd.Series(label_dict).to_csv(fashion_path / 'label_dict.csv', index=False, header=None)


# In[31]:


np.save(fashion_path / 'data', fashion_mnist.data.astype(np.uint8))
np.save(fashion_path / 'labels', fashion_mnist.target.astype(np.uint8))


# ## Bond Price Indexes

# The following code downloads several bond indexes from the Federal Reserve Economic Data service ([FRED](https://fred.stlouisfed.org/))

# In[27]:


securities = {'BAMLCC0A0CMTRIV'   : 'US Corp Master TRI',
              'BAMLHYH0A0HYM2TRIV': 'US High Yield TRI',
              'BAMLEMCBPITRIV'    : 'Emerging Markets Corporate Plus TRI',
              'GOLDAMGBD228NLBM'  : 'Gold (London, USD)',
              'DGS10'             : '10-Year Treasury CMR',
              }

df = web.DataReader(name=list(securities.keys()), data_source='fred', start=2000)
df = df.rename(columns=securities).dropna(how='all').resample('B').mean()

with pd.HDFStore(DATA_STORE) as store:
    store.put('fred/assets', df)

