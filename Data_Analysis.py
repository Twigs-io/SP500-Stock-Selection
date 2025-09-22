import yfinance as yf
import pandas as pd
import glob
import os
import requests

def Get_html():
    #get data from wikipedia on the s&p 500 companies
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    filename = ".\Final_DF\sp500.html"

    with open(filename, "w", encoding='utf-8') as file:
        file.write(response.text)
    
    print('Got HTML File')
    
    Concat(filename)


def Concat(filename: str):
    html = pd.read_html(filename)
    #Get all data into a list and download them to seperate files due to errors happening if combined instantly
    sp500_df = html[0]
    sp500_tickers = [ticker.replace('.', '-') for ticker in sp500_df['Symbol'].tolist()]

    for ticker in sp500_tickers:
        try:
            data = yf.download(tickers=ticker, start="2015-01-01", end="2020-12-12", group_by='ticker', threads=True, auto_adjust=False)

            file_path = f'.\data\{ticker}_data.csv'
            data.to_csv(file_path)
            print(f"Data successfully saved to {file_path}")
        
        except Exception as e:
            print(f'Failed to download {ticker}')

    # Concatinate all of the Data
    data_folder = '.\data' 

    all_files = glob.glob(os.path.join(data_folder, "*.csv"))

    df_list = []
    for file_path in all_files:
        try:
            # Makes it so that pandas considers the csv a two level header
            df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
            
            ticker = df.columns[0][0]
            
            # Drops first level so that other data packets can be obtained safely
            df.columns = df.columns.droplevel(0)
            df['Ticker'] = ticker
            df_list.append(df)
            
        except Exception as e:
            print(f"Could not process file {file_path}. Error: {e}")

    # Combine all of the data in the 500 files based on date and ticker
    combined_df = pd.concat(df_list)
    combined_df = combined_df.reset_index()
    combined_df = combined_df.sort_values(by=['Ticker', 'Date'])
    combined_df.to_csv('.\Final_DF\concatenated_sp500_data.csv', index=False)

    print("Concatenated data successfully saved.")
    print(combined_df.head())

def Merge():
    concat_data = pd.read_csv(".\Final_DF\concatenated_sp500_data.csv")
    concat_data['Date'] = pd.to_datetime(concat_data['Date'])
    
    ff_data = pd.read_csv(".\Final_DF\F-F_Research_Data_5_Factors_2x3_daily.csv")
    ff_data['Inc.'] = pd.to_datetime(ff_data['Inc.'], format='%Y%m%d')
    ff_data[['Mkt-RF','SMB','HML','RMW','CMA','RF']] = ff_data[['Mkt-RF','SMB','HML','RMW','CMA','RF']]/100
    ff_data = ff_data.rename(columns={'Inc.': 'Date'})
    merge_df = pd.merge(concat_data, ff_data, on='Date', how='inner')
    merge_df.to_csv('.\Final_DF\Merged_sp500.csv', index=False)
    print('Completed Merge')
    print(merge_df.head())


def Feature_Calc():
    data_file = pd.read_csv('.\Final_DF\Merged_sp500.csv', parse_dates=['Date'])
    grouped_by = data_file.sort_values(by=['Ticker', 'Date'])
    
    data_file['Daily_Return'] = data_file.groupby('Ticker')['Adj Close'].pct_change()
    
    data_file['Momentum_12M'] = data_file.groupby('Ticker')['Adj Close'].pct_change(periods=252)
    
    data_file['Volatility_12M'] = data_file.groupby('Ticker')['Daily_Return'].rolling(window=252).std().reset_index(level=0, drop=True)
    
    rolling_window = data_file.groupby('Ticker')[['Daily_Return', 'Mkt-RF']].rolling(window=252)
    cov_matrix = rolling_window.cov()
    rolling_cov = cov_matrix.unstack()['Daily_Return']['Mkt-RF']
    rolling_var = cov_matrix.unstack()['Mkt-RF']['Mkt-RF']
    
    beta = rolling_cov / rolling_var
    data_file['Market_Beta'] = beta.reset_index(level=0, drop=True)

    
    data_file['Future'] = data_file.groupby('Ticker')['Adj Close'].shift(periods=-21)
    data_file['Forward Return'] = (data_file['Future']/data_file['Adj Close'])-1
    
    data_file['Median Return'] = data_file.groupby('Date')['Forward Return'].transform('median')
    data_file['Target'] = (data_file['Forward Return'] > data_file['Median Return']).astype(int)
    
    data_file = data_file.dropna()
    
    print('Done with Min Feature Calc')
    print(data_file.tail()) 
    
    data_file.to_csv('.\Final_DF\Featured_sp500.csv', index=False)

    
def main():
    Get_html()
    Merge()
    Feature_Calc()
    return 0

if __name__ == "__main__":
    main()