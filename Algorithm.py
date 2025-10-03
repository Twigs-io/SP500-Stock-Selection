# region imports
from AlgorithmImports import *
import xgboost as xgb
import pandas as pd
from io import StringIO
# endregion

class UglyYellowGreenGalago(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2022, 1, 1)
        self.set_end_date(2024, 12 ,12)
        self.set_cash(100000)

        self.add_universe(self.universe.QC500)

        self.model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

        self.N_Top = 50
        self.lookback = 504

        ff_url = 'https://www.dropbox.com/scl/fi/ktcporjb448r05j3xdext/F-F_Research_Data_5_Factors_2x3_daily.csv?rlkey=1trirkws1hyzdba568wwkvd0x&st=ykfne9vc&dl=1'

        ff_file_content = self.download(ff_url)
        self.ff_data = pd.read_csv(StringIO(ff_file_content))
        self.ff_data = self.rename_and_prep_ff_data(self.ff_data)

        self.schedule.on(self.date_rules.month_end(), self.time_rules.midnight, self.Rebalance)


    def rename_and_prep_ff_data(self, ff_data):
        ff_data['Inc.'] = pd.to_datetime(ff_data['Inc.'], format='%Y%m%d')
        ff_data[['Mkt-RF','SMB','HML','RMW','CMA','RF']] = ff_data[['Mkt-RF','SMB','HML','RMW','CMA','RF']]/100
        ff_data = ff_data.rename(columns={'Inc.': 'Date'})
        self.Debug(f"Fama-French data prepared. Head:\n{ff_data.head()}")
        return ff_data
    
    def calculate_features(self, df):
        df = df.sort_values(by=['Ticker', 'Date'])
        df['Daily_Return'] = df.groupby('Ticker')['Adj Close'].pct_change()
        df['Momentum_12M'] = df.groupby('Ticker')['Adj Close'].pct_change(periods=252)
        volatility_values = df.groupby('Ticker')['Daily_Return'].rolling(window=252, min_periods=126).std()
        df['Volatility_12M'] = volatility_values.reset_index(level=0, drop=True)        

        rolling_window = df.groupby('Ticker')[['Daily_Return', 'Mkt-RF']].rolling(window=252)
        cov_matrix = rolling_window.cov()
        rolling_cov = cov_matrix.unstack()['Daily_Return']['Mkt-RF']
        rolling_var = cov_matrix.unstack()['Mkt-RF']['Mkt-RF']

        beta = rolling_cov / rolling_var
        df['Market_Beta'] = beta.reset_index(level=0, drop=True)

        df['Future Adj Close'] = df.groupby('Ticker')['Adj Close'].shift(periods=-21)
        df['Forward Return'] = (df['Future Adj Close']/df['Adj Close']) - 1   
        df['Median Return'] = df.groupby('Date')['Forward Return'].transform('median')
        df['Target'] = (df['Forward Return'] > df['Median Return']).astype(int)

        return df


    def Rebalance(self):
        if self.ff_data is None:
            self.Debug("Fama-French data not available.")
            return

        history = self.History(self.active_securities.keys, self.lookback, Resolution.DAILY)
        df = history['close'].unstack(level=0).rename_axis('Date').reset_index()

        df = df.melt(id_vars='Date', var_name='Ticker', value_name='Adj Close')

        df['Date'] = df['Date'].dt.normalize()
        self.debug(df.head())
        merge = pd.merge(df, self.ff_data, on='Date', how='inner')
        merge = merge.dropna(subset=['Adj Close'])
        data_file = self.calculate_features(merge)
        data_file = data_file.dropna()

        features = ['Momentum_12M', 'Volatility_12M','Market_Beta',
                    'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        
        x_train = data_file[features]
        y_train = data_file['Target']

        self.model.fit(x_train, y_train)

        latest_features = data_file.groupby('Ticker').last().reset_index()

        if latest_features.empty:
            self.Debug('Latest features was empty')
            return

        X_live = latest_features[features]

        predictions = self.model.predict(X_live)

        live_predictions = self.model.predict_proba(X_live)[:, 1]

        latest_features['Prediction'] = live_predictions

        top_stocks = latest_features.sort_values(by='Prediction', ascending=False).head(self.N_Top)
        self.stocks_to_hold = [self.Symbol(ticker) for ticker in top_stocks['Ticker']]

        for security in self.Portfolio.Values:
            if security.Invested and security.Symbol not in self.stocks_to_hold:
                self.Liquidate(security.Symbol, "No longer a top pick")

        weight = 1.0 / len(self.stocks_to_hold)
        for symbol in self.stocks_to_hold:
            self.SetHoldings(symbol, weight)
        return
