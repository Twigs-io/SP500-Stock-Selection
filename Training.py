import numpy as np
import pandas as pd
import xgboost as xgb

def calculate_performance_metrics(monthly_returns):
    if not isinstance(monthly_returns, pd.Series):
        monthly_returns = pd.Series(monthly_returns)
        
    TRADING_PERIODS_PER_YEAR = 12

    total_return = (1 + monthly_returns).prod() - 1
    annualized_return = (1 + total_return) ** (TRADING_PERIODS_PER_YEAR / len(monthly_returns)) - 1
    annualized_volatility = monthly_returns.std() * np.sqrt(TRADING_PERIODS_PER_YEAR)
    sharpe_ratio = annualized_return / annualized_volatility

    print("\n--- Strategy Performance ---")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {annualized_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

def run_backtest():
    df = pd.read_csv('.\Final_DF\Featured_sp500.csv', parse_dates=['Date'])

    features = ['Momentum_12M', 'Volatility_12M', 'Market_Beta', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    target = 'Target'
    
    backtest_start_date = '2018-01-01'
    N_Stocks = 50
    date_series = df[df['Date'] >= backtest_start_date].set_index('Date').index.to_series()
    rebalance = date_series.resample('M').last().values
    
    portfolio_returns = []
    
    for i in range(len(rebalance)):
        current = rebalance[i]
        print(f"Backtest for month: {pd.to_datetime(current).date()}")

        train_df = df[df['Date'] < current]
        test_df = df[df['Date'] == current]
        
        if test_df.empty or train_df.empty:
            continue
        
        x_train = train_df[features]
        x_test = test_df[features]
        
        y_train = train_df[target]
        
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        model.fit(x_train, y_train)

        predictions = model.predict_proba(x_test)[:, 1]
        results = pd.DataFrame({'Ticker': test_df['Ticker'], 'Prediction': predictions, 'Forward Return': test_df['Forward Return']})
        
        portfolio_df = results.sort_values(by='Prediction', ascending=False).head(N_Stocks)
        
        monthly_return = portfolio_df['Forward Return'].mean()
        portfolio_returns.append(monthly_return)
    
    
    
    calculate_performance_metrics(portfolio_returns)
    model.save_model(fname='Test_Model')
        
        
    return 0

if __name__ == "__main__":
    run_backtest()