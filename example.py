import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

# Import our Alpha Vantage API wrapper
from alpha_vantage_api import AlphaVantageAPI
from os import getenv
from dotenv import load_dotenv

load_dotenv()

# Set up the API with your key
api_key = load_dotenv('ALPHA_VANTAGE_API_KEY')  # Replace with your actual API key
av = AlphaVantageAPI(api_key)

def analyze_stock(ticker, lookback_days=365):
    """
    Comprehensive stock analysis using Alpha Vantage API.
    
    Args:
        ticker (str): Stock ticker symbol
        lookback_days (int): Number of days to analyze
    """
    print(f"\n{'='*50}")
    print(f"COMPREHENSIVE ANALYSIS FOR {ticker}")
    print(f"{'='*50}\n")
    
    # 1. Get company overview
    print("GETTING COMPANY OVERVIEW...")
    company_info = av.get_company_overview(ticker)
    
    # Print basic company info
    print(f"Company: {company_info.get('Name')} ({ticker})")
    print(f"Sector: {company_info.get('Sector')}")
    print(f"Industry: {company_info.get('Industry')}")
    print(f"Market Cap: ${float(company_info.get('MarketCapitalization', 0))/1000000000:.2f} Billion")
    print(f"P/E Ratio: {company_info.get('PERatio')}")
    print(f"Dividend Yield: {float(company_info.get('DividendYield', 0)) * 100:.2f}%")
    print(f"52-Week Range: ${company_info.get('52WeekLow')} - ${company_info.get('52WeekHigh')}")
    print(f"EPS: ${company_info.get('EPS')}")
    print(f"Beta: {company_info.get('Beta')}")
    '''
    # 2. Get daily stock data
    print("\nGETTING PRICE DATA...")
    daily_data = av.get_daily(ticker, adjusted=True, outputsize="full", convert_to_df=True)
    
    # Filter to the lookback period
    start_date = datetime.now() - timedelta(days=lookback_days)
    filtered_data = daily_data[daily_data.index >= start_date.strftime('%Y-%m-%d')]
    
    # Clean column names (remove digit prefixes)
    filtered_data.columns = [col.split('. ')[1] if '. ' in col else col for col in filtered_data.columns]
    
    # Calculate daily returns
    filtered_data['daily_return'] = filtered_data['adjusted close'].pct_change() * 100
    
    # Print price summary
    current_price = filtered_data['adjusted close'].iloc[-1]
    prev_close = filtered_data['adjusted close'].iloc[-2]
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"Daily Change: ${price_change:.2f} ({price_change_pct:.2f}%)")
    print(f"YTD Return: {(filtered_data['adjusted close'].iloc[-1] / filtered_data['adjusted close'].iloc[0] - 1) * 100:.2f}%")
    print(f"Average Daily Volume (30d): {filtered_data['volume'].tail(30).mean():.0f}")
    
    # 3. Get technical indicators
    print("\nCALCULATING TECHNICAL INDICATORS...")
    
    # SMA (20-day and 50-day)
    filtered_data['SMA_20'] = av.get_sma(ticker, time_period=20, convert_to_df=True)['SMA'].reindex(filtered_data.index)
    filtered_data['SMA_50'] = av.get_sma(ticker, time_period=50, convert_to_df=True)['SMA'].reindex(filtered_data.index)
    
    # RSI (14-day)
    filtered_data['RSI'] = av.get_rsi(ticker, time_period=14, convert_to_df=True)['RSI'].reindex(filtered_data.index)
    
    # MACD
    macd_data = av.get_macd(ticker, convert_to_df=True).reindex(filtered_data.index)
    filtered_data['MACD'] = macd_data['MACD']
    filtered_data['MACD_Signal'] = macd_data['MACD_Signal']
    filtered_data['MACD_Hist'] = macd_data['MACD_Hist']
    
    # Print technical indicator summary
    print(f"Current RSI (14-day): {filtered_data['RSI'].iloc[-1]:.2f}")
    print(f"Current MACD: {filtered_data['MACD'].iloc[-1]:.2f}")
    print(f"MACD Signal Line: {filtered_data['MACD_Signal'].iloc[-1]:.2f}")
    print(f"MACD Histogram: {filtered_data['MACD_Hist'].iloc[-1]:.2f}")
    
    # 4. Get earnings data
    print("\nGETTING EARNINGS DATA...")
    earnings_data = av.get_earnings(ticker)
    
    # Extract quarterly earnings
    quarterly_earnings = pd.DataFrame(earnings_data.get('quarterlyEarnings', []))
    if not quarterly_earnings.empty:
        quarterly_earnings['reportedEPS'] = quarterly_earnings['reportedEPS'].astype(float)
        quarterly_earnings['estimatedEPS'] = quarterly_earnings['estimatedEPS'].astype(float)
        quarterly_earnings['surprise'] = quarterly_earnings['surprise'].astype(float)
        quarterly_earnings['surprisePercentage'] = quarterly_earnings['surprisePercentage'].astype(float)
        
        # Print recent earnings
        print("Recent Quarterly Earnings:")
        for idx, row in quarterly_earnings.head(4).iterrows():
            eps_diff = row['reportedEPS'] - row['estimatedEPS']
            print(f"  {row['fiscalDateEnding']}: Reported ${row['reportedEPS']:.2f} vs Est. ${row['estimatedEPS']:.2f} " +
                 f"({'+' if eps_diff >= 0 else ''}{eps_diff:.2f}, {row['surprisePercentage']:.2f}%)")
    
    # 5. Get income statement data
    print("\nGETTING FINANCIAL DATA...")
    income_data = av.get_income_statement(ticker)
    
    # Extract annual income statements
    annual_income = pd.DataFrame(income_data.get('annualReports', []))
    if not annual_income.empty:
        # Convert numeric columns
        for col in ['totalRevenue', 'grossProfit', 'netIncome']:
            if col in annual_income.columns:
                annual_income[col] = annual_income[col].astype(float)
        
        # Print financial summary
        latest_year = annual_income.iloc[0]
        prev_year = annual_income.iloc[1] if len(annual_income) > 1 else None
        
        revenue = float(latest_year.get('totalRevenue', 0))
        profit = float(latest_year.get('netIncome', 0))
        
        print(f"Latest Annual Revenue (FY {latest_year.get('fiscalDateEnding', '')[:4]}): ${revenue/1000000000:.2f} Billion")
        print(f"Net Income: ${profit/1000000000:.2f} Billion")
        print(f"Profit Margin: {(profit/revenue)*100:.2f}%")
        
        if prev_year is not None:
            prev_revenue = float(prev_year.get('totalRevenue', 0))
            rev_growth = ((revenue - prev_revenue) / prev_revenue) * 100
            print(f"YoY Revenue Growth: {rev_growth:.2f}%")
    
    # 6. Get news sentiment
    print("\nGETTING NEWS SENTIMENT...")
    news_data = av.get_news_sentiment(ticker, limit=10)
    
    if 'feed' in news_data:
        feed_items = news_data['feed']
        sentiment_scores = [float(item.get('overall_sentiment_score', 0)) for item in feed_items if 'overall_sentiment_score' in item]
        
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            print(f"Average News Sentiment (scale -1 to 1): {avg_sentiment:.3f}")
            
            sentiment_label = "Neutral"
            if avg_sentiment > 0.35:
                sentiment_label = "Very Positive"
            elif avg_sentiment > 0.15:
                sentiment_label = "Positive"
            elif avg_sentiment < -0.35:
                sentiment_label = "Very Negative"
            elif avg_sentiment < -0.15:
                sentiment_label = "Negative"
            
            print(f"Sentiment Classification: {sentiment_label}")
            
            # Print recent news headlines
            print("\nRecent Headlines:")
            for i, item in enumerate(feed_items[:5], 1):
                print(f"  {i}. {item.get('title', 'No title')} ({item.get('source', 'Unknown source')})")
    
    # 7. Visualize the data
    print("\nCREATING VISUALIZATIONS...")
    
    # Set up the plot style
    sns.set(style="darkgrid")
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Stock Price with Moving Averages
    plt.subplot(3, 1, 1)
    plt.plot(filtered_data.index, filtered_data['adjusted close'], label='Price', color='blue')
    plt.plot(filtered_data.index, filtered_data['SMA_20'], label='20-day SMA', color='orange', linestyle='--')
    plt.plot(filtered_data.index, filtered_data['SMA_50'], label='50-day SMA', color='green', linestyle='--')
    plt.title(f'{ticker} Stock Price with Moving Averages')
    plt.ylabel('Price ($)')
    plt.legend()
    
    # Plot 2: Volume
    plt.subplot(3, 1, 2)
    plt.bar(filtered_data.index, filtered_data['volume'], color='purple', alpha=0.6)
    plt.title(f'{ticker} Trading Volume')
    plt.ylabel('Volume')
    
    # Plot 3: RSI
    plt.subplot(3, 1, 3)
    plt.plot(filtered_data.index, filtered_data['RSI'], color='red')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
    plt.title(f'{ticker} RSI (14-day)')
    plt.ylabel('RSI')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f"{ticker}_analysis.png")
    plt.close()
    
    print(f"\nAnalysis complete! Visualization saved as {ticker}_analysis.png")
    
    return filtered_data

'''
# Run the analysis for a few stocks
if __name__ == "__main__":
    # Analyze some popular stocks
    stocks = ["AAPL"]
    
    for stock in stocks:
        try:
            stock_data = analyze_stock(stock, lookback_days=180)
            print("\n" + "-"*70 + "\n")
        except Exception as e:
            print(f"Error analyzing {stock}: {str(e)}")
            continue