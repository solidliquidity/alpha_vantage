from os import getenv
from dotenv import load_dotenv
from alpha_vantage_api import AlphaVantageAPI

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
        
    Returns:
        str: Markdown formatted analysis results
    """
    markdown = []
    
    company_info = av.get_company_overview(ticker)
    
    # Basic company info in markdown format
    markdown.append(f"**Company:** {company_info.get('Name')} ({ticker})  ")
    markdown.append(f"**Sector:** {company_info.get('Sector')}  ")
    markdown.append(f"**Industry:** {company_info.get('Industry')}  ")
    markdown.append(f"**Market Cap:** ${float(company_info.get('MarketCapitalization', 0))/1000000000:.2f} Billion  ")
    markdown.append(f"**P/E Ratio:** {company_info.get('PERatio')}  ")
    markdown.append(f"**Dividend Yield:** {float(company_info.get('DividendYield', 0)) * 100:.2f}%  ")
    markdown.append(f"**52-Week Range:** ${company_info.get('52WeekLow')} - ${company_info.get('52WeekHigh')}  ")
    markdown.append(f"**EPS:** ${company_info.get('EPS')}  ")
    markdown.append(f"**Beta:** {company_info.get('Beta')}  ")
    
    # Combine all markdown elements and return
    markdown = "\n".join(markdown)
    return markdown