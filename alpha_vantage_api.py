import requests
import pandas as pd
from typing import Dict, Any, Optional, Union, List


class AlphaVantageAPI:
    """
    A comprehensive wrapper for the Alpha Vantage API.
    Documentation: https://www.alphavantage.co/documentation/
    """
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str):
        """
        Initialize the AlphaVantage API client.
        
        Args:
            api_key (str): Your Alpha Vantage API key
        """
        self.api_key = api_key
    
    def _fetch(self, function: str, **kwargs) -> Dict[str, Any]:
        """
        Helper method to fetch data from Alpha Vantage API.
        
        Args:
            function (str): The Alpha Vantage API function to call
            **kwargs: Additional parameters to include in the request
            
        Returns:
            Dict[str, Any]: The JSON response from the API
        """
        params = {"function": function, "apikey": self.api_key}
        params.update(kwargs)  # Add additional parameters if provided
        
        response = requests.get(self.BASE_URL, params=params)
        
        # Check for errors
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}")
        
        data = response.json()
        
        # Check for API error messages
        if "Error Message" in data:
            raise Exception(f"API Error: {data['Error Message']}")
        if "Information" in data and "please consider optimizing your API call frequency" in data["Information"]:
            print("Warning: API call frequency limit reached.")
            
        return data
    
    def _optional_dataframe(self, data: Dict[str, Any], convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Optionally convert the response to a pandas DataFrame.
        
        Args:
            data (Dict[str, Any]): The API response data
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The data in the requested format
        """
        if not convert_to_df:
            return data
            
        # Find the time series data key (varies by endpoint)
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key or "Technical Analysis" in key or "Weekly" in key or "Monthly" in key:
                time_series_key = key
                break
                
        if time_series_key:
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            return df
        else:
            # For non-time series data, just return the original response
            return data
    
    #######################
    ## STOCK TIME SERIES ##
    #######################
    
    def get_intraday(self, symbol: str, interval: str = "5min", adjusted: bool = True, 
                    outputsize: str = "compact", datatype: str = "json", convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch intraday time series stock data.
        
        Args:
            symbol (str): The stock ticker symbol
            interval (str): Time interval between data points (1min, 5min, 15min, 30min, 60min)
            adjusted (bool): Whether to return adjusted data
            outputsize (str): 'compact' (latest 100 data points) or 'full' (up to 20 years of data)
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The intraday time series data
        """
        function = "TIME_SERIES_INTRADAY"
        data = self._fetch(function, symbol=symbol, interval=interval, adjusted=str(adjusted).lower(),
                         outputsize=outputsize, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_daily(self, symbol: str, adjusted: bool = True, outputsize: str = "compact", 
                 datatype: str = "json", convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch daily time series stock data.
        
        Args:
            symbol (str): The stock ticker symbol
            adjusted (bool): Whether to return adjusted data
            outputsize (str): 'compact' (latest 100 data points) or 'full' (up to 20 years of data)
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The daily time series data
        """
        function = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
        data = self._fetch(function, symbol=symbol, outputsize=outputsize, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_weekly(self, symbol: str, adjusted: bool = True, datatype: str = "json", 
                  convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch weekly time series stock data.
        
        Args:
            symbol (str): The stock ticker symbol
            adjusted (bool): Whether to return adjusted data
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The weekly time series data
        """
        function = "TIME_SERIES_WEEKLY_ADJUSTED" if adjusted else "TIME_SERIES_WEEKLY"
        data = self._fetch(function, symbol=symbol, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_monthly(self, symbol: str, adjusted: bool = True, datatype: str = "json", 
                   convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch monthly time series stock data.
        
        Args:
            symbol (str): The stock ticker symbol
            adjusted (bool): Whether to return adjusted data
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The monthly time series data
        """
        function = "TIME_SERIES_MONTHLY_ADJUSTED" if adjusted else "TIME_SERIES_MONTHLY"
        data = self._fetch(function, symbol=symbol, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch global quote for a symbol.
        
        Args:
            symbol (str): The stock ticker symbol
            
        Returns:
            Dict[str, Any]: The global quote data
        """
        return self._fetch("GLOBAL_QUOTE", symbol=symbol)
    
    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch batch stock quotes for multiple symbols.
        
        Args:
            symbols (List[str]): List of stock ticker symbols
            
        Returns:
            Dict[str, Any]: The batch quote data
        """
        symbols_str = ",".join(symbols)
        return self._fetch("BATCH_STOCK_QUOTES", symbols=symbols_str)
    
    ####################
    ## FUNDAMENTALS ##
    ####################
    
    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch company overview, including sector, market cap, P/E ratio, etc.
        
        Args:
            symbol (str): The stock ticker symbol
            
        Returns:
            Dict[str, Any]: The company overview data
        """
        return self._fetch("OVERVIEW", symbol=symbol)
    
    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch earnings history (annual & quarterly).
        
        Args:
            symbol (str): The stock ticker symbol
            
        Returns:
            Dict[str, Any]: The earnings data
        """
        return self._fetch("EARNINGS", symbol=symbol)
    
    def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch income statement data.
        
        Args:
            symbol (str): The stock ticker symbol
            
        Returns:
            Dict[str, Any]: The income statement data
        """
        return self._fetch("INCOME_STATEMENT", symbol=symbol)
    
    def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch balance sheet data.
        
        Args:
            symbol (str): The stock ticker symbol
            
        Returns:
            Dict[str, Any]: The balance sheet data
        """
        return self._fetch("BALANCE_SHEET", symbol=symbol)
    
    def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch cash flow statement data.
        
        Args:
            symbol (str): The stock ticker symbol
            
        Returns:
            Dict[str, Any]: The cash flow data
        """
        return self._fetch("CASH_FLOW", symbol=symbol)

    def get_earnings_calendar(self, symbol: Optional[str] = None, horizon: str = "3month") -> Dict[str, Any]:
        """
        Fetch earnings calendar.
        
        Args:
            symbol (Optional[str]): The stock ticker symbol (optional)
            horizon (str): Time horizon (3month, 6month, 12month)
            
        Returns:
            Dict[str, Any]: The earnings calendar data
        """
        params = {"horizon": horizon}
        if symbol:
            params["symbol"] = symbol
        return self._fetch("EARNINGS_CALENDAR", **params)
    
    def get_ipo_calendar(self) -> Dict[str, Any]:
        """
        Fetch IPO calendar.
        
        Returns:
            Dict[str, Any]: The IPO calendar data
        """
        return self._fetch("IPO_CALENDAR")
    
    #########################
    ## TECHNICAL INDICATORS ##
    #########################
    
    def get_sma(self, symbol: str, interval: str = "daily", time_period: int = 50, 
               series_type: str = "close", convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch Simple Moving Average (SMA).
        
        Args:
            symbol (str): The stock ticker symbol
            interval (str): Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            time_period (int): Number of data points to calculate the SMA
            series_type (str): Price type (close, open, high, low)
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The SMA data
        """
        data = self._fetch("SMA", symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_ema(self, symbol: str, interval: str = "daily", time_period: int = 50, 
               series_type: str = "close", convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch Exponential Moving Average (EMA).
        
        Args:
            symbol (str): The stock ticker symbol
            interval (str): Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            time_period (int): Number of data points to calculate the EMA
            series_type (str): Price type (close, open, high, low)
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The EMA data
        """
        data = self._fetch("EMA", symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_macd(self, symbol: str, interval: str = "daily", series_type: str = "close", 
                fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9,
                convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch Moving Average Convergence/Divergence (MACD).
        
        Args:
            symbol (str): The stock ticker symbol
            interval (str): Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            series_type (str): Price type (close, open, high, low)
            fastperiod (int): Fast period
            slowperiod (int): Slow period
            signalperiod (int): Signal period
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The MACD data
        """
        data = self._fetch("MACD", symbol=symbol, interval=interval, series_type=series_type,
                         fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_rsi(self, symbol: str, interval: str = "daily", time_period: int = 14, 
               series_type: str = "close", convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch Relative Strength Index (RSI).
        
        Args:
            symbol (str): The stock ticker symbol
            interval (str): Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            time_period (int): Number of data points to calculate the RSI
            series_type (str): Price type (close, open, high, low)
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The RSI data
        """
        data = self._fetch("RSI", symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_bbands(self, symbol: str, interval: str = "daily", time_period: int = 20, 
                  series_type: str = "close", nbdevup: int = 2, nbdevdn: int = 2, matype: int = 0,
                  convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch Bollinger Bands.
        
        Args:
            symbol (str): The stock ticker symbol
            interval (str): Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            time_period (int): Number of data points to calculate the Bollinger Bands
            series_type (str): Price type (close, open, high, low)
            nbdevup (int): Standard deviation multiplier for upper band
            nbdevdn (int): Standard deviation multiplier for lower band
            matype (int): Moving average type (0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3)
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The Bollinger Bands data
        """
        data = self._fetch("BBANDS", symbol=symbol, interval=interval, time_period=time_period, 
                         series_type=series_type, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_stoch(self, symbol: str, interval: str = "daily", fastkperiod: int = 5, 
                 slowkperiod: int = 3, slowdperiod: int = 3, slowkmatype: int = 0, slowdmatype: int = 0,
                 convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch Stochastic Oscillator.
        
        Args:
            symbol (str): The stock ticker symbol
            interval (str): Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            fastkperiod (int): Fast K period
            slowkperiod (int): Slow K period
            slowdperiod (int): Slow D period
            slowkmatype (int): Moving average type for slow K
            slowdmatype (int): Moving average type for slow D
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The Stochastic Oscillator data
        """
        data = self._fetch("STOCH", symbol=symbol, interval=interval, fastkperiod=fastkperiod,
                         slowkperiod=slowkperiod, slowdperiod=slowdperiod, 
                         slowkmatype=slowkmatype, slowdmatype=slowdmatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_adx(self, symbol: str, interval: str = "daily", time_period: int = 14,
               convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch Average Directional Movement Index (ADX).
        
        Args:
            symbol (str): The stock ticker symbol
            interval (str): Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            time_period (int): Number of data points to calculate the ADX
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The ADX data
        """
        data = self._fetch("ADX", symbol=symbol, interval=interval, time_period=time_period)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_vwap(self, symbol: str, interval: str = "15min", convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch Volume Weighted Average Price (VWAP).
        
        Args:
            symbol (str): The stock ticker symbol
            interval (str): Time interval (1min, 5min, 15min, 30min, 60min)
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The VWAP data
        """
        data = self._fetch("VWAP", symbol=symbol, interval=interval)
        return self._optional_dataframe(data, convert_to_df)
    
    ####################
    ## FOREX & CRYPTO ##
    ####################
    
    def get_forex_intraday(self, from_symbol: str, to_symbol: str, interval: str = "5min", 
                          outputsize: str = "compact", datatype: str = "json", 
                          convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch intraday forex data.
        
        Args:
            from_symbol (str): The from currency symbol
            to_symbol (str): The to currency symbol
            interval (str): Time interval (1min, 5min, 15min, 30min, 60min)
            outputsize (str): 'compact' (latest 100 data points) or 'full' (up to 20 years of data)
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The intraday forex data
        """
        data = self._fetch("FX_INTRADAY", from_symbol=from_symbol, to_symbol=to_symbol, 
                         interval=interval, outputsize=outputsize, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_forex_daily(self, from_symbol: str, to_symbol: str, outputsize: str = "compact", 
                       datatype: str = "json", convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch daily forex data.
        
        Args:
            from_symbol (str): The from currency symbol
            to_symbol (str): The to currency symbol
            outputsize (str): 'compact' (latest 100 data points) or 'full' (up to 20 years of data)
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The daily forex data
        """
        data = self._fetch("FX_DAILY", from_symbol=from_symbol, to_symbol=to_symbol, 
                         outputsize=outputsize, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_forex_weekly(self, from_symbol: str, to_symbol: str, datatype: str = "json", 
                        convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch weekly forex data.
        
        Args:
            from_symbol (str): The from currency symbol
            to_symbol (str): The to currency symbol
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The weekly forex data
        """
        data = self._fetch("FX_WEEKLY", from_symbol=from_symbol, to_symbol=to_symbol, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_forex_monthly(self, from_symbol: str, to_symbol: str, datatype: str = "json", 
                         convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch monthly forex data.
        
        Args:
            from_symbol (str): The from currency symbol
            to_symbol (str): The to currency symbol
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The monthly forex data
        """
        data = self._fetch("FX_MONTHLY", from_symbol=from_symbol, to_symbol=to_symbol, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_currency_exchange_rate(self, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """
        Fetch currency exchange rate.
        
        Args:
            from_currency (str): The from currency symbol
            to_currency (str): The to currency symbol
            
        Returns:
            Dict[str, Any]: The currency exchange rate data
        """
        return self._fetch("CURRENCY_EXCHANGE_RATE", from_currency=from_currency, to_currency=to_currency)
    
    def get_digital_currency_daily(self, symbol: str, market: str = "USD", 
                                  convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch daily digital currency data.
        
        Args:
            symbol (str): The digital currency symbol
            market (str): The market to convert to (USD, EUR, CNY, etc.)
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The daily digital currency data
        """
        data = self._fetch("DIGITAL_CURRENCY_DAILY", symbol=symbol, market=market)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_digital_currency_weekly(self, symbol: str, market: str = "USD", 
                                   convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch weekly digital currency data.
        
        Args:
            symbol (str): The digital currency symbol
            market (str): The market to convert to (USD, EUR, CNY, etc.)
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The weekly digital currency data
        """
        data = self._fetch("DIGITAL_CURRENCY_WEEKLY", symbol=symbol, market=market)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_digital_currency_monthly(self, symbol: str, market: str = "USD", 
                                    convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch monthly digital currency data.
        
        Args:
            symbol (str): The digital currency symbol
            market (str): The market to convert to (USD, EUR, CNY, etc.)
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The monthly digital currency data
        """
        data = self._fetch("DIGITAL_CURRENCY_MONTHLY", symbol=symbol, market=market)
        return self._optional_dataframe(data, convert_to_df)
    
    ########################
    ## ECONOMIC INDICATORS ##
    ########################
    
    def get_real_gdp(self, interval: str = "quarterly", datatype: str = "json", 
                    convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch US Real GDP.
        
        Args:
            interval (str): 'annual' or 'quarterly'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The real GDP data
        """
        data = self._fetch("REAL_GDP", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_real_gdp_per_capita(self, datatype: str = "json", 
                               convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch US Real GDP per capita.
        
        Args:
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The real GDP per capita data
        """
        data = self._fetch("REAL_GDP_PER_CAPITA", datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_treasury_yield(self, interval: str = "daily", maturity: str = "10year", 
                          datatype: str = "json", convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch US Treasury yield.
        
        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
            maturity (str): '3month', '5year', '10year', or '30year'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The Treasury yield data
        """
        data = self._fetch("TREASURY_YIELD", interval=interval, maturity=maturity, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_federal_funds_rate(self, interval: str = "daily", datatype: str = "json", 
                              convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch US Federal Funds Rate.
        
        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The Federal Funds Rate data
        """
        data = self._fetch("FEDERAL_FUNDS_RATE", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_cpi(self, interval: str = "monthly", datatype: str = "json", 
               convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch US Consumer Price Index (CPI).
        
        Args:
            interval (str): 'monthly' or 'semiannual'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The CPI data
        """
        data = self._fetch("CPI", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_inflation(self, datatype: str = "json", 
                     convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch US Inflation data.
        
        Args:
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The inflation data
        """
        data = self._fetch("INFLATION", datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_retail_sales(self, datatype: str = "json", 
                        convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch US Retail Sales data.
        
        Args:
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The retail sales data
        """
        data = self._fetch("RETAIL_SALES", datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_unemployment(self, datatype: str = "json", 
                        convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch US Unemployment data.
        
        Args:
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The unemployment data
        """
        data = self._fetch("UNEMPLOYMENT", datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_nonfarm_payroll(self, datatype: str = "json", 
                           convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch US Nonfarm Payroll data.
        
        Args:
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The nonfarm payroll data
        """
        data = self._fetch("NONFARM_PAYROLL", datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    ####################
    ## SECTOR PERFORMANCE ##
    ####################
    
    def get_sector_performance(self) -> Dict[str, Any]:
        """
        Fetch US sector performance data.
        
        Returns:
            Dict[str, Any]: The sector performance data
        """
        return self._fetch("SECTOR")
    
    ####################
    ## SEARCH & LISTINGS ##
    ####################
    
    def symbol_search(self, keywords: str) -> Dict[str, Any]:
        """
        Search for symbols based on keywords.
        
        Args:
            keywords (str): Keywords to search for
            
        Returns:
            Dict[str, Any]: The search results
        """
        return self._fetch("SYMBOL_SEARCH", keywords=keywords)
    
    def get_listing_status(self, date: Optional[str] = None, state: str = "active", 
                          datatype: str = "json") -> Dict[str, Any]:
        """
        Fetch listing status for symbols.
        
        Args:
            date (Optional[str]): Date in yyyy-MM-dd format (default: latest)
            state (str): 'active' or 'delisted'
            datatype (str): 'json' or 'csv'
            
        Returns:
            Dict[str, Any]: The listing status data
        """
        params = {"state": state, "datatype": datatype}
        if date:
            params["date"] = date
        return self._fetch("LISTING_STATUS", **params)
    
    ####################
    ## NEWS & SENTIMENT ##
    ####################
    
    def get_news_sentiment(self, tickers: Optional[str] = None, topics: Optional[str] = None, 
                          time_from: Optional[str] = None, time_to: Optional[str] = None, 
                          sort: str = "LATEST", limit: int = 50) -> Dict[str, Any]:
        """
        Fetch news and sentiment data.
        
        Args:
            tickers (Optional[str]): Comma-separated list of ticker symbols
            topics (Optional[str]): Comma-separated list of topics
            time_from (Optional[str]): Starting time (YYYYMMDDTHHMM format)
            time_to (Optional[str]): Ending time (YYYYMMDDTHHMM format)
            sort (str): 'LATEST', 'EARLIEST', or 'RELEVANCE'
            limit (int): Number of results to return (1-1000)
            
        Returns:
            Dict[str, Any]: The news and sentiment data
        """
        params = {"sort": sort, "limit": limit}
        if tickers:
            params["tickers"] = tickers
        if topics:
            params["topics"] = topics
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to
        return self._fetch("NEWS_SENTIMENT", **params)
    
    def get_top_gainers_losers(self) -> Dict[str, Any]:
        """
        Fetch top gainers and losers in the US market.
        
        Returns:
            Dict[str, Any]: The top gainers and losers data
        """
        return self._fetch("TOP_GAINERS_LOSERS")
    
    ####################
    ## COMMODITIES ##
    ####################
    
    def get_wti(self, interval: str = "daily", datatype: str = "json", 
               convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch WTI crude oil prices.
        
        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The WTI price data
        """
        data = self._fetch("WTI", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_brent(self, interval: str = "daily", datatype: str = "json", 
                convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch Brent crude oil prices.
        
        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The Brent price data
        """
        data = self._fetch("BRENT", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_natural_gas(self, interval: str = "daily", datatype: str = "json", 
                       convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch natural gas prices.
        
        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The natural gas price data
        """
        data = self._fetch("NATURAL_GAS", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_copper(self, interval: str = "daily", datatype: str = "json", 
                 convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch copper prices.
        
        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The copper price data
        """
        data = self._fetch("COPPER", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_aluminum(self, interval: str = "daily", datatype: str = "json", 
                    convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch aluminum prices.
        
        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The aluminum price data
        """
        data = self._fetch("ALUMINUM", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_wheat(self, interval: str = "daily", datatype: str = "json", 
                convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch wheat prices.
        
        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The wheat price data
        """
        data = self._fetch("WHEAT", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_corn(self, interval: str = "daily", datatype: str = "json", 
               convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch corn prices.
        
        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The corn price data
        """
        data = self._fetch("CORN", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_cotton(self, interval: str = "daily", datatype: str = "json", 
                 convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch cotton prices.
        
        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The cotton price data
        """
        data = self._fetch("COTTON", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_sugar(self, interval: str = "daily", datatype: str = "json", 
                convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch sugar prices.
        
        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The sugar price data
        """
        data = self._fetch("SUGAR", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    def get_coffee(self, interval: str = "daily", datatype: str = "json", 
                 convert_to_df: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Fetch coffee prices.
        
        Args:
            interval (str): 'daily', 'weekly', or 'monthly'
            datatype (str): 'json' or 'csv'
            convert_to_df (bool): Whether to convert the response to a DataFrame
            
        Returns:
            Union[Dict[str, Any], pd.DataFrame]: The coffee price data
        """
        data = self._fetch("COFFEE", interval=interval, datatype=datatype)
        return self._optional_dataframe(data, convert_to_df)
    
    ####################
    ## USAGE EXAMPLE ##
    ####################


# Usage Example
if __name__ == "__main__":
    api_key = "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with your API key
    av = AlphaVantageAPI(api_key)
    
    # Example 1: Get company overview for Apple
    ticker = "AAPL"
    company_info = av.get_company_overview(ticker)
    print(f"Company Name: {company_info.get('Name')}")
    print(f"Sector: {company_info.get('Sector')}")
    print(f"Market Cap: {company_info.get('MarketCapitalization')}")
    print(f"P/E Ratio: {company_info.get('PERatio')}")
    print(f"Dividend Yield: {company_info.get('DividendYield')}")
    print(f"52 Week High: {company_info.get('52WeekHigh')}")
    print(f"52 Week Low: {company_info.get('52WeekLow')}")
    
    # Example 2: Get daily stock data for Apple and convert to DataFrame
    daily_data = av.get_daily(ticker, convert_to_df=True)
    print("\nLatest Daily Data:")
    print(daily_data.head())
    
    # Example 3: Get technical indicators
    sma_data = av.get_sma(ticker, time_period=20, convert_to_df=True)
    print("\nSMA Data:")
    print(sma_data.head())
    
    # Example 4: Get sector performance
    sector_data = av.get_sector_performance()
    print("\nSector Performance:")
    for sector, performance in sector_data.get('Rank A: Real-Time Performance', {}).items():
        print(f"{sector}: {performance}")
    
    # Example 5: Get forex data
    forex_data = av.get_forex_daily("EUR", "USD", convert_to_df=True)
    print("\nEUR/USD Daily Data:")
    print(forex_data.head())
    
    # Example 6: Get economic indicator
    gdp_data = av.get_real_gdp(convert_to_df=True)
    print("\nUS Real GDP Data:")
    print(gdp_data.head())
