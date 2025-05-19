import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class MarketDataCollector:
    """Collects market data from various sources"""
    
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com"
    
    def get_stock_data(self, ticker, period="1y", interval="1d"):
        """Get historical stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if df.empty:
                print(f"No data found for {ticker}")
                return None
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def get_market_indices(self, indices=["^GSPC", "^DJI", "^IXIC"]):
        """Get major market indices data"""
        indices_data = {}
        for index in indices:
            indices_data[index] = self.get_stock_data(index)
        return indices_data
    
    def get_sector_performance(self):
        """Get sector performance data using sector ETFs"""
        sectors = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financials": "XLF",
            "Consumer Discretionary": "XLY",
            "Industrials": "XLI",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Materials": "XLB",
            "Real Estate": "XLRE",
            "Consumer Staples": "XLP",
            "Communication Services": "XLC"
        }
        sector_data = {}
        for sector, ticker in sectors.items():
            sector_data[sector] = self.get_stock_data(ticker)
        return sector_data
    
    def get_company_fundamentals(self, ticker):
        """Get company fundamental data"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fundamentals = {
                "marketCap": info.get("marketCap", None),
                "trailingPE": info.get("trailingPE", None),
                "forwardPE": info.get("forwardPE", None),
                "dividendYield": info.get("dividendYield", None),
                "returnOnEquity": info.get("returnOnEquity", None),
                "debtToEquity": info.get("debtToEquity", None),
                "beta": info.get("beta", None),
                "profitMargins": info.get("profitMargins", None),
                "revenueGrowth": info.get("revenueGrowth", None),
                "earningsGrowth": info.get("earningsGrowth", None),
                "sector": info.get("sector", None),
                "industry": info.get("industry", None),
                "targetMeanPrice": info.get("targetMeanPrice", None),
                "recommendationMean": info.get("recommendationMean", None),
                "shortPercentOfFloat": info.get("shortPercentOfFloat", None)
            }
            return fundamentals
        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
            return None
    
    def screen_stocks(self, criteria):
        """
        Screen stocks based on criteria
        criteria: Dict with screening parameters
        """
        # This would be more sophisticated in production
        # For now, we'll use a simplified approach with predefined tickers
        example_tickers = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
            "TSLA", "NVDA", "JPM", "V", "WMT", 
            "JNJ", "PG", "UNH", "HD", "BAC",
            "MA", "XOM", "DIS", "NFLX", "ADBE"
        ]
        
        results = []
        for ticker in example_tickers:
            fundamentals = self.get_company_fundamentals(ticker)
            if fundamentals:
                stock_data = self.get_stock_data(ticker, period="3mo")
                if stock_data is not None:
                    # Calculate current metrics
                    current_price = stock_data['Close'].iloc[-1]
                    price_30d_ago = stock_data['Close'].iloc[-21] if len(stock_data) >= 21 else stock_data['Close'].iloc[0]
                    performance_30d = (current_price / price_30d_ago - 1) * 100
                    
                    # Check criteria
                    matches_criteria = True
                    if criteria.get('min_marketCap') and (fundamentals.get('marketCap', 0) or 0) < criteria['min_marketCap']:
                        matches_criteria = False
                    if criteria.get('max_PE') and (fundamentals.get('trailingPE', 999) or 999) > criteria['max_PE']:
                        matches_criteria = False
                    if criteria.get('min_30d_performance') and performance_30d < criteria['min_30d_performance']:
                        matches_criteria = False
                    
                    if matches_criteria:
                        results.append({
                            'ticker': ticker,
                            'price': current_price,
                            '30d_performance': performance_30d,
                            'marketCap': fundamentals.get('marketCap', None),
                            'PE': fundamentals.get('trailingPE', None),
                            'sector': fundamentals.get('sector', None)
                        })
        
        return pd.DataFrame(results)


class NewsCollector:
    """Collects news and sentiment data"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def get_latest_news(self, ticker, num_articles=5):
        """
        Get latest news for a ticker
        In a real implementation, this would connect to a news API
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            processed_news = []
            
            for i, article in enumerate(news[:num_articles]):
                title = article.get('title', '')
                sentiment = self.analyze_sentiment(title)
                processed_news.append({
                    'title': title,
                    'link': article.get('link', ''),
                    'publisher': article.get('publisher', ''),
                    'published': datetime.fromtimestamp(article.get('providerPublishTime', 0)),
                    'sentiment': sentiment
                })
            
            return processed_news
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using VADER"""
        sentiment = self.sia.polarity_scores(text)
        return sentiment
    
    def get_market_sentiment(self, tickers):
        """Get overall market sentiment based on news for multiple tickers"""
        all_sentiments = []
        for ticker in tickers:
            news = self.get_latest_news(ticker)
            for article in news:
                all_sentiments.append(article['sentiment']['compound'])
        
        if all_sentiments:
            avg_sentiment = sum(all_sentiments) / len(all_sentiments)
            if avg_sentiment >= 0.05:
                sentiment_label = "Positive"
            elif avg_sentiment <= -0.05:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
                
            return {
                'avg_score': avg_sentiment,
                'sentiment': sentiment_label,
                'num_articles': len(all_sentiments)
            }
        else:
            return {
                'avg_score': 0,
                'sentiment': "Neutral",
                'num_articles': 0
            }


class TechnicalAnalyzer:
    """Performs technical analysis on stock data"""
    
    def __init__(self):
        pass
    
    def add_indicators(self, df):
        """Add technical indicators to dataframe"""
        if df is None or len(df) < 30:
            return None
            
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Add basic indicators
        # Moving averages
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        
        # MACD
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        
        # RSI
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        
        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20)
        df = pd.concat([df, bbands], axis=1)
        
        # Average True Range
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Volume indicators
        df['Volume_SMA_20'] = ta.sma(df['Volume'], length=20)
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Price momentum
        df['ROC_10'] = ta.roc(df['Close'], length=10)
        
        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        return df
    
    def identify_signals(self, df):
        """Identify trading signals based on technical indicators"""
        if df is None or len(df) < 50:
            return {}
            
        signals = {}
        
        # Check for Moving Average crossovers
        if df['SMA_20'].iloc[-2] < df['SMA_50'].iloc[-2] and df['SMA_20'].iloc[-1] >= df['SMA_50'].iloc[-1]:
            signals['MA_CROSSOVER'] = "Bullish: 20-day SMA crossed above 50-day SMA"
        elif df['SMA_20'].iloc[-2] > df['SMA_50'].iloc[-2] and df['SMA_20'].iloc[-1] <= df['SMA_50'].iloc[-1]:
            signals['MA_CROSSOVER'] = "Bearish: 20-day SMA crossed below 50-day SMA"
            
        # MACD signals
        if df['MACD_12_26_9'].iloc[-2] < df['MACDs_12_26_9'].iloc[-2] and df['MACD_12_26_9'].iloc[-1] >= df['MACDs_12_26_9'].iloc[-1]:
            signals['MACD'] = "Bullish: MACD crossed above signal line"
        elif df['MACD_12_26_9'].iloc[-2] > df['MACDs_12_26_9'].iloc[-2] and df['MACD_12_26_9'].iloc[-1] <= df['MACDs_12_26_9'].iloc[-1]:
            signals['MACD'] = "Bearish: MACD crossed below signal line"
            
        # RSI signals
        last_rsi = df['RSI_14'].iloc[-1]
        if last_rsi < 30:
            signals['RSI'] = f"Oversold: RSI at {last_rsi:.2f}"
        elif last_rsi > 70:
            signals['RSI'] = f"Overbought: RSI at {last_rsi:.2f}"
            
        # Bollinger Band signals
        last_close = df['Close'].iloc[-1]
        upper_band = df['BBU_20_2.0'].iloc[-1]
        lower_band = df['BBL_20_2.0'].iloc[-1]
        
        if last_close > upper_band:
            signals['BBANDS'] = "Price above upper Bollinger Band - potential reversal or continuation"
        elif last_close < lower_band:
            signals['BBANDS'] = "Price below lower Bollinger Band - potential reversal or continuation"
            
        # Trend detection
        if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
            signals['TREND'] = "Strong uptrend: 20 SMA > 50 SMA > 200 SMA"
        elif df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] < df['SMA_200'].iloc[-1]:
            signals['TREND'] = "Strong downtrend: 20 SMA < 50 SMA < 200 SMA"
            
        # Volume confirmation
        if df['Volume'].iloc[-1] > df['Volume_SMA_20'].iloc[-1] * 1.5:
            signals['VOLUME'] = "Unusually high volume - possible strong move confirmation"
            
        return signals
    
    def calculate_support_resistance(self, df, window=20):
        """Calculate support and resistance levels"""
        if df is None or len(df) < window * 2:
            return None, None
            
        # Find local minima and maxima in the window
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            # Check if this is a local maximum
            if all(df['High'].iloc[i] >= df['High'].iloc[i-j] for j in range(1, window)) and \
               all(df['High'].iloc[i] >= df['High'].iloc[i+j] for j in range(1, window)):
                highs.append(df['High'].iloc[i])
                
            # Check if this is a local minimum
            if all(df['Low'].iloc[i] <= df['Low'].iloc[i-j] for j in range(1, window)) and \
               all(df['Low'].iloc[i] <= df['Low'].iloc[i+j] for j in range(1, window)):
                lows.append(df['Low'].iloc[i])
        
        # Group close levels together (within 2% of each other)
        if highs:
            highs = sorted(highs, reverse=True)
            grouped_highs = []
            current_group = [highs[0]]
            
            for i in range(1, len(highs)):
                if abs(highs[i] - current_group[0]) / current_group[0] <= 0.02:  # Within 2%
                    current_group.append(highs[i])
                else:
                    grouped_highs.append(sum(current_group) / len(current_group))
                    current_group = [highs[i]]
                    
            if current_group:
                grouped_highs.append(sum(current_group) / len(current_group))
                
            resistance_levels = grouped_highs[:3]  # Top 3 resistance levels
        else:
            resistance_levels = []
            
        if lows:
            lows = sorted(lows)
            grouped_lows = []
            current_group = [lows[0]]
            
            for i in range(1, len(lows)):
                if abs(lows[i] - current_group[0]) / current_group[0] <= 0.02:  # Within 2%
                    current_group.append(lows[i])
                else:
                    grouped_lows.append(sum(current_group) / len(current_group))
                    current_group = [lows[i]]
                    
            if current_group:
                grouped_lows.append(sum(current_group) / len(current_group))
                
            support_levels = grouped_lows[:3]  # Top 3 support levels
        else:
            support_levels = []
            
        return support_levels, resistance_levels

    def get_summary_stats(self, df):
        """Get summary statistics of price action"""
        if df is None or len(df) < 30:
            return {}
            
        try:
            # Calculate metrics over last 30 days
            recent_df = df.iloc[-30:].copy()
            
            # Volatility (standard deviation of returns)
            volatility_30d = recent_df['Daily_Return'].std() * (252 ** 0.5)  # Annualized
            
            # Maximum drawdown
            cumulative_returns = (1 + recent_df['Daily_Return']).cumprod()
            max_return = cumulative_returns.cummax()
            drawdown = (cumulative_returns / max_return - 1)
            max_drawdown = drawdown.min()
            
            # Performance metrics
            total_return_30d = recent_df['Close'].iloc[-1] / recent_df['Close'].iloc[0] - 1
            
            # Up vs Down days
            up_days = sum(1 for x in recent_df['Daily_Return'] if x > 0)
            down_days = sum(1 for x in recent_df['Daily_Return'] if x < 0)
            
            # Average volume
            avg_volume = recent_df['Volume'].mean()
            
            return {
                'volatility_30d': volatility_30d,
                'max_drawdown_30d': max_drawdown,
                'total_return_30d': total_return_30d,
                'up_days': up_days,
                'down_days': down_days,
                'up_down_ratio': up_days / down_days if down_days > 0 else float('inf'),
                'avg_volume': avg_volume,
                'current_rsi': df['RSI_14'].iloc[-1]
            }
        except Exception as e:
            print(f"Error calculating summary stats: {e}")
            return {}


class StockPredictor:
    """Machine learning model to predict stock price movements"""
    
    def __init__(self):
        self.model = None
        self.features = None
        self.scaler = StandardScaler()
    
    def engineer_features(self, df):
        """Create features for prediction model"""
        if df is None or len(df) < 50:
            return None
            
        features_df = df.copy()
        
        # Calculate price differences and percentage changes
        features_df['price_diff'] = features_df['Close'].diff()
        features_df['pct_change'] = features_df['Close'].pct_change()
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features_df[f'rolling_mean_{window}'] = features_df['Close'].rolling(window=window).mean()
            features_df[f'rolling_std_{window}'] = features_df['Close'].rolling(window=window).std()
            features_df[f'rolling_volume_{window}'] = features_df['Volume'].rolling(window=window).mean()
        
        # Relative positions
        for window in [5, 10, 20]:
            features_df[f'close_rel_mean_{window}'] = features_df['Close'] / features_df[f'rolling_mean_{window}']
        
        # Technical indicator features
        features_df['above_200ma'] = (features_df['Close'] > features_df['SMA_200']).astype(int)
        features_df['above_50ma'] = (features_df['Close'] > features_df['SMA_50']).astype(int)
        
        # RSI indicators
        features_df['rsi_oversold'] = (features_df['RSI_14'] < 30).astype(int)
        features_df['rsi_overbought'] = (features_df['RSI_14'] > 70).astype(int)
        
        # MACD features
        features_df['macd_above_signal'] = (features_df['MACD_12_26_9'] > features_df['MACDs_12_26_9']).astype(int)
        
        # Bollinger Band features
        features_df['bb_above_upper'] = (features_df['Close'] > features_df['BBU_20_2.0']).astype(int)
        features_df['bb_below_lower'] = (features_df['Close'] < features_df['BBL_20_2.0']).astype(int)
        
        # Create target: next day's price movement
        features_df['target'] = features_df['Close'].shift(-1) - features_df['Close']
        features_df['target_direction'] = (features_df['target'] > 0).astype(int)
        
        # Drop NaN values
        features_df = features_df.dropna()
        
        return features_df
    
    def train(self, df):
        """Train the prediction model"""
        try:
            features_df = self.engineer_features(df)
            if features_df is None or len(features_df) < 50:
                print("Not enough data for model training")
                return False
                
            # Define feature columns to use
            self.features = [
                'RSI_14', 'ROC_10', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
                'price_diff', 'pct_change',
                'rolling_mean_5', 'rolling_std_5', 'rolling_volume_5',
                'rolling_mean_10', 'rolling_std_10', 'rolling_volume_10',
                'rolling_mean_20', 'rolling_std_20', 'rolling_volume_20',
                'close_rel_mean_5', 'close_rel_mean_10', 'close_rel_mean_20',
                'above_200ma', 'above_50ma', 'rsi_oversold', 'rsi_overbought',
                'macd_above_signal', 'bb_above_upper', 'bb_below_lower'
            ]
            
            # Prepare data
            X = features_df[self.features].values
            y = features_df['target'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
            
            # Train model
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Model trained - MSE: {mse:.4f}, R²: {r2:.4f}")
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_next_day(self, df):
        """Predict the next day's price movement"""
        if self.model is None or df is None or len(df) < max(50, len(self.features)):
            return None
            
        try:
            features_df = self.engineer_features(df)
            if features_df is None:
                return None
                
            # Get the latest data point
            X = features_df[self.features].iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            current_price = df['Close'].iloc[-1]
            predicted_price = current_price + prediction
            predicted_change = prediction / current_price
            
            result = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change': predicted_change,
                'predicted_direction': 'Up' if prediction > 0 else 'Down',
                'confidence': min(0.99, max(0.50, abs(predicted_change * 20)))  # Simple scaling for confidence
            }
            
            return result
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None


class PortfolioManager:
    """Manages portfolio allocation and risk"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {ticker: {'shares': n, 'entry_price': p}}
        self.transactions = []
        self.risk_per_trade = 0.02  # Maximum risk per trade (2% of portfolio)
        self.portfolio_stats = {
            'start_date': datetime.now().strftime('%Y-%m-%d'),
            'current_value': initial_capital,
            'returns': 0.0,
            'drawdown': 0.0,
            'sharpe': 0.0
        }
    
    def calculate_position_size(self, ticker, entry_price, stop_loss):
        """Calculate position size based on risk management"""
        if entry_price <= stop_loss:
            return 0  # Invalid stop loss for long position
            
        risk_amount = self.portfolio_value() * self.risk_per_trade
        price_risk = entry_price - stop_loss
        
        if price_risk <= 0:
            return 0
            
        shares = risk_amount / price_risk
        max_position = self.portfolio_value() * 0.2  # Max 20% in one position
        
        if shares * entry_price > max_position:
            shares = max_position / entry_price
            
        return int(shares)
    
    def portfolio_value(self):
        """Calculate current portfolio value"""
        positions_value = sum(
            position['shares'] * self.get_current_price(ticker)
            for ticker, position in self.positions.items()
        )
        return self.cash + positions_value
    
    def get_current_price(self, ticker):
        """Get current price for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period='1d')
            if data.empty:
                return None
            return data['Close'].iloc[-1]
        except:
            # If error, use the last known entry price
            if ticker in self.positions:
                return self.positions[ticker]['entry_price']
            return None
    
    def buy(self, ticker, shares=None, capital_fraction=None, stop_loss=None, take_profit=None):
        """
        Buy a position
        shares: Number of shares to buy
        capital_fraction: Alternatively, fraction of available capital to use
        """
        current_price = self.get_current_price(ticker)
        if current_price is None:
            return False, "Could not get current price"
        
        if shares is None and capital_fraction is not None:
            # Calculate shares based on capital fraction
            max_capital = self.cash * capital_fraction
            shares = int(max_capital / current_price)
        
        if shares is None and stop_loss is not None:
            # Calculate shares based on risk management
            shares = self.calculate_position_size(ticker, current_price, stop_loss)
        
        if not shares or shares <= 0:
            return False, "Invalid number of shares"
        
        cost = shares * current_price
        if cost > self.cash:
            affordable_shares = int(self.cash / current_price)
            if affordable_shares <= 0:
                return False, "Not enough cash available"
            shares = affordable_shares
            cost = shares * current_price
            
        # Execute the buy
        self.cash -= cost
        
        if ticker in self.positions:
            # Update existing position (average down/up)
            total_shares = self.positions[ticker]['shares'] + shares
            total_cost = (self.positions[ticker]['shares'] * self.positions[ticker]['entry_price']) + cost
            self.positions[ticker] = {
                'shares': total_shares,
                'entry_price': total_cost / total_shares,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_date': datetime.now().strftime('%Y-%m-%d')
            }
        else:
            # New position
            self.positions[ticker] = {
                'shares': shares,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_date': datetime.now().strftime('%Y-%m-%d')
            }
        
        # Record transaction
        self.transactions.append({
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'BUY',
            'ticker': ticker,
            'shares': shares,
            'price': current_price,
            'value': cost
        })
        
        return True, f"Bought {shares} shares of {ticker} at {current_price:.2f}"
    
    def sell(self, ticker, shares=None, sell_all=False):
        """
        Sell a position
        shares: Number of shares to sell
        sell_all: Whether to sell the entire position
        """
        if ticker not in self.positions:
            return False, f"No position in {ticker}"
            
        current_price = self.get_current_price(ticker)
        if current_price is None:
            return False, "Could not get current price"
            
        if sell_all or shares is None:
            shares = self.positions[ticker]['shares']
        
        if shares > self.positions[ticker]['shares']:
            shares = self.positions[ticker]['shares']
            
        proceeds = shares * current_price
        self.cash += proceeds
        
        # Update or remove position
        if shares == self.positions[ticker]['shares']:
            entry_price = self.positions[ticker]['entry_price']
            profit_loss = (current_price - entry_price) * shares
        
        # Record transaction
        self.transactions.append({
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'SELL',
            'ticker': ticker,
            'shares': shares,
            'price': current_price,
            'value': proceeds,
            'profit_loss': (current_price - entry_price) * shares
        })
        
        return True, f"Sold {shares} shares of {ticker} at {current_price:.2f}, P&L: {(current_price - entry_price) * shares:.2f}"
    
    def check_stop_loss(self):
        """Check if any positions have hit their stop loss"""
        triggered = []
        for ticker, position in list(self.positions.items()):
            if position.get('stop_loss') is not None:
                current_price = self.get_current_price(ticker)
                if current_price and current_price <= position['stop_loss']:
                    success, message = self.sell(ticker, sell_all=True)
                    if success:
                        triggered.append({
                            'ticker': ticker,
                            'action': 'Sold (Stop Loss)',
                            'shares': position['shares'],
                            'entry': position['entry_price'],
                            'exit': current_price,
                            'loss': (current_price - position['entry_price']) * position['shares']
                        })
        return triggered
    
    def check_take_profit(self):
        """Check if any positions have hit their take profit level"""
        triggered = []
        for ticker, position in list(self.positions.items()):
            if position.get('take_profit') is not None:
                current_price = self.get_current_price(ticker)
                if current_price and current_price >= position['take_profit']:
                    success, message = self.sell(ticker, sell_all=True)
                    if success:
                        triggered.append({
                            'ticker': ticker,
                            'action': 'Sold (Take Profit)',
                            'shares': position['shares'],
                            'entry': position['entry_price'],
                            'exit': current_price,
                            'profit': (current_price - position['entry_price']) * position['shares']
                        })
        return triggered
    
    def calculate_portfolio_stats(self):
        """Calculate portfolio performance statistics"""
        current_value = self.portfolio_value()
        
        # Calculate returns
        returns = (current_value / self.initial_capital) - 1
        
        # Create a history of transactions to calculate drawdown and Sharpe ratio
        if len(self.transactions) > 0:
            # This would be more sophisticated in a real implementation
            # For now, just use the returns
            max_value = max(current_value, self.initial_capital)
            drawdown = (current_value - max_value) / max_value if max_value > 0 else 0
            
            # Simple Sharpe ratio approximation (assumes risk-free rate of 0)
            # In practice, we would calculate this properly using daily returns
            if returns > 0:
                sharpe = returns / 0.15  # Assume 15% volatility as a placeholder
            else:
                sharpe = 0
        else:
            drawdown = 0
            sharpe = 0
        
        self.portfolio_stats = {
            'start_date': self.portfolio_stats['start_date'],
            'current_date': datetime.now().strftime('%Y-%m-%d'),
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'cash': self.cash,
            'invested': current_value - self.cash,
            'returns': returns,
            'returns_pct': returns * 100,
            'drawdown': drawdown,
            'drawdown_pct': drawdown * 100,
            'sharpe': sharpe
        }
        
        return self.portfolio_stats
    
    def get_positions_summary(self):
        """Get a summary of current positions"""
        summary = []
        for ticker, position in self.positions.items():
            current_price = self.get_current_price(ticker)
            if current_price:
                market_value = position['shares'] * current_price
                cost_basis = position['shares'] * position['entry_price']
                unrealized_pl = market_value - cost_basis
                unrealized_pl_pct = (current_price / position['entry_price'] - 1) * 100
                
                summary.append({
                    'ticker': ticker,
                    'shares': position['shares'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'market_value': market_value,
                    'cost_basis': cost_basis,
                    'unrealized_pl': unrealized_pl,
                    'unrealized_pl_pct': unrealized_pl_pct,
                    'stop_loss': position.get('stop_loss'),
                    'take_profit': position.get('take_profit'),
                    'entry_date': position.get('entry_date', 'N/A')
                })
        
        return summary


class InvestmentAdvisor:
    """AI-powered investment advisor that integrates all components"""
    
    def __init__(self, initial_capital=100000):
        self.data_collector = MarketDataCollector()
        self.news_collector = NewsCollector()
        self.tech_analyzer = TechnicalAnalyzer()
        self.stock_predictor = StockPredictor()
        self.portfolio = PortfolioManager(initial_capital)
        
    def analyze_stock(self, ticker, period="1y", interval="1d"):
        """Comprehensive analysis of a stock"""
        # Get stock data
        stock_data = self.data_collector.get_stock_data(ticker, period, interval)
        if stock_data is None:
            return None, "Failed to retrieve stock data"
            
        # Get fundamentals
        fundamentals = self.data_collector.get_company_fundamentals(ticker)
        
        # Add technical indicators
        stock_data_with_indicators = self.tech_analyzer.add_indicators(stock_data)
        
        # Get technical signals
        signals = self.tech_analyzer.identify_signals(stock_data_with_indicators)
        
        # Calculate support and resistance
        support, resistance = self.tech_analyzer.calculate_support_resistance(stock_data)
        
        # Get summary statistics
        summary_stats = self.tech_analyzer.get_summary_stats(stock_data_with_indicators)
        
        # Get news and sentiment
        news = self.news_collector.get_latest_news(ticker, num_articles=10)
        
        # Average sentiment score
        avg_sentiment = sum(article['sentiment']['compound'] for article in news) / len(news) if news else 0
        
        # Try to generate price prediction
        prediction = None
        if stock_data_with_indicators is not None and len(stock_data_with_indicators) >= 100:
            # Train the model if not already trained
            if self.stock_predictor.model is None:
                self.stock_predictor.train(stock_data_with_indicators)
            
            # Get prediction
            prediction = self.stock_predictor.predict_next_day(stock_data_with_indicators)
        
        # Calculate entry/exit points
        current_price = stock_data['Close'].iloc[-1] if stock_data is not None and not stock_data.empty else None
        
        if current_price and prediction and prediction['predicted_direction'] == 'Up':
            # Bullish scenario
            stop_loss = current_price * 0.95  # 5% stop loss
            take_profit = current_price * (1 + 2 * abs(prediction['predicted_change']))  # 2x risk/reward
        elif current_price and prediction and prediction['predicted_direction'] == 'Down':
            # Bearish scenario - no buy recommendation
            stop_loss = None
            take_profit = None
        else:
            # No clear prediction
            stop_loss = current_price * 0.93 if current_price else None
            take_profit = current_price * 1.15 if current_price else None
        
        # Determine recommendation
        recommendation = self._generate_recommendation(
            signals, summary_stats, avg_sentiment, 
            prediction, support, resistance, current_price
        )
        
        # Comprehensive analysis report
        analysis = {
            'ticker': ticker,
            'current_price': current_price,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'fundamentals': fundamentals,
            'technical': {
                'signals': signals,
                'support_levels': support,
                'resistance_levels': resistance,
                'summary_stats': summary_stats
            },
            'news': {
                'articles': news,
                'avg_sentiment': avg_sentiment,
                'sentiment_label': 'Positive' if avg_sentiment > 0.05 else 'Negative' if avg_sentiment < -0.05 else 'Neutral'
            },
            'prediction': prediction,
            'recommendation': recommendation,
            'risk_management': {
                'suggested_stop_loss': stop_loss,
                'suggested_take_profit': take_profit,
                'risk_reward_ratio': abs((take_profit - current_price) / (current_price - stop_loss)) if stop_loss and take_profit and current_price else None
            }
        }
        
        return analysis, "Analysis completed successfully"
    
    def _generate_recommendation(self, signals, stats, sentiment, prediction, support, resistance, current_price):
        """Generate a trading recommendation based on all signals"""
        # Count positive and negative signals
        positive_signals = 0
        negative_signals = 0
        neutral_signals = 0
        
        # Technical signals
        for signal_type, signal in signals.items():
            if "Bullish" in signal:
                positive_signals += 1
            elif "Bearish" in signal:
                negative_signals += 1
            elif "Oversold" in signal:
                positive_signals += 0.5
            elif "Overbought" in signal:
                negative_signals += 0.5
            else:
                neutral_signals += 1
                
        # Sentiment
        if sentiment > 0.2:
            positive_signals += 1
        elif sentiment < -0.2:
            negative_signals += 1
            
        # Prediction
        if prediction and prediction['predicted_direction'] == 'Up' and prediction['confidence'] > 0.6:
            positive_signals += 1
        elif prediction and prediction['predicted_direction'] == 'Down' and prediction['confidence'] > 0.6:
            negative_signals += 1
            
        # Support/Resistance
        if support and resistance and current_price:
            closest_support = min(support, key=lambda x: abs(x - current_price))
            closest_resistance = min(resistance, key=lambda x: abs(x - current_price))
            
            # If price is close to support (within 3%)
            if abs(closest_support - current_price) / current_price < 0.03:
                positive_signals += 0.5
                
            # If price is close to resistance (within 3%)
            if abs(closest_resistance - current_price) / current_price < 0.03:
                negative_signals += 0.5
                
        # Performance stats
        if stats.get('up_down_ratio', 1) > 1.5:
            positive_signals += 0.5
        elif stats.get('up_down_ratio', 1) < 0.7:
            negative_signals += 0.5
            
        # RSI conditions
        rsi = stats.get('current_rsi', 50)
        if rsi < 30:
            positive_signals += 0.5  # Oversold
        elif rsi > 70:
            negative_signals += 0.5  # Overbought
            
        # Determine overall recommendation
        total_score = positive_signals - negative_signals
        
        if total_score >= 2:
            action = "STRONG BUY"
            rationale = "Multiple strong bullish signals indicate potential upside"
        elif total_score >= 1:
            action = "BUY"
            rationale = "Bullish signals slightly outweigh bearish ones"
        elif total_score <= -2:
            action = "STRONG SELL"
            rationale = "Multiple strong bearish signals indicate potential downside"
        elif total_score <= -1:
            action = "SELL"
            rationale = "Bearish signals slightly outweigh bullish ones"
        else:
            action = "HOLD"
            rationale = "Mixed signals suggest waiting for clearer direction"
            
        return {
            'action': action,
            'rationale': rationale,
            'score': total_score,
            'positive_signals': positive_signals,
            'negative_signals': negative_signals,
            'neutral_signals': neutral_signals
        }
    
    def build_portfolio(self, capital=None, risk_level='moderate', sectors=None, num_stocks=5):
        """Build a recommended portfolio based on market analysis"""
        if capital:
            # Reset portfolio with new capital
            self.portfolio = PortfolioManager(capital)
        
        # Define risk parameters based on risk level
        if risk_level == 'conservative':
            max_allocation_per_stock = 0.15  # Maximum 15% in any one stock
            volatility_threshold = 0.25  # Prefer lower volatility stocks
            risk_tolerance = 0.01  # Willing to risk 1% per trade
        elif risk_level == 'aggressive':
            max_allocation_per_stock = 0.30  # Up to 30% in one stock
            volatility_threshold = 0.40  # Can handle higher volatility
            risk_tolerance = 0.03  # Willing to risk 3% per trade
        else:  # moderate (default)
            max_allocation_per_stock = 0.20  # Maximum 20% in any one stock
            volatility_threshold = 0.30  # Moderate volatility
            risk_tolerance = 0.02  # Willing to risk 2% per trade
        
        # Update portfolio risk settings
        self.portfolio.risk_per_trade = risk_tolerance
        
        # Sector preferences
        if not sectors:
            sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer Discretionary', 'Industrials']
        
        # Screening criteria based on risk level
        if risk_level == 'conservative':
            criteria = {
                'min_marketCap': 10000000000,  # $10B+ market cap
                'max_PE': 25,
                'min_30d_performance': -5  # Allow slight underperformers
            }
        elif risk_level == 'aggressive':
            criteria = {
                'min_marketCap': 1000000000,  # $1B+ market cap
                'max_PE': 50,
                'min_30d_performance': -10  # Allow more volatility
            }
        else:  # moderate
            criteria = {
                'min_marketCap': 5000000000,  # $5B+ market cap
                'max_PE': 35,
                'min_30d_performance': -7
            }
        
        # Screen stocks
        screened_stocks = self.data_collector.screen_stocks(criteria)
        
        # Analyze top candidates
        candidates = []
        for _, row in screened_stocks.iterrows():
            ticker = row['ticker']
            analysis, _ = self.analyze_stock(ticker)
            if analysis:
                recommendation = analysis['recommendation']
                if recommendation['action'] in ['BUY', 'STRONG BUY']:
                    candidates.append({
                        'ticker': ticker,
                        'score': recommendation['score'],
                        'analysis': analysis,
                        'sector': row.get('sector', 'Unknown')
                    })
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top stocks with sector diversification
        selected_stocks = []
        selected_sectors = set()
        
        for candidate in candidates:
            sector = candidate['sector']
            if len(selected_stocks) < num_stocks:
                # Ensure sector diversification
                if sector not in selected_sectors or len(selected_sectors) >= len(sectors):
                    selected_stocks.append(candidate)
                    selected_sectors.add(sector)
                    
        if not selected_stocks:
            return None, "No suitable stocks found for the portfolio"
        
        # Calculate allocation for each stock
        total_portfolio = self.portfolio.portfolio_value()
        remaining_capital = total_portfolio
        
        portfolio_plan = []
        
        for stock in selected_stocks:
            ticker = stock['ticker']
            analysis = stock['analysis']
            
            # Allocate capital based on score and risk level
            score = stock['score']
            allocation_pct = min(
                max_allocation_per_stock,
                0.10 + (score * 0.05)  # Base 10% + adjustment based on score
            )
            
            allocation_amount = total_portfolio * allocation_pct
            if allocation_amount > remaining_capital:
                allocation_amount = remaining_capital
                
            current_price = analysis['current_price']
            if current_price and allocation_amount > 0:
                shares = int(allocation_amount / current_price)
                stop_loss = analysis['risk_management']['suggested_stop_loss']
                take_profit = analysis['risk_management']['suggested_take_profit']
                
                portfolio_plan.append({
                    'ticker': ticker,
                    'shares': shares,
                    'allocation': allocation_amount,
                    'allocation_pct': allocation_pct * 100,
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'recommendation': analysis['recommendation']['action'],
                    'recommendation_rationale': analysis['recommendation']['rationale']
                })
                
                remaining_capital -= allocation_amount
        
        # Execute the trades
        executed_trades = []
        for stock in portfolio_plan:
            ticker = stock['ticker']
            shares = stock['shares']
            stop_loss = stock['stop_loss']
            take_profit = stock['take_profit']
            
            success, message = self.portfolio.buy(
                ticker, shares=shares, 
                stop_loss=stop_loss, 
                take_profit=take_profit
            )
            
            executed_trades.append({
                'ticker': ticker,
                'success': success,
                'message': message,
                'planned_shares': shares,
                'planned_allocation': stock['allocation']
            })
        
        # Calculate portfolio stats
        stats = self.portfolio.calculate_portfolio_stats()
        
        # Return the portfolio summary
        return {
            'risk_level': risk_level,
            'portfolio_value': stats['current_value'],
            'cash_remaining': stats['cash'],
            'trades': executed_trades,
            'positions': self.portfolio.get_positions_summary(),
            'portfolio_plan': portfolio_plan
        }, "Portfolio successfully built"
    
    def rebalance_portfolio(self, reallocate_pct=0.3):
        """Rebalance the portfolio by selling underperformers and buying promising stocks"""
        positions = self.portfolio.get_positions_summary()
        
        # Identify underperformers
        underperformers = []
        for position in positions:
            # Analyze current holding
            analysis, _ = self.analyze_stock(position['ticker'])
            if analysis and analysis['recommendation']['action'] in ['SELL', 'STRONG SELL']:
                underperformers.append({
                    'ticker': position['ticker'],
                    'shares': position['shares'],
                    'market_value': position['market_value'],
                    'recommendation': analysis['recommendation']
                })
        
        # Sell underperformers
        sold_value = 0
        for stock in underperformers:
            ticker = stock['ticker']
            success, message = self.portfolio.sell(ticker, sell_all=True)
            if success:
                sold_value += stock['market_value']
        
        # Calculate how much capital to reallocate
        total_value = self.portfolio.portfolio_value()
        reallocate_value = sold_value + (self.portfolio.cash * reallocate_pct)
        
        # Find new opportunities
        _, new_portfolio = self.build_portfolio(
            capital=None,  # Don't reset the portfolio
            risk_level='moderate',
            num_stocks=max(1, int(len(underperformers) * 1.5))  # Replace with slightly more diversification
        )
        
        # Return rebalance summary
        return {
            'previous_value': total_value,
            'new_value': self.portfolio.portfolio_value(),
            'sold_positions': underperformers,
            'sold_value': sold_value,
            'new_positions': self.portfolio.get_positions_summary(),
            'cash': self.portfolio.cash
        }, "Portfolio successfully rebalanced"
    
    def generate_market_outlook(self):
        """Generate overall market outlook and sentiment"""
        # Get major indices data
        indices = self.data_collector.get_market_indices()
        
        # Get sector performance
        sectors = self.data_collector.get_sector_performance()
        
        # Analyze index technical signals
        index_signals = {}
        for index_name, index_data in indices.items():
            if index_data is not None:
                with_indicators = self.tech_analyzer.add_indicators(index_data)
                signals = self.tech_analyzer.identify_signals(with_indicators)
                stats = self.tech_analyzer.get_summary_stats(with_indicators)
                index_signals[index_name] = {
                    'signals': signals,
                    'stats': stats
                }
        
        # Analyze sector trends
        sector_trends = {}
        for sector_name, sector_data in sectors.items():
            if sector_data is not None:
                recent_return = sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[-22] - 1 if len(sector_data) >= 22 else None
                sector_trends[sector_name] = {
                    'monthly_return': recent_return * 100 if recent_return else None,
                    'trend': 'Bullish' if recent_return and recent_return > 0.03 else 
                            'Bearish' if recent_return and recent_return < -0.03 else 'Neutral'
                }
        
        # Get market sentiment from news
        market_tickers = ["SPY", "QQQ", "DIA", "IWM", "VIX"]
        market_sentiment = self.news_collector.get_market_sentiment(market_tickers)
        
        # Determine overall market direction
        bullish_signals = 0
        bearish_signals = 0
        
        # Check index signals
        for index, data in index_signals.items():
            for signal_type, signal in data['signals'].items():
                if "Bullish" in signal:
                    bullish_signals += 1
                elif "Bearish" in signal:
                    bearish_signals += 1
                    
        # Check sector trends
        bullish_sectors = sum(1 for sector, data in sector_trends.items() if data['trend'] == 'Bullish')
        bearish_sectors = sum(1 for sector, data in sector_trends.items() if data['trend'] == 'Bearish')
        
        bullish_signals += bullish_sectors * 0.5
        bearish_signals += bearish_sectors * 0.5
        
        # Consider sentiment
        if market_sentiment['sentiment'] == 'Positive':
            bullish_signals += 1
        elif market_sentiment['sentiment'] == 'Negative':
            bearish_signals += 1
            
        # Determine market direction
        if bullish_signals > bearish_signals + 2:
            market_direction = "Strongly Bullish"
        elif bullish_signals > bearish_signals:
            market_direction = "Moderately Bullish"
        elif bearish_signals > bullish_signals + 2:
            market_direction = "Strongly Bearish"
        elif bearish_signals > bullish_signals:
            market_direction = "Moderately Bearish"
        else:
            market_direction = "Neutral"
            
        # Identify top-performing sectors
        sorted_sectors = sorted(
            [(sector, data['monthly_return']) for sector, data in sector_trends.items() if data['monthly_return']],
            key=lambda x: x[1],
            reverse=True
        )
        
        top_sectors = sorted_sectors[:3]
        bottom_sectors = sorted_sectors[-3:]
        
        # Generate market summary
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'market_direction': market_direction,
            'market_sentiment': market_sentiment,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'index_analysis': index_signals,
            'sector_trends': sector_trends,
            'top_sectors': top_sectors,
            'bottom_sectors': bottom_sectors,
            'outlook_summary': f"Market outlook: {market_direction}. "
                          f"News sentiment: {market_sentiment['sentiment']}. "
                          f"Top performing sectors: {', '.join([s[0] for s in top_sectors])}."
        }
    
    def generate_report(self, portfolio_id=None):
        """Generate a comprehensive portfolio and market report"""
        # Get portfolio stats
        portfolio_stats = self.portfolio.calculate_portfolio_stats()
        positions = self.portfolio.get_positions_summary()
        
        # Get market outlook
        market_outlook = self.generate_market_outlook()
        
        # Get individual position analyses
        position_analyses = []
        for position in positions:
            ticker = position['ticker']
            analysis, _ = self.analyze_stock(ticker)
            if analysis:
                position_analyses.append({
                    'ticker': ticker,
                    'position': position,
                    'analysis': analysis
                })
        
        # Generate recommendations
        recommendations = []
        for analysis in position_analyses:
            ticker = analysis['ticker']
            position = analysis['position']
            stock_analysis = analysis['analysis']
            
            # Determine if action needed
            current_action = stock_analysis['recommendation']['action']
            
            if current_action in ['SELL', 'STRONG SELL']:
                recommendations.append({
                    'ticker': ticker,
                    'action': 'SELL',
                    'rationale': stock_analysis['recommendation']['rationale'],
                    'urgency': 'High' if current_action == 'STRONG SELL' else 'Medium',
                    'current_position': position
                })
            elif current_action == 'STRONG BUY' and position['unrealized_pl_pct'] < -5:
                recommendations.append({
                    'ticker': ticker,
                    'action': 'AVERAGE DOWN',
                    'rationale': "Strong buy signal while position is down. Consider adding to position.",
                    'urgency': 'Medium',
                    'current_position': position
                })
            elif current_action == 'HOLD' and position['unrealized_pl_pct'] > 20:
                recommendations.append({
                    'ticker': ticker,
                    'action': 'TAKE PARTIAL PROFITS',
                    'rationale': "Significant gains achieved. Consider securing partial profits.",
                    'urgency': 'Low',
                    'current_position': position
                })
        
        # Look for new opportunities
        criteria = {'min_marketCap': 5000000000, 'min_30d_performance': 5}
        opportunities = self.data_collector.screen_stocks(criteria)
        
        top_opportunities = []
        for _, stock in opportunities.iterrows():
            ticker = stock['ticker']
            analysis, _ = self.analyze_stock(ticker)
            if analysis and analysis['recommendation']['action'] == 'STRONG BUY':
                top_opportunities.append({
                    'ticker': ticker,
                    'price': stock['price'],
                    'sector': stock['sector'],
                    'recommendation': analysis['recommendation']
                })
                if len(top_opportunities) >= 3:
                    break
        
        # Generate report
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'portfolio_id': portfolio_id,
            'portfolio_summary': portfolio_stats,
            'positions': positions,
            'market_outlook': market_outlook,
            'recommendations': recommendations,
            'opportunities': top_opportunities,
            'transactions': self.portfolio.transactions[-10:] if len(self.portfolio.transactions) > 0 else []
        }
        
        return report


# Helper functions for UI demo
def format_currency(value):
    """Format a value as currency"""
    if value is None:
        return "N/A"
    return f"${value:,.2f}"

def format_percentage(value):
    """Format a value as percentage"""
    if value is None:
        return "N/A"
    return f"{value:.2f}%"

def demo_analysis(ticker="AAPL"):
    """Run a demo analysis"""
    advisor = InvestmentAdvisor(initial_capital=50000)
    analysis, message = advisor.analyze_stock(ticker)
    
    if analysis:
        print(f"\n===== Analysis for {ticker} =====")
        print(f"Current Price: {format_currency(analysis['current_price'])}")
        print(f"Recommendation: {analysis['recommendation']['action']}")
        print(f"Rationale: {analysis['recommendation']['rationale']}")
        
        print("\n----- Technical Signals -----")
        for signal_type, signal in analysis['technical']['signals'].items():
            print(f"{signal_type}: {signal}")
            
        print("\n----- Support & Resistance -----")
        print(f"Support levels: {[format_currency(level) for level in analysis['technical']['support_levels']]}")
        print(f"Resistance levels: {[format_currency(level) for level in analysis['technical']['resistance_levels']]}")
        
        print("\n----- News Sentiment -----")
        print(f"Average sentiment: {analysis['news']['avg_sentiment']:.2f} ({analysis['news']['sentiment_label']})")
        print("\nLatest news:")
        for article in analysis['news']['articles'][:3]:
            print(f"- {article['title']} (Sentiment: {article['sentiment']['compound']:.2f})")
            
        if analysis['prediction']:
            print("\n----- Price Prediction -----")
            print(f"Predicted direction: {analysis['prediction']['predicted_direction']}")
            print(f"Predicted change: {format_percentage(analysis['prediction']['predicted_change'] * 100)}")
            print(f"Confidence: {format_percentage(analysis['prediction']['confidence'] * 100)}")
            
        print("\n----- Risk Management -----")
        print(f"Suggested entry: {format_currency(analysis['current_price'])}")
        print(f"Suggested stop loss: {format_currency(analysis['risk_management']['suggested_stop_loss'])}")
        print(f"Suggested take profit: {format_currency(analysis['risk_management']['suggested_take_profit'])}")
        
        if analysis['risk_management']['risk_reward_ratio']:
            print(f"Risk/Reward ratio: 1:{analysis['risk_management']['risk_reward_ratio']:.2f}")
    else:
        print(f"Failed to analyze {ticker}: {message}")

def demo_portfolio():
    """Run a demo portfolio construction"""
    advisor = InvestmentAdvisor(initial_capital=50000)
    portfolio, message = advisor.build_portfolio(risk_level='moderate', num_stocks=5)

    if portfolio:
        print("\n===== Portfolio Summary =====")
        print(f"Risk Level: {portfolio['risk_level']}")
        print(f"Portfolio Value: {format_currency(portfolio['portfolio_value'])}")
        print(f"Cash Remaining: {format_currency(portfolio['cash_remaining'])}")
        
        print("\n----- Executed Trades -----")
        for trade in portfolio['trades']:
            print(f"{trade['ticker']}: {trade['message']}")
        
        print("\n----- Current Positions -----")
        for pos in portfolio['positions']:
            print(f"{pos['ticker']} - Shares: {pos['shares']} - P/L: {format_currency(pos['unrealized_pl'])} ({format_percentage(pos['unrealized_pl_pct'])})")
    else:
        print(f"Failed to build portfolio: {message}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GenAI-powered Financial Advisor")
    parser.add_argument('--mode', type=str, default='analysis', choices=['analysis', 'portfolio', 'report'], help='Mode to run')
    parser.add_argument('--ticker', type=str, help='Stock ticker for analysis')

    args = parser.parse_args()

    if args.mode == 'analysis' and args.ticker:
        demo_analysis(args.ticker)
    elif args.mode == 'portfolio':
        demo_portfolio()
    elif args.mode == 'report':
        advisor = InvestmentAdvisor(initial_capital=50000)
        report = advisor.generate_report()
        print(json.dumps(report, indent=2))
    else:
        print("Please provide a valid mode or ticker. Example: --mode analysis --ticker AAPL")
