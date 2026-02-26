"""
Market data fetching module with CCXT integration
Handles real-time and historical data fetching with caching
"""
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import time
from cachetools import TTLCache

class MarketDataFetcher:
    """Robust market data fetcher with error handling and rate limiting"""
    
    def __init__(self, exchange_id: str = "binance"):
        self.logger = logging.getLogger(__name__)
        self.exchange_id = exchange_id
        self.cache = TTLCache(maxsize=100, ttl=300)  # 5-minute cache
        
        # Initialize exchange with config
        self.exchange = self._init_exchange()
        self.markets = self.exchange.load_markets()
        self.logger.info(f"Initialized {exchange_id} with {len(self.markets)} markets")
    
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange with error handling"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000,
                'rateLimit': 1000  # ms between requests
            })
            
            # Test connection
            exchange.fetch_status()
            self.logger.info(f"Successfully connected to {self.exchange_id}")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.exchange_id}: {str(e)}")
            raise ConnectionError(f"Exchange connection failed: {str(e)}")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with robust error handling
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe string
            since: Timestamp in milliseconds
            limit: Number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}_{since}_{limit}"
        
        # Check cache first
        if cache_key in self.cache:
            self.logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key].copy()
        
        try:
            # Validate symbol
            if symbol not in self.markets:
                raise ValueError(f"Symbol {symbol} not available on {self.exchange_id}")
            
            # Fetch data
            ohlcv = self.exchange.fetch