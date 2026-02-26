"""
ATEE Configuration Manager
Handles environment variables, Firebase initialization, and global settings
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class FirebaseConfig:
    """Firebase configuration dataclass"""
    project_id: str = os.getenv("FIREBASE_PROJECT_ID", "atee-system")
    credential_path: str = os.getenv("FIREBASE_CREDENTIAL_PATH", "./firebase-key.json")
    collection_prefix: str = os.getenv("FIREBASE_COLLECTION_PREFIX", "atee_")

@dataclass
class TradingConfig:
    """Trading system configuration"""
    default_timeframe: str = "1h"
    max_drawdown_limit: float = 0.25  # 25% max drawdown
    min_sharpe_ratio: float = 1.2
    min_win_rate: float = 0.45
    backtest_period_days: int = 365
    warmup_periods: int = 50

@dataclass
class EvolutionConfig:
    """Genetic algorithm configuration"""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elite_count: int = 5

class ConfigManager:
    """Central configuration manager with validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.firebase_config = FirebaseConfig()
        self.trading_config = TradingConfig()
        self.evolution_config = EvolutionConfig()
        self._firestore_client: Optional[Client] = None
        self._validate_config()
        self._init_firebase()
        
    def _validate_config(self) -> None:
        """Validate all configuration values"""
        required_env_vars = ["FIREBASE_PROJECT_ID", "FIREBASE_CREDENTIAL_PATH"]
        missing = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing:
            error_msg = f"Missing required environment variables: {missing}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not os.path.exists(self.firebase_config.credential_path):
            error_msg = f"Firebase credential file not found: {self.firebase_config.credential_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        self.logger.info("Configuration validation passed")
    
    def _init_firebase(self) -> None:
        """Initialize Firebase Admin SDK"""
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.firebase_config.credential_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': self.firebase_config.project_id
                })
                self.logger.info("Firebase Admin SDK initialized successfully")
            
            self._firestore_client = firestore.client()
            self.logger.info("Firestore client connected")
            
        except Exception as e:
            self.logger.error(f"Firebase initialization failed: {str(e)}")
            raise
    
    @property
    def firestore(self) -> Client:
        """Get Firestore client with lazy initialization"""
        if self._firestore_client is None:
            self._init_firebase()
        return self._firestore_client
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all configs as dictionary"""
        return {
            "firebase": asdict(self.firebase_config),
            "trading": asdict(self.trading_config),
            "evolution": asdict(self.evolution_config)
        }

# Global config instance
config = ConfigManager()