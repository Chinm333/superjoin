"""
Configuration module for Semantic Spreadsheet Search Engine
"""
import os
from typing import Optional

try:
    from dotenv import load_dotenv
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    env_file = project_root / '.env'
    
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment from: {env_file}")
    else:
        load_dotenv()
        
except ImportError:
    pass

class Config:
    """Configuration class for the application"""
    GEMINI_API_KEY: Optional[str] = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
    GEMINI_MAX_TOKENS: int = int(os.getenv('GEMINI_MAX_TOKENS', '1000'))
    GEMINI_TEMPERATURE: float = float(os.getenv('GEMINI_TEMPERATURE', '0.3'))

    USE_GEMINI: bool = GEMINI_API_KEY is not None
    FALLBACK_TO_RULES: bool = True 
    
    GEMINI_BATCH_SIZE: int = 10 
    GEMINI_TIMEOUT: int = 30  
    
    @classmethod
    def validate_gemini_config(cls) -> bool:
        """Validate Gemini configuration"""
        if not cls.GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not found. Using rule-based approach only.")
            return False
        return True
    
    @classmethod
    def get_gemini_config(cls) -> dict:
        """Get Gemini configuration dictionary"""
        return {
            'api_key': cls.GEMINI_API_KEY,
            'model': cls.GEMINI_MODEL,
            'max_tokens': cls.GEMINI_MAX_TOKENS,
            'temperature': cls.GEMINI_TEMPERATURE,
            'timeout': cls.GEMINI_TIMEOUT
        }
