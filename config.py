# Configuration settings for the project
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_VERSION = "v1"
TIMEOUT = 180  # seconds

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")  # Primary NVIDIA API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Added for Gemini
AI_PIPELINE_ORG_KEY = os.getenv("AI_PIPELINE_ORG_KEY")  # AI Pipeline organization key
AI_PIPELINE_API_KEY = os.getenv("AI_PIPELINE_API_KEY")  # AI Pipeline API key


# Model Configuration (use valid model IDs; allow override via env)
DEFAULT_NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-405b-instruct")  # NVIDIA NIM model
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
TEMPERATURE = 0.3  # Lower temperature for more accurate analysis
MAX_TOKENS = 4000

# Vector Store Configuration
VECTOR_STORE_TYPE = "chromadb"  # Options: chromadb, faiss
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Chain Configuration
CHAIN_TIMEOUT = 120  # seconds
MAX_RETRIES = 3


def get_openai_client(provider="nvidia"):
    """
    Get an OpenAI client instance configured for different providers.
    :param provider: 'nvidia', 'openai', or 'google'
    """
    try:
        from openai import OpenAI
        
        if provider == "nvidia":
            if not NVIDIA_API_KEY:
                raise ValueError("NVIDIA_API_KEY not set; NVIDIA LLM is not configured")
            return OpenAI(
                api_key=NVIDIA_API_KEY,
                base_url="https://integrate.api.nvidia.com/v1"
            )
        elif provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set; OpenAI LLM is not configured")
            return OpenAI(api_key=OPENAI_API_KEY)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose 'nvidia' or 'openai'.")
    except ImportError:
        raise ImportError("Could not import OpenAI client. Please install openai package.")
