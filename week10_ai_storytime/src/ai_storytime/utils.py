import importlib.util
import os


# Helper function to check if a module can be imported
def _is_module_available(module_name: str) -> bool:
    """Checks if a Python module is available for import."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None


# --- Key Retrieval Strategies ---


def _get_from_colab() -> str | None:
    """Tries to get the API key from Google Colab userdata."""
    if _is_module_available("google.colab"):
        from google.colab import userdata  # type: ignore

        return userdata.get("GEMINI_API_KEY")
    return None


def _get_from_kaggle() -> str | None:
    """Tries to get the API key from Kaggle secrets."""
    if _is_module_available("kaggle_secrets"):
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore

            user_secrets = UserSecretsClient()
            # Use get_secret which might raise if not found, or return the value
            # Wrap in try/except in case get_secret itself fails
            return user_secrets.get_secret("GEMINI_API_KEY")
        except Exception:  # Catch potential errors from get_secret itself
            # You might want more specific error handling depending on kaggle_secrets behavior
            return None
    return None


def _get_from_env() -> str | None:
    """Tries to get the API key from environment variables (optionally using python-dotenv)."""
    if _is_module_available("dotenv"):
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except ImportError:
            # Should not happen due to _is_module_available check, but defensive
            pass
    # Always try os.getenv, as dotenv just loads vars into the environment
    return os.getenv("GEMINI_API_KEY")


# --- Main Function ---


def get_gemini_api_key() -> str:
    """
    Retrieves the GEMINI_API_KEY from various sources in order of preference:
    1. Google Colab userdata
    2. Kaggle secrets
    3. Environment variables (optionally loaded from .env file)

    Raises:
        ValueError: If the GEMINI_API_KEY cannot be found in any source.

    Returns:
        The found GEMINI_API_KEY.
    """
    # Define the order of strategies to try
    strategies = [
        _get_from_colab,
        _get_from_kaggle,
        _get_from_env,
    ]

    api_key: str | None = None
    for strategy in strategies:
        try:
            key = strategy()
            if key:  # Check if key is not None and not an empty string
                api_key = key
                print(f"Found GEMINI_API_KEY using strategy: {strategy.__name__}")
                break  # Key found, stop searching
        except Exception:
            # Log or print a warning if needed, e.g., about Kaggle get_secret failure
            # print(f"Warning: Strategy {strategy.__name__} failed: {e}")
            continue  # Try the next strategy

    if api_key is None:
        raise ValueError(  # Using ValueError might be more semantically correct than ImportError
            "GEMINI_API_KEY not found. Please set it in Colab/Kaggle secrets "
            "or as an environment variable (e.g., in a .env file)."
        )

    return api_key
