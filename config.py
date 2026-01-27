import os
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Load environment variables from .env
load_dotenv()

# Model parameters
PARAMETERS = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 356,
}

# watsonx credentials (from .env)
CREDENTIALS = os.getenv("WATSONX_CREDENTIALS")

# Optional safety check
if not CREDENTIALS:
    raise ValueError("Missing WATSONX_CREDENTIALS in .env file")

## Credentials, url, project_id, api_key etc are fixed on:
## https://cloud.ibm.com/login in your own account

# Model IDs
LLAMA_MODEL_ID = "meta-llama/llama-3-2-11b-vision-instruct"
GRANITE_MODEL_ID = "ibm/granite-4-h-small"
## OLD: ibm/granite-3-3-8b-instruct
MISTRAL_MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_ID = "gpt-4o-mini"  # or gpt-4o, gpt-4.1, gpt-3.5-turbo

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

# DeepSeek (OpenAI-compatible API)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL_ID = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

if not DEEPSEEK_API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY in .env file")