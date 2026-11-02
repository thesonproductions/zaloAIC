import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent

# from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.lite_llm import LiteLlm

# from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
# from google.adk.agents.context_cache_config import ContextCacheConfig
from google.genai.types import GenerateContentConfig
from prompt import EMAIL_SYSTEM_INSTRUCTION
from zaic_tools.apps.email import Email

load_dotenv()
APP_NAME = os.getenv("APP_NAME")

# Configuration from env or defaults
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://chatbot_ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "ollama_chat/smollm2:135m")

tools_list = [
    Email.list,
    Email.read,
    Email.send,
    Email.create_draft,
    Email.reply,
    Email.archive
]

llm = LiteLlm(
    model=MODEL_NAME,
    api_base=OLLAMA_HOST,
)

EmailAgent = LlmAgent(
    name='EmailingAgent',
    description='Performing email-related tasks for user',
    model=llm,
    instruction=EMAIL_SYSTEM_INSTRUCTION,
    tools=tools_list
    # before_agent_callback=,
    # before_model_callback=,
)