from typing import Optional

from dotenv import load_dotenv
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai.types import Content, Part

load_dotenv()

def loading_env_context():

    """
        To do: This func will be used for root agent to know the current context and information
    """

def logging_called_tools(callback_context: CallbackContext, llm_request: LlmRequest):

    
    return None

