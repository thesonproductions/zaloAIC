import os

from dotenv import load_dotenv
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.models.lite_llm import LiteLlm

# from google.adk.tools import tool
from google.adk.sessions import InMemorySessionService, Session  # noqa: F401
from google.adk.tools.agent_tool import AgentTool
from prompt import ROOT_SYSTEM_INSTRUCTION
from zaic_sub_agents import EmailAgent
from zaic_adk import SampleRuntime, create_sample_runtime

load_dotenv()
APP_NAME = os.getenv("APP_NAME", "lifemate")

# Configuration from env or defaults
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://chatbot_ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "ollama_chat/smollm2:135m")

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

agent_tools = [
    AgentTool(agent=EmailAgent),
]

llm = LiteLlm(
    model=MODEL_NAME,
    api_base=OLLAMA_HOST,
)

root_agent = LlmAgent(
    name="LifeMate",
    model=llm,
    instruction=ROOT_SYSTEM_INSTRUCTION,
    tools=agent_tools
)

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)


async def create_zaic_runtime(
    *,
    split: str,
    sample_id: str,
    data_root: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> SampleRuntime:
    """Create an ADK runner bound to one sample's official zaic_tools session."""
    return await create_sample_runtime(
        split=split,
        sample_id=sample_id,
        data_root=data_root,
        user_id=user_id,
        session_id=session_id,
    )