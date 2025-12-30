import os
import sys
import logging
# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from advanced_multi_agent_system import DatabricksChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

token = os.getenv('DATABRICKS_TOKEN')
base_url = os.getenv('DATABRICKS_BASE_URL', '')

llm = DatabricksChatOpenAI(
    temperature=0,
    api_key=token,
    base_url=base_url,
    model_name="databricks-gpt-5-1"
)

print("--- Testing databricks-gpt-5-1 ---")
res = llm.invoke([HumanMessage(content="Hello, who are you?")])
print(f"Response: {res.content}")

print("\n--- Testing with 'car salesman' context ---")
res = llm.invoke([HumanMessage(content="You are a car salesman. Show me a BMW X3.")])
print(f"Response: {res.content}")
