import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool
from langchain_classic.memory import ConversationBufferWindowMemory

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator

from enhanced_inventory_manager import get_inventory_manager, CarSearchResult
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Custom ChatOpenAI wrapper that removes the 'stop' parameter for Databricks compatibility
class DatabricksChatOpenAI(ChatOpenAI):
    """
    Custom ChatOpenAI wrapper that removes the 'stop' parameter before sending to Databricks.
    Databricks serving endpoints (and some newer OpenAI models) don't support the stop parameter.
    """
    
    def _stream(self, messages, stop=None, **kwargs):
        """Override _stream to ignore stop parameter"""
        # Call parent's _stream without the stop parameter
        if 'max_tokens' in kwargs:
             del kwargs['max_tokens']
        return super()._stream(messages, stop=None, **kwargs)
    
    def _generate(self, messages, stop=None, **kwargs):
        """Override _generate to simulate stop parameter client-side"""
        # Call parent's _generate without the stop parameter (API doesn't support it)
        if 'max_tokens' in kwargs:
             del kwargs['max_tokens']
        
        # We purposely don't pass 'stop' to the backend
        res = super()._generate(messages, stop=None, **kwargs)
        
        # Client-side stop sequence handling
        if stop and res.generations:
            text = res.generations[0].text
            min_index = len(text)
            found = False
            for stop_word in stop:
                idx = text.find(stop_word)
                if idx != -1 and idx < min_index:
                    min_index = idx
                    found = True
            
            if found:
                # Truncate text at the first stop word found
                res.generations[0].text = text[:min_index]
                # Also update the message content if it exists
                if hasattr(res.generations[0], 'message'):
                    res.generations[0].message.content = text[:min_index]
                    
        return res

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('carbot_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SalesStage(Enum):
    GREETING = "greeting"
    DISCOVERY = "discovery"
    PRESENTATION = "presentation"
    OBJECTION_HANDLING = "objection_handling"
    NEGOTIATION = "negotiation"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"

class AgentRole(Enum):
    CARLOS_SALES = "carlos_sales"
    MARIA_RESEARCH = "maria_research"
    MANAGER_COORDINATOR = "manager_coordinator"

@dataclass
class CustomerProfile:
    """Comprehensive customer profile"""
    name: Optional[str] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    preferred_make: Optional[str] = None
    preferred_color: Optional[str] = None
    body_style_preference: Optional[str] = None
    fuel_type_preference: Optional[str] = None
    family_size: Optional[int] = None
    primary_use: Optional[str] = None
    safety_priority: bool = False
    luxury_preference: bool = False
    eco_friendly: bool = False
    needs: List[str] = None
    objections: List[str] = None
    interaction_history: List[Dict] = None
    
    def __post_init__(self):
        if self.needs is None:
            self.needs = []
        if self.objections is None:
            self.objections = []
        if self.interaction_history is None:
            self.interaction_history = []

@dataclass
class AgentCommunication:
    """Inter-agent communication structure"""
    from_agent: AgentRole
    to_agent: AgentRole
    message_type: str
    content: str
    timestamp: datetime
    priority: str = "normal"
    requires_response: bool = False

class AgentState(TypedDict):
    """The state of the multi-agent workflow"""
    input: str
    chat_history: List[Any]
    customer_profile_summary: str
    internal_communications_summary: str
    customer_notes_summary: str
    sales_stage: str
    # Nodes fill these
    manager_output: Optional[str]
    maria_output: Optional[str]
    carlos_output: Optional[str]
    # To track current sender
    next_agent: str
    language: str

class AdvancedCarSalesSystem:
    """Advanced multi-agent car sales system with professional workflows"""
    
    def __init__(self, databricks_token: str, serpapi_api_key: str = None, 
                 databricks_base_url: str = None):
        self.databricks_token = databricks_token
        self.serpapi_api_key = serpapi_api_key
        
        # Default Databricks base URL if not provided
        if databricks_base_url is None:
            databricks_base_url = os.getenv('DATABRICKS_BASE_URL', '')
        self.databricks_base_url = databricks_base_url
        
        # Language settings
        self.language = "English"
        self.language_code = "en"
        
        # Initialize inventory manager
        self.inventory_manager = get_inventory_manager()
        
        # Initialize customer profile
        self.customer_profile = CustomerProfile()
        self.sales_stage = SalesStage.GREETING
        
        # Communication system
        self.agent_communications = []
        self.conversation_log = []
        self.carlos_customer_notes: List[str] = [] # Carlos's customer notes
        
        # Databricks models don't support the 'stop' parameter, use custom wrapper
        self.carlos_llm = DatabricksChatOpenAI(
            temperature=1,
            api_key=databricks_token,
            base_url=databricks_base_url,
            model_name="databricks-gpt-5-1",
            max_tokens=1000
        )
        
        self.maria_llm = DatabricksChatOpenAI(
            temperature=1,
            api_key=databricks_token,
            base_url=databricks_base_url,
            model_name="databricks-gpt-5",
            max_tokens=800
        )
        
        self.manager_llm = DatabricksChatOpenAI(
            temperature=1,
            api_key=databricks_token,
            base_url=databricks_base_url,
            model_name="databricks-gpt-5-mini",
            max_tokens=600
        )
        
        # Initialize memory for each agent
        self.carlos_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="input",
            k=10,
            return_messages=True
        )
        
        # Initialize tools and agents
        self.tools = self._create_advanced_tools()
        self.carlos_agent = self._create_carlos_agent()
        self.maria_agent = self._create_maria_agent()
        self.manager_agent = self._create_manager_agent()
        
        # Initialize LangGraph workflow
        self.workflow = self._create_langgraph_workflow()
        self.graph = self.workflow.compile()
        
        # Add initial system log and communication
        self._log_agent_action(AgentRole.MANAGER_COORDINATOR, "system_startup", "LangGraph-based Multi-Agent System Initialized")
        self._log_agent_communication(
            AgentRole.MANAGER_COORDINATOR, 
            AgentRole.CARLOS_SALES, 
            "directive", 
            "Welcome Carlos. We are open for business. Prioritize selling our local inventory first."
        )
        
        logger.info("🚀 LangGraph Car Sales System initialized successfully")

    def set_language(self, language_code: str) -> None:
        """Update the target language for the system"""
        languages = {
            "en": "English",
            "ja": "Japanese (日本語)",
            "zh-tw": "Traditional Chinese (繁體中文)",
            "es": "Spanish (Español)"
        }
        self.language_code = language_code
        self.language = languages.get(language_code, "English")
        logger.info(f"🌐 System language set to: {self.language}")

    def get_graph_image(self):
        """Returns the bytes of the mermaid graph image"""
        try:
            return self.graph.get_graph().draw_mermaid_png()
        except Exception as e:
            logger.error(f"Error generating graph image: {e}")
            return None

    def _create_langgraph_workflow(self) -> StateGraph:
        """Create the LangGraph multi-agent workflow"""
        workflow = StateGraph(AgentState)

        # Define the nodes
        workflow.add_node("eddy", self.eddy_node)
        workflow.add_node("maria", self.maria_node)
        workflow.add_node("carlos", self.carlos_node)

        # Build the edges
        workflow.set_entry_point("eddy")
        workflow.add_edge("eddy", "maria")
        workflow.add_edge("maria", "carlos")
        workflow.add_edge("carlos", END)

        return workflow

    def eddy_node(self, state: AgentState) -> Dict:
        """Eddy (Manager) handles inventory search and strategy"""
        user_input = state['input']
        language = state['language']
        
        # Localize log message
        log_msg = self._localize_manager_response(f"Analyzing: {user_input[:50]}...", language)
        self._log_agent_action(AgentRole.MANAGER_COORDINATOR, "analysis_start", log_msg)
        
        # Log communication to Carlos (simulated)
        comm_msg = self._localize_manager_response(f"Analyzing customer request: {user_input[:100]}", language)
        self._log_agent_communication(
            AgentRole.MANAGER_COORDINATOR,
            AgentRole.CARLOS_SALES,
            "instruction",
            comm_msg
        )
        
        manager_response = self._manager_decision_engine(user_input, language=language)
        
        # Log the response back
        self._log_agent_communication(
            AgentRole.MANAGER_COORDINATOR,
            AgentRole.CARLOS_SALES,
            "inventory_report",
            manager_response[:200] + "..."
        )
        
        return {
            "manager_output": manager_response,
            "next_agent": "maria"
        }

    def maria_node(self, state: AgentState) -> Dict:
        """Maria handles external research if necessary"""
        user_input = state['input']
        language = state['language']
        
        # Heuristic: if input contains specific car models but not "have", Maria researches
        if any(brand in user_input.lower() for brand in ['audi', 'bmw', 'mercedes', 'tesla', 'electric', 'safe']):
            log_msg = self._localize_manager_response(f"Researching: {user_input[:50]}...", language)
            self._log_agent_action(AgentRole.MARIA_RESEARCH, "research_start", log_msg)
            
            research_result = self._maria_research_engine(user_input)
            
            self._log_agent_communication(
                AgentRole.MARIA_RESEARCH,
                AgentRole.CARLOS_SALES,
                "research_findings",
                research_result[:200] + "..."
            )
            
            return {
                "maria_output": research_result,
                "next_agent": "carlos"
            }
        
        return {
            "maria_output": "No specific external research required for this query.",
            "next_agent": "carlos"
        }

    def carlos_node(self, state: AgentState) -> Dict:
        """Carlos generates the final response using everything provided"""
        user_input = state['input']
        
        # Construct the context for Carlos with very aggressive labeling
        internal_comms = f"""
### 🏢 OFFICIAL DEALERSHIP INVENTORY (Source: Manager) ###
[ONLY USE THIS FOR STOCK AVAILABILITY]
{state.get('manager_output', 'No local inventory found for this query.')}

### 🔬 EXTERNAL MARKET RESEARCH (Source: Maria) ###
[FOR GENERAL KNOWLEDGE ONLY - NOT IN STOCK]
{state.get('maria_output', 'No external research performed.')}

### SOURCE RULES FOR CARLOS:
- Check the MANAGER section for VINs/Price/Availability.
- Use the MARIA section for Specs/Reviews/Safety.
- NEVER mix them up.
"""
        language = state['language']
        
        log_msg = self._localize_manager_response("Synthesizing final message", language)
        self._log_agent_action(AgentRole.CARLOS_SALES, "response_generation", log_msg)
        
        response = self.carlos_agent.invoke({
            'input': user_input,
            'sales_stage': state['sales_stage'],
            'customer_profile_summary': state['customer_profile_summary'],
            'internal_communications_summary': internal_comms,
            'customer_notes_summary': state['customer_notes_summary'],
            'language': state['language']
        })
        
        carlos_text = response.get('output', 'I am here to help.')
        
        return {
            "carlos_output": carlos_text
        }
    
    def _perform_intelligent_inventory_search(self, query: str) -> str:
        """Helper method for intelligent inventory search, used by the Manager."""
        try:
            results_objects = self.inventory_manager.intelligent_search(query, max_results=8)
            formatted_results = self.inventory_manager.format_search_results_for_agent(results_objects, max_display=len(results_objects))
            
            logger.info(f"⚙️ System performed inventory search for query '{query}', found {len(results_objects)} vehicles.")
            return formatted_results
        except Exception as e:
            logger.error(f"❌ Error in _perform_intelligent_inventory_search: {e}")
            return "❌ Internal error while performing inventory search."
    
    def _create_advanced_tools(self) -> List[Tool]:
        """Create advanced tools for the multi-agent system"""
        tools = []
        
        # Manager consultation tool
        def consult_manager(request: str) -> str:
            """Consult with the manager for pricing, priorities, and business decisions"""
            logger.info(f"☎️ TOOL CALL: ConsultManager with request: '{request}'")
            # Clean input: if agent sends JSON string, extract just the request content
            clean_request = request
            if request.strip().startswith('{'):
                try:
                    data = json.loads(request)
                    if isinstance(data, dict):
                        clean_request = data.get('request', data.get('query', request))
                        logger.info(f"🧹 Cleaned JSON request from Carlos: {request} -> {clean_request}")
                except:
                    pass

            self._log_agent_communication(
                AgentRole.CARLOS_SALES,
                AgentRole.MANAGER_COORDINATOR,
                "consultation_request",
                clean_request
            )
            
            try:
                # Get state to access language
                target_lang = "English"
                if hasattr(self, 'language'):
                    target_lang = self.language
                    
                manager_response = self._manager_decision_engine(clean_request, language=target_lang)
                
                self._log_agent_communication(
                    AgentRole.MANAGER_COORDINATOR,
                    AgentRole.CARLOS_SALES,
                    "consultation_response",
                    manager_response
                )
                
                return manager_response
                
            except Exception as e:
                logger.error(f"❌ Error consulting manager: {e}")
                return "The manager is not available at this moment. Proceed with standard policies."
        
        tools.append(Tool(
            name="ConsultManager",
            func=consult_manager,
            description="**PRIMARY TOOL FOR INVENTORY**. Use this to search our dealership's inventory (data/enhanced_inventory.csv), get vehicle availability, pricing, VINs, and specific details about cars WE HAVE IN STOCK. Also for sales policies and discounts. DO NOT use ResearchVehicleInfo for inventory searches - use this tool instead."
        ))
        
        # Research tool via Maria
        def research_vehicle_info(query: str) -> str:
            """Research detailed vehicle information, reviews, and market data"""
            self._log_agent_communication(
                AgentRole.CARLOS_SALES,
                AgentRole.MARIA_RESEARCH,
                "research_request",
                query
            )
            
            try:
                research_result = self._maria_research_engine(query)
                
                self._log_agent_communication(
                    AgentRole.MARIA_RESEARCH,
                    AgentRole.CARLOS_SALES,
                    "research_response",
                    research_result
                )
                
                return research_result
                
            except Exception as e:
                logger.error(f"❌ Error in research: {e}")
                return "I could not obtain additional information at this time."
        
        tools.append(Tool(
            name="ResearchVehicleInfo",
            func=research_vehicle_info,
            description="Research EXTERNAL market information ONLY - general model reviews, competitor comparisons, safety ratings from NHTSA/IIHS. DO NOT use for our inventory - use ConsultManager for that."
        ))
        
        # Customer profiling tool
        def update_customer_profile(info: str) -> str:
            """Update customer profile with new information"""
            self._log_agent_action(AgentRole.CARLOS_SALES, "profile_update", info)
            
            try:
                self._update_customer_profile_from_text(info)
                profile_summary = self._get_customer_profile_summary()
                
                logger.info(f"📝 Customer profile updated: {profile_summary}")
                return f"Profile updated: {profile_summary}"
                
            except Exception as e:
                logger.error(f"❌ Error updating customer profile: {e}")
                return "Error updating the customer profile."
        
        tools.append(Tool(
            name="UpdateCustomerProfile",
            func=update_customer_profile,
            description="Update customer profile with preferences, needs, budget, and other relevant information"
        ))
        
        # Sales stage management
        def update_sales_stage(stage: str) -> str:
            """Update the current sales stage"""
            try:
                new_stage = SalesStage(stage.lower())
                old_stage = self.sales_stage
                self.sales_stage = new_stage
                
                self._log_agent_action(
                    AgentRole.CARLOS_SALES,
                    "stage_transition",
                    f"{old_stage.value} -> {new_stage.value}"
                )
                
                return f"Sales stage updated to: {new_stage.value}"
                
            except ValueError:
                return f"Invalid sales stage: {stage}"
        
        tools.append(Tool(
            name="UpdateSalesStage",
            func=update_sales_stage,
            description="Update the current sales stage (greeting, discovery, presentation, objection_handling, negotiation, closing)"
        ))

        # Tool to finalize sale and reserve vehicle
        def finalize_sale_and_reserve_vehicle(vin: str) -> str:
            """Finalizes the sale of a vehicle and marks it as reserved in the inventory."""
            self._log_agent_action(AgentRole.CARLOS_SALES, "finalize_sale_attempt", f"VIN: {vin}")
            try:
                success = self.inventory_manager.reserve_vehicle(vin)
                if success:
                    logger.info(f"🎉 Sale finalized and vehicle {vin} reserved by Carlos.")
                    return f"Excellent! The vehicle with VIN {vin} has been successfully reserved. The purchase process has concluded. Thank you!"
                else:
                    logger.warning(f"⚠️ Carlos failed to reserve vehicle {vin}. It might be already reserved or VIN is incorrect.")
                    return f"There was a problem trying to reserve vehicle {vin}. Please check the VIN or vehicle status. It might no longer be available."
            except Exception as e:
                logger.error(f"❌ Error during finalize_sale_and_reserve_vehicle tool: {e}", exc_info=True)
                return f"A technical error occurred while trying to reserve vehicle {vin}."

        tools.append(Tool(
            name="FinalizeSaleAndReserveVehicle",
            func=finalize_sale_and_reserve_vehicle,
            description="Use to finalize a sale and reserve a specific vehicle by its VIN when the customer agrees to purchase."
        ))

        # Tool to give final response to client
        def respond_to_client(response: str) -> str:
            """Delivers your message directly to the customer."""
            logger.info(f"🗣️ CARLOS TO CLIENT (via RespondToClient tool): {response[:100]}...")
            return response

        tools.append(Tool(
            name="RespondToClient",
            func=respond_to_client,
            description="Use this tool to provide your final answer or response directly to the customer."
        ))

        # Carlos's Customer Notes tool
        def update_customer_notes(note_to_add: str, mode: str = "append") -> str:
            """Adds or overwrites notes in Carlos's personal customer notepad."""
            self._log_agent_action(AgentRole.CARLOS_SALES, "update_customer_notes_attempt", f"Mode: {mode}, Note: {note_to_add[:50]}...")
            if mode.lower() == "overwrite":
                self.carlos_customer_notes = [note_to_add]
                logger.info(f"📝 Carlos's customer notes OVERWRITTEN.")
                return f"Notes overwritten. New note: '{note_to_add[:100]}...'"
            elif mode.lower() == "append":
                self.carlos_customer_notes.append(note_to_add)
                logger.info(f"📝 Carlos's customer note APPENDED.")
                return f"Note added: '{note_to_add[:100]}...'. Total notes: {len(self.carlos_customer_notes)}."
            else:
                return "Invalid mode. Use 'append' or 'overwrite'."

        tools.append(Tool(
            name="UpdateCustomerNotes",
            func=update_customer_notes,
            description="Manage your personal notes about the client. Useful for qualitative details. Modes: 'append', 'overwrite'."
        ))
        
        return tools
    
    def _create_carlos_agent(self) -> AgentExecutor:
        """Create Carlos - the expert sales agent"""
        
        carlos_prompt = PromptTemplate.from_template("""
You are Carlos, an expert car salesman with 15 years of experience.
Your MISSION is to find the perfect car for the customer from OUR inventory and close the sale.

CRITICAL: DO NOT RESPOND WITH JSON. DO NOT use keys like 'stage' or 'request' in a JSON block. 
You MUST use the THOUGHT / ACTION / OBSERVATION / FINAL ANSWER format.

PERSONALITY:
- Charismatic, professional, and trustworthy.
- Active listener; you lead the conversation.

SALES PROCESS:
1. GREETING -> 2. DISCOVERY -> 3. PRESENTATION -> 4. OBJECTION_HANDLING -> 5. NEGOTIATION -> 6. CLOSING

TOOLS:
{tools}

TOOL NAMES:
{tool_names}

KEY DIRECTIVES:
- `ConsultManager`: **YOUR PRIMARY TOOL**. Use it for ALL inventory searches.
  **CRITICAL**: If the customer asks for a car (e.g., BMW, Audi), you MUST use ConsultManager FIRST.
- `ResearchVehicleInfo`: ONLY for general model facts (safety ratings, reviews). NOT for our stock.

FORMAT TO FOLLOW:
You must strictly follow this format for EVERY step. KEEP IT SHORT.

Thought: [Brief reasoning (1 sentence max)]
Action: [The name of the tool to use, must be one of {tool_names}]
Action Input: [The input for the tool]
Observation: [The tool result will appear here]

... (Repeat if needed)

When you are ready to give your final answer:
Thought: I have finished my research and I'm ready to respond to the customer. I will clearly distinguish between our inventory and general market research.
Final Answer: [Your complete response. You MUST cite your internal sources where appropriate, e.g., "Eddy (🏢) confirmed we have..." or "Maria (🔬) researched that globally..."]

Current context:
- Sales stage: {sales_stage}
- Customer profile: {customer_profile_summary}
- Internal communications: {internal_communications_summary}
- Personal Notes: {customer_notes_summary}
- Target Language: {language}

Conversation history:
{chat_history}

Customer input: {input}

CRITICAL: You MUST write your final response and internal thoughts entirely in {language}.
If the customer speaks to you in another language, you must still respond in {language} as per dealership policy.

SOURCE ATTRIBUTION RULES:
1.  **Manager (🏢 Coordinator)**: The ONLY source for "What is currently in our lot/stock".
    - You MUST use phrases like "Our Manager confirms..." or "Looking at our inventory..." when discussing availability.
2.  **Maria (🔬 Researcher)**: The source for "General car facts, reviews, and market trends". 
    - Maria DOES NOT know what we have in stock. If Maria mentions a car year or price, it is GLOBAL MARKET data, NOT our inventory.
    - You MUST use phrases like "According to Maria's technical research..." or "Market data shows..." when discussing specifications.

TRUTH AND HONESTY:
- **ZERO TOLERANCE for mixing sources.** If the Manager says "0 matches", do NOT say "We have it" just because Maria mentioned it in a research report.
- Admitting we don't have a car is PROFESSIONAL. Hallucinating stock is FRAUD.
- If you cite Maria's info for a car we DON'T have, be clear: "Maria found that 2024 models have [Feature], but we currently only have the 2023 in our local stock."

{agent_scratchpad}
""")
        
        carlos_tools = self.tools
        carlos_agent_runnable = create_react_agent(
            llm=self.carlos_llm,
            tools=carlos_tools,
            prompt=carlos_prompt
        )
        
        return AgentExecutor(
            agent=carlos_agent_runnable,
            tools=carlos_tools,
            memory=self.carlos_memory,
            verbose=True,
            max_iterations=50,
            handle_parsing_errors=True
        )
    
    def _create_maria_agent(self) -> AgentExecutor:
        """Create Maria - the research specialist"""
        return None
    
    def _create_manager_agent(self) -> AgentExecutor:
        """Create Manager - the business coordinator"""
        return None
    
    def _handle_vin_request(self, request: str) -> str:
        """Handles requests for a vehicle's VIN."""
        logger.info(f"🏢 Manager received VIN request: {request}")
        match = re.search(r'(of|for)\s+([a-zA-Z0-9\s-]+)', request, re.IGNORECASE)
        if not match:
            return "I could not identify the vehicle to look up the VIN. Please be more specific, e.g., 'I need the VIN for the 2023 Toyota Camry'."

        vehicle_query = match.group(2).strip()
        vehicle_query = re.sub(r'\b(20\d{2})\b', '', vehicle_query).strip()

        search_results = self.inventory_manager.intelligent_search(vehicle_query, max_results=1)
        
        if search_results:
            vehicle = search_results[0]
            response = f"""
🏢 **MANAGER RESPONSE - VIN REQUEST:**

I found the vehicle matching your request '{vehicle_query}'.

- **Vehicle:** {vehicle.year} {vehicle.make} {vehicle.model}
- **VIN:** `{vehicle.vin}`

Use this VIN to proceed with the reservation.
"""
            return response.strip()
        else:
            return f"I didn't find a vehicle matching '{vehicle_query}' in the available inventory."

    def _localize_manager_response(self, text: str, target_language: str) -> str:
        """Helper to localize a manager's response to the target language"""
        if target_language.lower() == "english":
            return text
            
        prompt = f"""
        You are a translation assistant for a dealership manager. 
        Translate the following manager response into {target_language}.
        Keep the technical details, markdown formatting, and icons (🏢, 🎯, 📋, etc.) exactly as they are.
        The goal is a professional, localized version of the message.
        
        MESSAGE TO TRANSLATE:
        {text}
        
        LOCALIZED MESSAGE:
        """
        try:
            localized_text = self.manager_llm.invoke(prompt).content
            return localized_text.strip()
        except Exception as e:
            logger.error(f"❌ Error localizing manager response: {e}")
            return text # Fallback to English

    def _manager_decision_engine(self, request: str, language: str = "English") -> str:
        """Manager's decision-making engine for business policies"""
        logger.info(f"🏢 MANAGER CONSULTATION: '{request}' in {language}")
        
        # Use LLM to localize the manager's response if it's not a standard search
        # For simplicity in this demo, we'll add a instruction to the prompt context
        # but the decision logic below is mostly string based.
        
        request_lower = request.lower()
        
        if 'vin' in request_lower:
            return self._localize_manager_response(self._handle_vin_request(request), language)

        # Priority: Check for inventory search requests with broader keyword matching
        inventory_keywords = [
            "search", "find", "have", "stock", "inventory", "available", 
            "looking for", "need", "want", "show me", "vehicles", "cars", "options",
            "units", "models", "any", "which"
        ]
        
        # Check if this is an inventory-related request
        is_inventory_request = any(keyword in request_lower for keyword in inventory_keywords)
        
        # If they ask for "price" or "cost", check if they also mentioned a specific car or make
        if 'price' in request_lower or 'cost' in request_lower:
            car_makes = ['audi', 'bmw', 'mercedes', 'toyota', 'honda', 'ford', 'volkswagen', 'nissan', 'hyundai', 'kia', 'mazda', 'subaru']
            if any(make in request_lower for make in car_makes) or any(keyword in request_lower for keyword in ['car', 'vehicle', 'unit', 'model']):
                is_inventory_request = True
            else:
                # If just asking about "price" in general, it might be a policy request
                # We'll let it fall through to the pricing policy block if not explicitly inventory
                is_inventory_request = is_inventory_request 

        if is_inventory_request:
            logger.info(f"🔍 Manager detected inventory search request: {request}")
            search_results_objects = self.inventory_manager.intelligent_search(request, max_results=8)
            
            if not search_results_objects:
                # If no matches found in local inventory, explicitly tell Carlos
                return f"🏢 **MANAGER RESPONSE - INVENTORY SEARCH:**\n\n❌ **STRENGTHENED WARNING:** I checked our official local dealership records (data/enhanced_inventory.csv) and we currently have **ZERO** vehicles matching: '{request}'.\n\n**CARLOS:** Just because Maria (the researcher) might mention this car in her report, it does NOT mean it's in our inventory. You are STRICTLY FORBBIDDEN from telling the customer we have this specific year/model in stock. You must honestly state it's unavailable locally and offer the closest available alternative from the Manager's reports."

            formatted_search_results = self.inventory_manager.format_search_results_for_agent(search_results_objects, max_display=len(search_results_objects))

            directives_list = []
            if search_results_objects:
                priority_1 = search_results_objects[0]
                directives_list.append(f"1. **High Priority:** Actively present the **{priority_1.year} {priority_1.make} {priority_1.model} (VIN: {priority_1.vin})**.")
                
            directives = "\n".join(directives_list)
            
            response = f"""
🏢 **MANAGER RESPONSE - STRATEGIC INVENTORY SEARCH:**

Carlos, I've processed your request: '{request}'.

Found Vehicles in OUR LOCAL INVENTORY (data/enhanced_inventory.csv):
{formatted_search_results}

🎯 **SALES DIRECTIVE (Prioritize these options):**
{directives if directives else "No specific directives, use your judgment."}

**Note on Pricing:** If they asked for pricing, the prices listed above ARE our current internet prices. You are authorized for up to a 10% discount if necessary to close.
"""
            return self._localize_manager_response(response.strip(), language)
        
        if any(word in request_lower for word in ['price', 'discount', 'offer']):
            return self._localize_manager_response(self._handle_pricing_request(request), language)
        
        elif any(word in request_lower for word in ['priority', 'recommend', 'inventory']):
            return self._localize_manager_response(self._handle_inventory_priority_request(request), language)
        
        elif any(word in request_lower for word in ['policy', 'rule', 'procedure']):
            return self._localize_manager_response(self._handle_policy_request(request), language)
        
        else:
            return self._localize_manager_response(self._handle_general_consultation(request), language)
    
    def _handle_pricing_request(self, request: str) -> str:
        """Handle pricing and discount requests contextually."""
        response = """
🏢 **MANAGER DECISION - PRICING POLICY:**

📋 **Discount Authorization:**
- Standard authorized discount: up to 10%
- For larger discounts (10-15%): requires solid justification.
- Premium vehicles: maximum 5% discount.

💰 **Strategy:** Focus on value before discussing price.
"""
        return response.strip()
    
    def _handle_inventory_priority_request(self, request: str) -> str:
        """Handle inventory priority and recommendation requests"""
        stats = self.inventory_manager.get_inventory_stats()
        response = f"""
🏢 **MANAGER DECISION - INVENTORY PRIORITIES:**

Total vehicles: {stats.get('total_vehicles', 'N/A')}
Focus on high-margin vehicles (BMW, Mercedes, Audi) and inventory older than 4 months.
"""
        return response.strip()
    
    def _handle_policy_request(self, request: str) -> str:
        """Handle policy and procedure questions"""
        return "🏢 **COMPANY POLICIES:** 1-year minimum warranty, transparency in pricing, and mandatory pre-delivery inspection."
    
    def _handle_general_consultation(self, request: str) -> str:
        """Handle general business consultations"""
        return "🏢 **MANAGER CONSULTATION:** Focus on customer needs, build value, and document all interactions."
    
    def _maria_research_engine(self, query: str) -> str:
        """Maria's research engine for vehicle information"""
        logger.info(f"🔬 MARIA RESEARCH REQUEST: {query}")
        
        raw_search_snippets = ""
        source_type = "Internal Knowledge Base"
        
        if self.serpapi_api_key:
            try:
                search_wrapper = SerpAPIWrapper(serpapi_api_key=self.serpapi_api_key)
                raw_search_snippets = search_wrapper.run(f"car review {query} specifications safety reliability")
                source_type = "Web Search (SerpAPI)"
            except Exception as e:
                raw_search_snippets = self._knowledge_based_research(query, internal_call=True)

        maria_analyzer_prompt_text = (
            "You are Maria, an expert and analytical car researcher. Carlos, a salesman, has asked you:\n"
            "QUERY: \"{carlos_query}\"\n\n"
            "Raw info from {source_type}:\n"
            "\"{snippets}\"\n\n"
            "Provide a concise report starting with '🔬 **MARIA'S DETAILED ANALYSIS:**'. Focus on safety, reliability, and key features.\n"
            "CRITICAL: You MUST write your analysis entirely in {language}."
        )
        
        maria_analyzer_prompt_template = PromptTemplate.from_template(maria_analyzer_prompt_text)
        analyzer_prompt = maria_analyzer_prompt_template.format(
            carlos_query=query,
            snippets=raw_search_snippets[:2000],
            source_type=source_type,
            language=self.language
        )

        analytical_report = self.maria_llm.invoke(analyzer_prompt).content
        return f"🔬 **MARIA'S RESEARCH REPORT:**\n\n{analytical_report}"

    def _knowledge_based_research(self, query: str, internal_call: bool = False) -> str:
        """Fallback knowledge-based research."""
        return "Generic knowledge about vehicle safety and efficiency."
    
    def _update_customer_profile_from_text(self, text: str) -> None:
        """Update customer profile from conversation text"""
        text_lower = text.lower()
        
        # Simple extraction logic for budget and family needs
        if 'budget' in text_lower:
            self.customer_profile.budget_max = 50000 # Placeholder
        if any(word in text_lower for word in ['family', 'kids']):
            self.customer_profile.safety_priority = True
        
        self.customer_profile.interaction_history.append({
            'timestamp': datetime.now(),
            'content': text
        })
        
    def _extract_customer_interest(self, text: str, language: str = "English") -> Optional[str]:
        """Use LLM to extract specific customer interests from their message"""
        prompt = f"""
        Analyze the following customer message to a car dealership and extract their core interests (models, features, budget, lifestyle needs).
        Provide a single, concise bullet point (max 15 words).
        You MUST write the response in {language}.
        If no specific interest is found, return 'NONE'.
        
        MESSAGE: "{text}"
        
        INTEREST:
        """
        try:
            response = self.manager_llm.invoke(prompt).content.strip()
            if response.upper() == 'NONE' or len(response) < 3:
                return None
            return response
        except Exception as e:
            logger.error(f"❌ Error extracting interests: {e}")
            return None
    
    def _get_customer_profile_summary(self) -> str:
        """Get a summary of the customer profile"""
        profile = self.customer_profile
        summary = f"Budget: {profile.budget_max if profile.budget_max else 'Unknown'}; Safety: {profile.safety_priority}"
        return summary
    
    def _log_agent_action(self, agent: AgentRole, action: str, details: str) -> None:
        """Log agent actions for debugging and analysis"""
        log_entry = {
            'timestamp': datetime.now(),
            'agent': agent.value,
            'action': action,
            'details': details
        }
        self.conversation_log.append(log_entry)
        logger.info(f"🤖 {agent.value.upper()}: {action} - {details[:100]}...")
    
    def _log_agent_communication(self, from_agent: AgentRole, to_agent: AgentRole, 
                                message_type: str, content: str) -> None:
        """Log inter-agent communications"""
        communication = AgentCommunication(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            timestamp=datetime.now()
        )
        self.agent_communications.append(communication)
        logger.info(f"📡 {from_agent.value} -> {to_agent.value}: {message_type}")
    
    def process_customer_input(self, user_input: str) -> str:
        """Main method to process customer input through the LangGraph workflow"""
        try:
            self._log_agent_action(AgentRole.CARLOS_SALES, "input_received", user_input)
            self._update_customer_profile_from_text(user_input)
            
            # Extract and update Carlos's personal notes automatically
            extracted_interest = self._extract_customer_interest(user_input, language=self.language)
            if extracted_interest:
                self.carlos_customer_notes.append(extracted_interest)
                self._log_agent_action(AgentRole.CARLOS_SALES, "note_auto_tag", extracted_interest)
            
            logger.info(f"🕸️ Running LangGraph workflow for: {user_input}")
            
            # Prepare initial state
            initial_state: AgentState = {
                "input": user_input,
                "chat_history": [], # We could pass chat history here if needed
                "customer_profile_summary": self._get_customer_profile_summary(),
                "internal_communications_summary": self._get_recent_communications_summary(),
                "customer_notes_summary": self._get_customer_notes_summary(),
                "sales_stage": self.sales_stage.value,
                "manager_output": None,
                "maria_output": None,
                "carlos_output": None,
                "next_agent": "eddy",
                "language": self.language
            }
            
            # Run the graph with a 1-minute timeout
            start_time = time.time()
            timeout = 60 # 1 minute
            final_state = initial_state.copy()
            
            try:
                # Use stream to catch updates from each node
                for output in self.graph.stream(initial_state):
                    # Update our tracking state with the output from each node
                    for node_name, node_output in output.items():
                        if isinstance(node_output, dict):
                            final_state.update(node_output)
                    
                    # Check for timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        logger.warning(f"⏱️ Workflow timeout after {elapsed:.2f}s. Preparing partial response.")
                        break
            except Exception as stream_err:
                logger.error(f"⚠️ Error during graph streaming: {stream_err}")
                # We continue with whatever state we have
            
            # If Carlos already finished, return his output
            if final_state.get('carlos_output'):
                return final_state['carlos_output']
            
            # If we timed out or Carlos didn't finish, generate a fallback from current info
            logger.info("📡 Carlos has not finished. Generating fallback response from current state.")
            return self._generate_timeout_fallback(final_state)
            
        except Exception as e:
            logger.error(f"❌ Error in LangGraph workflow: {e}", exc_info=True)
            return "I am having some technical difficulties with my coordination system. Could you please rephrase?"

    def _get_recent_communications_summary(self) -> str:
        """Get a summary of recent inter-agent communications for Carlos's context"""
        if not self.agent_communications:
            return "No recent internal communications."
        
        summary_parts = []
        for comm in self.agent_communications[-5:]:
            time_str = comm.timestamp.strftime("%H:%M:%S")
            summary_parts.append(f"[{time_str}] {comm.from_agent.value} -> {comm.to_agent.value} ({comm.message_type}): {comm.content[:200]}...")
            
        return "\n".join(summary_parts)

    def _get_customer_notes_summary(self) -> str:
        if not self.carlos_customer_notes:
            return "No personal notes yet."
        return "\n".join([f"Note {i+1}: {note}" for i, note in enumerate(self.carlos_customer_notes)])

    def _calculate_profile_completeness(self) -> float:
        return 50.0 # Placeholder

    def _generate_timeout_fallback(self, state: AgentState) -> str:
        """Generate a quick response when the full workflow times out"""
        manager_info = state.get('manager_output', 'Inventory check in progress...')
        maria_info = state.get('maria_output', 'Technical research in progress...')
        user_input = state.get('input', '')
        language = state.get('language', 'English')
        
        fallback_prompt = f"""
        You are Carlos, the car salesman. You are taking a bit longer than usual to coordinate with the Manager and Maria.
        However, you want to give a professional update to the customer RIGHT NOW based on what you know so far.
        
        CUSTOMER REQUEST: "{user_input}"
        
        CURRENT INTERNAL INFORMATION:
        - MANAGER INFO (Inventory): {str(manager_info)[:500]}
        - MARIA INFO (Research): {str(maria_info)[:500]}
        
        TASK:
        Write a professional, brief response to the customer in {language}.
        1. Acknowledge that you are still gathering some finer details.
        2. Provide the key information you have (especially if the Manager found specific cars).
        3. Keep it helpful and sales-oriented.
        4. Do NOT say "it timed out" or use technical jargon.
        
        RESPONSE:
        """
        
        try:
            # Use the LLM directly for a quick synthesis
            response = self.carlos_llm.invoke([HumanMessage(content=fallback_prompt)])
            return response.content
        except Exception as e:
            logger.error(f"❌ Fallback generation failed: {e}")
            if language == "Spanish (Español)":
                return "Estoy procesando su solicitud con nuestro equipo. De momento, estoy revisando nuestro inventario y datos técnicos. ¡Le daré una respuesta completa en un momento!"
            return "I'm currently coordinating with my team to get you the most accurate information. I'm reviewing our current inventory and technical specs right now and will have a detailed update for you in just a moment. Thank you for your patience!"

def get_advanced_multi_agent_system(databricks_token: str, serpapi_api_key: str = None,
                                     databricks_base_url: str = None) -> AdvancedCarSalesSystem:
    return AdvancedCarSalesSystem(databricks_token, serpapi_api_key, databricks_base_url)