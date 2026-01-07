import langchain

import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
from dotenv import load_dotenv

# Load environment variables

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(project_root, ".env"))

# Import systems
try:
    from advanced_multi_agent_system import get_advanced_multi_agent_system
    from enhanced_inventory_manager import get_inventory_manager
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="CarBot Pro - AI Car Salesman", 
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sales-metric {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1e3c72;
    }
    .car-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .agent-status {
        background: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #28a745;
    }
    .agent-communication {
        background: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #ffc107;
        margin: 0.2rem 0;
        font-size: 0.8rem;
        color: #503e00; /* Darker text for yellow background */
    }
    .log-entry {
        background: #f8f9fa;
        padding: 0.3rem;
        border-radius: 3px;
        margin: 0.1rem 0;
        font-family: monospace;
        font-size: 0.7rem;
        color: #212529; /* Darker text for light gray background */
    }
    .comm-card {
        background: #fff8e1;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #ffc107;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .comm-header {
        font-weight: bold;
        color: #856404;
        font-size: 0.85rem;
        margin-bottom: 5px;
        display: flex;
        justify-content: space-between;
    }
    .comm-content {
        font-size: 0.8rem;
        color: #503e00;
        line-height: 1.4;
    }
    .agent-tag {
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        text-transform: uppercase;
        font-weight: bold;
    }
    .tag-carlos { background: #e3f2fd; color: #1976d2; }
    .tag-maria { background: #fce4ec; color: #c2185b; }
    .tag-manager { background: #e8f5e9; color: #388e3c; }
    .customer-profile {
        background: #e7f3ff; /* Light blue background */
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        color: #002b5c; /* Dark blue text for good contrast */
    }
    .customer-profile p,
    .customer-profile li {
        color: #002b5c; /* Ensure p and li elements also inherit/use dark color */
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🚗 CarBot Pro - Advanced Multi-Agent System</h1>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🔧 System Configuration")
    
    # Agent Type Selection
    st.markdown("**🕸️ LangGraph Orchestrated Multi-Agent System**")
    st.markdown("- **Eddy Node** (databricks-gpt-5-mini): Analyzes inventory and set strategy")
    st.markdown("- **Maria Node** (databricks-gpt-5): Conducts technical research")
    st.markdown("- **Carlos Node** (databricks-gpt-5-1): Expert salesperson synthesizing all inputs")
    
    st.markdown("---")
    
    # API Keys
    st.subheader("🔑 API Keys")
    
    # Try to get from environment first
    default_databricks_token = os.getenv('DATABRICKS_TOKEN', '')
    default_databricks_base_url = os.getenv('DATABRICKS_BASE_URL', '')
    default_serpapi_key = os.getenv('SERPAPI_API_KEY', '')
    
    databricks_token = st.text_input(
        "Databricks Token", 
        value=default_databricks_token,
        type="password", 
        placeholder="dapi...",
        help="Required for Databricks models (databricks-gpt-5-1 and databricks-gpt-5)"
    )
    databricks_base_url = st.text_input(
        "Databricks Base URL",
        value=default_databricks_base_url,
        type="password", 
        placeholder="https://...",
        help="Databricks serving endpoint URL"
    )
    serpapi_api_key = st.text_input(
        "SerpAPI Key", 
        value=default_serpapi_key,
        type="password", 
        placeholder="Optional for web search",
        help="Optional: allows Maria to do real-time web research"
    )
    
    # Initialize Agent Button
    if st.button("🚀 Initialize Advanced System", type="primary"):
        if databricks_token:
            with st.spinner("Initializing advanced multi-agent system..."):
                try:
                    st.session_state.agent_system = get_advanced_multi_agent_system(
                        databricks_token, serpapi_api_key, databricks_base_url
                    )
                    st.session_state.agent_type = "advanced_multiagent"
                    st.session_state.system_initialized = True
                    
                    st.success("✅ Advanced system initialized successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Initialization error: {e}")
                    st.session_state.system_initialized = False
        else:
            st.error("❌ Databricks Token is required")
    
    st.markdown("---")
    
    # System Status
    st.subheader("📊 System Status")
    if st.session_state.get('system_initialized', False):
        st.markdown('<div class="agent-status">🟢 System Operational</div>', unsafe_allow_html=True)
        st.write(f"**Type:** {st.session_state.get('agent_type', 'Unknown')}")
        
        st.write("**Active Agents:**")
        st.write("- 🎯 Carlos (databricks-gpt-5-1 - Sales)")
        st.write("- 🔍 Maria (databricks-gpt-5 - Research)")
        st.write("- 👔 Edu (databricks-gpt-5-1 - Coordination)")

        # Display LangGraph visualization
        if hasattr(st.session_state.agent_system, 'get_graph_image'):
            with st.expander("🕸️ View LangGraph Workflow", expanded=False):
                graph_img = st.session_state.agent_system.get_graph_image()
                if graph_img:
                    st.image(graph_img, use_container_width=True)
                else:
                    st.info("Graph image not available (requires mermaid dependencies)")
        
        # Show system analytics if available
        if hasattr(st.session_state.agent_system, 'get_conversation_analytics'):
            analytics = st.session_state.agent_system.get_conversation_analytics()
            st.write("**Statistics:**")
            st.write(f"- Interactions: {analytics.get('total_interactions', 0)}")
            st.write(f"- Inter-agent communications: {analytics.get('agent_communications', 0)}")
            st.write(f"- Sales stage: {analytics.get('current_sales_stage', 'N/A')}")
            st.write(f"- Profile completed: {analytics.get('customer_profile_completeness', 0):.1f}%")
    else:
        st.warning("⚠️ System not initialized")
    
    st.markdown("---")
    
    # Debug Mode Toggle
    st.subheader("🔧 System Visibility")
    debug_mode = st.checkbox("Show detailed system logs", value=True)
    show_agent_comms = st.checkbox("Show inter-agent communications", value=True)
    
    st.markdown("---")
    
    # Language Selection
    st.subheader("🌐 Language / 語言 / 言語")
    languages = {
        "en": "English",
        "ja": "Japanese (日本語)",
        "zh-tw": "Traditional Chinese (繁體中文)",
        "es": "Spanish (Español)"
    }
    selected_lang_code = st.selectbox(
        "Select Language",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=0, # Default to English
        key="target_language"
    )
    st.session_state.selected_language = languages[selected_lang_code]

# Initialize demo_concluded state if not present
if 'demo_concluded' not in st.session_state:
    st.session_state.demo_concluded = False

# --- Main Content ---
if not st.session_state.get('system_initialized', False):
    # Welcome Screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ##  Welcome to Advanced CarBot Pro
        
        ### What makes this system special?
        
        **Multi-Agent Orchestrated Architecture:**
        - Eddy - Inventory Manager: analyzes your query against our inventory (databricks-gpt-5-mini)
        - Maria - Technical expert: research specifications and market data (databricks-gpt-5)
        - Carlos - Sales lead: synthesizes all information into a personal response (databricks-gpt-5-1)
        
        **🔧 Advanced Capabilities:**
        - ✅ Structured graph-based agent coordination
        - ✅ Intelligent search in enhanced inventory
        - ✅ Real-time web research
        - ✅ Automatic customer profiling
        - ✅ Full observability of inter-agent reasoning
        - ✅ Detailed logs and analytics
        
        **📈 Professional Sales Flow:**
        1. **Greeting and Rapport** - Carlos builds trust
        2. **Discovery** - Identifies customer needs
        3. **Consult with Edu** - Gets inventory priorities
        4. **Presentation** - Shows relevant vehicles
        5. **Research** - Maria provides technical data
        6. **Negotiation** - Edu authorizes discounts
        7. **Closing** - Professional finalization
        
        **🎯 Included Demo Script:**
        - Realistic sales scenarios
        - Family use cases
        - Objection handling
        - Price negotiation
        
        👈 **Configure API keys in the sidebar to begin**
        """)

else:
    # Main Application Layout
    col1, col2 = st.columns([2, 1]) # Define columns for the main layout

    with col1: # Chat Area
        st.markdown(
        """
        <style>
        .stImage {
            margin-left: 150px;
        }
        </style>
        """,
        unsafe_allow_html=True
        )
        st.image(os.path.join(project_root, "picture/landing_page.png"), width=500)
        st.subheader("💬 Chat with CarBot Pro")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            welcome_msg = """Hello! I'm **Carlos**, your personal AI-powered car salesperson. 
I have 15 years of experience helping families find the perfect vehicle. I work together with **Maria** (our research specialist) and **Eddy** (our Manager) to offer you the best service.
How can I help you today? Are you looking for something specific, or would you like me to recommend options based on your needs?
💡 *Tip: You can tell me things like "I'm looking for a safe car for my family" or "I need a red sedan less than 2 years old"*"""
            st.session_state.messages.append({
                "role": "assistant", "content": welcome_msg,
                "timestamp": datetime.now(), "agent": "Carlos"
            })

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if not st.session_state.get('demo_concluded', False):
            if user_input := st.chat_input("What are you looking for today?", key="customer_chat_input_active"):
                st.session_state.messages.append({
                    "role": "user", "content": user_input,
                    "timestamp": datetime.now(), "agent": "Customer"
                })
                
                with st.spinner("Carlos is thinking..."):
                    if hasattr(st.session_state, 'agent_system') and st.session_state.agent_system:
                        try:
                            # Pass selected language to the system
                            if hasattr(st.session_state.agent_system, 'set_language'):
                                st.session_state.agent_system.set_language(st.session_state.target_language)
                                
                            carlos_response = st.session_state.agent_system.process_customer_input(user_input)
                            st.session_state.messages.append({
                                "role": "assistant", "content": carlos_response,
                                "timestamp": datetime.now(), "agent": "Carlos"
                            })
                            # if "ha sido reservado exitosamente" in carlos_response or \
                            #    "proceso de compra ha concluido" in carlos_response:
                            #     st.session_state.demo_concluded = True
                            #     if hasattr(st.session_state.agent_system, 'inventory_manager'):
                            #         st.session_state.agent_system.inventory_manager.load_inventory()
                            #     st.rerun() 

                            # Check if the reservation was successful or the purchase process concluded
                            success_phrases = ["has been successfully reserved", "purchase process has concluded"]

                            if any(phrase in carlos_response.lower() for phrase in success_phrases):
                                st.session_state.demo_concluded = True
                                
                                # Reload inventory if the manager is present in the agent system
                                if hasattr(st.session_state.agent_system, 'inventory_manager'):
                                    st.session_state.agent_system.inventory_manager.load_inventory()
                                
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error processing input: {e}")
                            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an internal error occurred: {e}"})
                        if not st.session_state.get('demo_concluded', False): # Rerun if not concluded by this response
                           st.rerun()
                    else:
                        st.error("The agent system is not initialized.")
        
        elif st.session_state.get('demo_concluded', False): # Chat input area when demo is concluded
            st.info("Chat disabled. Demo has concluded.")

        # Sale conclusion message and Restart button (still within col1, below chat area)
        if st.session_state.get('demo_concluded', False):
            st.success("🎉 Sale Completed! Vehicle has been reserved.")
            st.info("For a new demo, please restart.")
            if st.button("🔁 Restart Demo", key="restart_demo_button_col1"):
                keys_to_reset = ['messages', 'agent_system', 'system_initialized', 'demo_concluded', 'agent_type']
                for key in keys_to_reset:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    # Information Panel (col2) - should always render if system is initialized
    if st.session_state.get('system_initialized', False):
        with col2:

            # Display Carlos's Customer Notes
            if st.session_state.get('system_initialized', False) and hasattr(st.session_state.agent_system, 'carlos_customer_notes'):
                notes = st.session_state.agent_system.carlos_customer_notes
                with st.expander("📝 Carlos's Customer Notes", expanded=False):
                    if notes:
                        st.markdown('<div class="agent-communication" style="border-left-color: #6f42c1; background-color: #f3e8ff; color: #3d236b;">', unsafe_allow_html=True) # Purple-ish theme
                        for i, note in enumerate(notes, 1):
                            st.markdown(f"**Note {i}:** {note}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("Carlos hasn't taken personal notes about this customer yet.")
            
            # Display Recent Inventory (Simplified)
            if st.session_state.get('system_initialized', False):
                st.subheader("📊 Vehicle Inventory")
                if hasattr(st.session_state, 'agent_system') and st.session_state.agent_system and hasattr(st.session_state.agent_system, 'inventory_manager'):
                    inventory_df_display = st.session_state.agent_system.inventory_manager.inventory_df.copy()
                    if 'status' not in inventory_df_display.columns:
                         inventory_df_display['status'] = 'Available'
                    
                    def highlight_status(row):
                        if row['status'] == 'Reserved':
                            return ['background-color: lightcoral'] * len(row)
                        return [''] * len(row)

                    display_columns = ['make', 'model', 'year', 'price', 'mileage', 'status', 'vin']
                    display_columns = [col for col in display_columns if col in inventory_df_display.columns]
                    
                    if display_columns:
                        st.dataframe(
                            inventory_df_display[display_columns].style.apply(highlight_status, axis=1), 
                            height=300, width='stretch'
                        )
                    else:
                        st.warning("Inventory columns not found.")
                else:
                    st.info("Inventory not available.")

            # Display Recent Inter-Agent Communications
            st.subheader("📡 Recent Inter-Agent Communications")
            if hasattr(st.session_state, 'agent_system') and st.session_state.agent_system:
                comms = st.session_state.agent_system.agent_communications
                if comms:
                    # Show last 8 communications
                    for comm in reversed(comms[-8:]):
                        from_tag = f"tag-{comm.from_agent.value.split('_')[0]}"
                        to_tag = f"tag-{comm.to_agent.value.split('_')[0]}"
                        
                        st.markdown(f"""
                        <div class="comm-card">
                            <div class="comm-header">
                                <span>
                                    <span class="agent-tag {from_tag}">{comm.from_agent.value.upper()}</span> 
                                    ➡️ 
                                    <span class="agent-tag {to_tag}">{comm.to_agent.value.upper()}</span>
                                </span>
                                <span style="color: #999;">{comm.timestamp.strftime('%H:%M:%S')}</span>
                            </div>
                            <div style="font-size: 0.75rem; font-weight: bold; margin-bottom: 5px; color: #666;">
                                Type: {comm.message_type}
                            </div>
                            <div class="comm-content">{comm.content}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No inter-agent communications yet.")
            else:
                st.info("System not initialized.")

            if debug_mode:
                st.markdown("---")

            if debug_mode:
                st.subheader("⚙️ System Log (Recent Actions)")
                if hasattr(st.session_state, 'agent_system') and \
                   st.session_state.agent_system and st.session_state.agent_system.conversation_log:
                    if st.session_state.agent_system.conversation_log:
                        for log in reversed(st.session_state.agent_system.conversation_log[-15:]):
                            log_content = f"{log['timestamp'].strftime('%H:%M:%S')} | {log['agent']} | {log['action']} | {str(log['details'])[:100]}"
                            st.markdown(f"<div class='log-entry'>{log_content}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No system logs.")
                else:
                    st.info("System logs not available.")

