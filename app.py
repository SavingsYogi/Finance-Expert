import streamlit as st
from agno.agent import Agent
from agno.models.azure import AzureOpenAI
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Set Streamlit page configuration
st.set_page_config(
    page_title="📊 Investment Analyst",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better formatting
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .response-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        padding: 8px 12px !important;
        border: 1px solid #ddd !important;
    }
    th {
        background-color: #f2f2f2;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the agent


@st.cache_resource
def initialize_agent():
    return Agent(
        model=AzureOpenAI(
            id="gpt-4o-mini",
            api_key=api_key,
            azure_endpoint=azure_endpoint
        ),
        tools=[
            DuckDuckGoTools(),
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True
            )
        ],
        markdown=True,
        show_tool_calls=True,
        description="You are a friendly and knowledgeable **Finance Expert** who teaches financial concepts to beginners and intermediate learners.",
        instructions=[
            "Explain financial concepts clearly and concisely using simple language.",
            "Use **markdown formatting** to organize the response — include headings, bullet points, tables, and bold for key terms.",
            "Where applicable, provide **real-world examples** (e.g., for compound interest, P/E ratio).",
            '''Use duckduckgo tools to go through the below links  :
    CNBC": "https://www.cnbc.com/personal-finance/",
    "Forbes Money": "https://www.forbes.com/money/",
    "Investopedia": "https://www.investopedia.com/financial-literacy-5214701",
    "Economic Times": "https://economictimes.indiatimes.com/wealth/",
    "MoneyControl": "https://www.moneycontrol.com/news/business/personal-finance/",
    "RBI Publications": "https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
    "SEBI Investor Corner": "https://investor.sebi.gov.in/",
    "ClearTax Blog": "https://cleartax.in/s/financial-planning",
    If the response is not found in these above links use your own knowledge and external links,
''',
            "If a concept is complex, break it down into **step-by-step explanations**.",
            "Avoid giving investment advice — focus on **teaching and explaining concepts**.",
            "Add proper citation and links from where the data is scraped",
            "List down news ,blogs, articles separately "
        ],
        expected_output="""
            # Research Summary Report
            
            ## Topic: [Research Topic]
            
            ### Key Findings
            - **Finding 1:** [Detailed explanation with supporting data]
            - **Finding 2:** [Detailed explanation with supporting data]
            - **Finding 3:** [Detailed explanation with supporting data]
            
            ### Source-Based Insights
            #### Source 1: [Source Name / URL]
            - **Summary:** [Concise summary of key points]
            - **Relevant Data:** [Key statistics, dates, or figures]
            - **Notable Quotes:** [Direct citations from experts, if available]
            
            #### Source 2: [Source Name / URL]
            - **Summary:** [Concise summary of key points]
            - **Relevant Data:** [Key statistics, dates, or figures]
            - **Notable Quotes:** [Direct citations from experts, if available]
            
            (...repeat for all sources...)
            
            ### Overall Trends & Patterns
            - **Consensus among sources:** [Common viewpoints and recurring themes]
            - **Diverging Opinions:** [Conflicting perspectives and debates]
            - **Emerging Trends:** [New insights, innovations, or potential shifts]
            
            ### Citations & References
            - [[Source 1 Name]]([URL])
            - [[Source 2 Name]]([URL])
            - [...list all sources with links...]
            
            ---""",
        add_context=True,
        add_references=True
    )


agent = initialize_agent()

# Streamlit UI
st.markdown('<div class="main-header">📈 Finance Expert</div>',
            unsafe_allow_html=True)

st.markdown(
    """
    Ask anything related to **Basic Finance Concepts**, **Investment & Markets** , **Banking & Loans**, or **Financial Literacy & Personal Finance**.
    
    **Example queries:**
    - What is the difference between saving and investing?
    - How does inflation affect purchasing power?
    - Can you explain what an IPO is?
    - What is a credit score and how is it calculated?
    """
)

# Create columns for the form
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_area("💬 Your Query:", height=100,
                         placeholder="e.g. What are the top 3 AI stocks to invest in?")

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    submit_button = st.button(
        "Get Insights", type="primary", use_container_width=True)
    clear_button = st.button("Clear", use_container_width=True)

# Initialize session state for storing responses
if 'responses' not in st.session_state:
    st.session_state.responses = []

# Clear functionality
if clear_button:
    st.session_state.responses = []
    query = ""
    st.rerun()

if submit_button and query.strip():
    with st.spinner("⏳ Analyzing investment data..."):
        # Create a placeholder for progress
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)

        # Simulate progress while waiting for response
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i + 1)

        # Get response from agent and extract the string content
        response = agent.run(query, markdown=True)

        # Handle the response based on its type
        if hasattr(response, 'content'):
            # For newer Agno versions that return a RunResponse object
            response_md = response.content
        elif hasattr(response, 'response'):
            # For some versions that might have a response attribute
            response_md = response.response
        elif isinstance(response, str):
            # If response is already a string
            response_md = response
        else:
            # Fallback - convert to string
            response_md = str(response)

        # Add response to session state
        st.session_state.responses.append({
            "query": query,
            "response": response_md,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        # Remove progress indicator
        progress_placeholder.empty()

# Display responses
if st.session_state.responses:
    for i, item in enumerate(reversed(st.session_state.responses)):
        st.markdown("---")
        st.markdown(f"**Query ({item['timestamp']}):**")
        st.info(item["query"])

        st.markdown("**Response:**")
        st.markdown('<div class="response-container">', unsafe_allow_html=True)
        st.markdown(item["response"], unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Download button for this response
        response_filename = f"investment_analysis_{i+1}.md"

        # Write to file safely with proper string handling
        with open(response_filename, "w") as f:
            f.write(
                f"# Investment Analysis\n\n**Query:** {item['query']}\n\n**Date:** {item['timestamp']}\n\n")
            f.write(item["response"])

        with open(response_filename, "rb") as file:
            st.download_button(
                label="Download this analysis",
                data=file,
                file_name=response_filename,
                mime="text/markdown",
                key=f"download_{i}"
            )

# Footer
st.markdown("---")
st.markdown(
    "*Powered by Azure OpenAI and Agno AI. Data provided by YFinance and DuckDuckGo.*")
