"""
Scientific RAG Pipeline - Streamlit UI
A beautiful, user-friendly interface for the RAG pipeline.
"""

import streamlit as st
import requests
import os
from typing import Optional, Dict

# Configuration - use environment variable or default to localhost
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Scientific RAG Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Modern professional dark theme */
    .stApp { 
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%); 
    }
    
    /* Headers - clean white/silver */
    h1, h2, h3, h4 { 
        color: #e8e8e8 !important; 
        font-weight: 600 !important;
    }
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.5rem !important; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }
    
    /* Primary buttons - modern blue */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        border: none !important; 
        border-radius: 8px !important; 
        color: white !important; 
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Sidebar - Hidden */
    [data-testid="stSidebar"] { 
        display: none !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] { 
        color: #60a5fa !important; 
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important;
    }
    
    /* Tabs - modern styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 14px 28px !important;
        background: rgba(255,255,255,0.05) !important;
        color: #9ca3af !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px 8px 0 0 !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background: rgba(59, 130, 246, 0.1) !important;
        color: #e8e8e8 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border-color: #3b82f6 !important;
    }
    
    /* Tab content */
    [data-testid="stTabContent"] {
        padding: 24px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Input fields */
    .stTextInput > div > div, .stTextArea > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div:focus-within, .stTextArea > div > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.15);
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: 8px !important;
    }
    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 8px !important;
    }
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        border-radius: 8px !important;
    }
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 8px !important;
        color: #e8e8e8 !important;
    }
    
    /* Slider */
    .stSlider [data-baseweb="slider"] {
        color: #3b82f6 !important;
    }
    
    /* Toggle */
    [data-testid="stToggle"] {
        color: #9ca3af !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: #0d1117 !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.2);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)


def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        # Debug: Print error to console
        print(f"API Health check failed: {e}")
        return False


def get_stats() -> Dict:
    try:
        r = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {}


def query_documents(
    query: str, top_k: int = 5, use_cache: bool = True
) -> Optional[Dict]:
    try:
        r = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query, "top_k": top_k, "use_cache": use_cache},
            timeout=180,
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def execute_code(code: str) -> Optional[Dict]:
    try:
        r = requests.post(f"{API_BASE_URL}/execute", json={"code": code}, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def upload_file(uploaded_file, clear_first: bool = True) -> Optional[Dict]:
    try:
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), "application/zip")
        }
        r = requests.post(
            f"{API_BASE_URL}/ingest",
            files=files,
            data={"clear_first": clear_first},
            timeout=300,
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def run_evaluation(num_queries: int = 10) -> Optional[Dict]:
    try:
        r = requests.post(
            f"{API_BASE_URL}/eval", json={"num_queries": num_queries}, timeout=600
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def clear_data() -> bool:
    try:
        response = requests.delete(f"{API_BASE_URL}/clear", timeout=30)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
        return False


# Main tabs - Ingest, Query, Evaluate, Execute, Delete, Help
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📤 Ingest", "💬 Query", "📈 Evaluate", "🧮 Execute", "🗑️ Delete Data", "💡 Help"]
)

# API Status Indicator - positioned at the right of the tab bar
is_connected = check_api_health()
indicator_color = "#22c55e" if is_connected else "#ef4444"
indicator_text = "Connected" if is_connected else "Disconnected"

# Place indicator in a smaller container that floats at the top right
# Using position: sticky to keep it visible while scrolling
st.markdown(
    f"""
<div id="api-status-indicator" style="
    position: sticky;
    top: 10px;
    right: 20px;
    z-index: 9999;
    background: rgba(15, 15, 35, 0.95);
    padding: 6px 14px;
    border-radius: 15px;
    border: 1px solid {indicator_color};
    font-size: 11px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    float: right;
    margin-bottom: 10px;
">
    <span style="color: {indicator_color}; font-size: 0.9rem;">●</span>
    <span style="color: #e8e8e8;">{indicator_text}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ===== TAB 1: INGEST =====
with tab1:
    st.header("📤 Ingest Data")
    st.markdown(
        "Upload a ZIP file containing documents (PDF, TXT, CSV, DOCX, XLSX) to index."
    )

    uploaded_file = st.file_uploader("Choose a ZIP file:", type=["zip"])

    col1, col2 = st.columns([3, 1])
    with col1:
        clear_first = st.toggle("Clear existing data first", value=False)
    with col2:
        st.write("")
        st.write("")
        ingest_btn = st.button(
            "📤 Upload & Process", type="primary", use_container_width=True
        )

    if uploaded_file and ingest_btn:
        with st.spinner("Processing files..."):
            result = upload_file(uploaded_file, clear_first)
            if result:
                st.success(f"✅ {result.get('message', 'Done!')}")
                c1, c2 = st.columns(2)
                c1.metric("Files", result.get("files_processed", 0))
                c2.metric("Chunks", result.get("chunks_created", 0))
            else:
                st.error("Upload failed.")

    st.markdown("---")
    st.subheader("📋 Supported Formats")
    cols = st.columns(5)
    for i, fmt in enumerate(
        [("📄", "PDF"), ("📝", "TXT"), ("📊", "CSV"), ("📋", "DOCX"), ("📈", "XLSX")]
    ):
        cols[i].info(f"{fmt[0]} {fmt[1]}")

    with st.expander("💡 Tips"):
        st.markdown("""
        - **PDF**: Research papers, reports
        - **CSV/Excel**: Structured data, measurements
        - **TXT**: Plain text
        - **ZIP**: Combine multiple file types
        """)


# ===== TAB 2: QUERY =====
with tab2:
    st.header("💬 Ask Questions")
    st.markdown("Query your indexed documents and data using natural language.")

    query = st.text_area(
        "Ask a question:",
        placeholder="e.g., What does the research say about thermal conductivity?",
        height=80,
        label_visibility="collapsed",
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        top_k = st.slider("Results", 1, 20, 5)
    with c2:
        use_cache = st.toggle("Cache", value=True)
    with c3:
        st.write("")
        submit = st.button("🔍 Search", type="primary", use_container_width=True)

    if submit and query:
        with st.spinner("Searching..."):
            result = query_documents(query, top_k, use_cache)

            if result:
                st.subheader("📝 Answer")
                st.markdown(result.get("answer", "No answer found."))

                # Verification
                verification = result.get("verification", {})
                if verification:
                    is_hall = result.get("is_hallucinated", False)
                    if is_hall:
                        st.warning("⚠️ Potential hallucination detected")
                    else:
                        st.success("✅ Verified - answer supported by sources")
                    with st.expander("Verification details"):
                        st.json(verification)

                # Sources
                st.subheader("📚 Sources")
                sources = result.get("sources", [])
                if sources:
                    for i, s in enumerate(sources, 1):
                        st.text(f"{i}. {s}")
                else:
                    st.info("No sources found.")

                # Latency
                lat = result.get("latency_breakdown", {})
                if lat:
                    st.subheader("⏱️ Performance")
                    lc = st.columns(5)
                    lc[0].metric("Total", f"{lat.get('total_request_time', 0):.2f}s")
                    lc[1].metric("Analysis", f"{lat.get('query_analysis', 0):.2f}s")
                    lc[2].metric("Embed", f"{lat.get('embedding_generation', 0):.2f}s")
                    lc[3].metric("Retrieval", f"{lat.get('retrieval', 0):.2f}s")
                    lc[4].metric("LLM", f"{lat.get('llm_generation', 0):.2f}s")

                # Context
                with st.expander("🔍 Retrieved Context"):
                    for i, ctx in enumerate(result.get("retrieved_context", []), 1):
                        st.markdown(f"**Source {i}:** {ctx.get('source', 'Unknown')}")
                        st.markdown(
                            f"Type: `{ctx.get('type', 'Unknown')}` | Similarity: `{ctx.get('similarity', 0):.3f}`"
                        )
                        st.code(ctx.get("content", "")[:500])
                        st.markdown("---")


# ===== TAB 3: EVALUATE =====
with tab3:
    st.header("📈 Evaluate Pipeline")
    st.markdown("Run evaluation queries to test the pipeline's performance.")

    num_queries = st.slider("Number of queries", 1, 50, 10)

    if st.button("🚀 Run Evaluation", type="primary"):
        with st.spinner("Running evaluation..."):
            result = run_evaluation(num_queries)

            if result:
                st.success("✅ Evaluation Complete!")

                summary = result.get("summary", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Queries", result.get("num_queries", 0))
                c2.metric("Success", result.get("successful", 0))
                c3.metric("Failed", result.get("failed", 0))
                c4.metric("Avg Time", f"{summary.get('average_latency', 0):.2f}s")

                c1, c2 = st.columns(2)
                c1.metric("Cache Hit Rate", summary.get("cache_hit_rate", "N/A"))
                c2.metric(
                    "Hallucination Rate", summary.get("hallucination_rate", "N/A")
                )

                with st.expander("📊 Detailed Results"):
                    for i, r in enumerate(result.get("results", []), 1):
                        st.markdown(f"**Query {i}:** {r.get('query', '')}")
                        if "error" in r:
                            st.error(f"Error: {r.get('error', '')}")
                        else:
                            st.text(
                                f"Time: {r.get('total_time', 0):.2f}s | Sources: {r.get('sources_count', 0)} | Cached: {r.get('is_cached', False)}"
                            )
                            st.text(f"Answer: {r.get('answer', '')[:200]}...")
                        st.markdown("---")
            else:
                st.error("Evaluation failed.")


# ===== TAB 4: EXECUTE =====
with tab4:
    st.header("🧮 Execute Python Code")
    st.markdown("Run Python code in a safe sandbox environment.")

    code = st.text_area(
        "Enter Python code:",
        value="""# Example
result = (5 + 10) ** 2
print(f"Result: {result}")""",
        height=200,
    )

    if st.button("▶️ Execute", type="primary"):
        with st.spinner("Executing..."):
            result = execute_code(code)
            if result:
                if result.get("success"):
                    st.success("✅ Execution successful!")
                    st.subheader("Output:")
                    st.code(result.get("output", "No output"), language="text")
                else:
                    st.error("❌ Execution failed!")
                    st.subheader("Error:")
                    st.code(result.get("error", "Unknown error"), language="text")
                st.metric("Time", f"{result.get('execution_time', 0)}ms")


# ===== TAB 5: DELETE DATA =====
with tab5:
    st.header("🗑️ Delete Data")
    st.markdown("Manage your data by deleting all indexed documents and tables.")

    st.markdown(
        "<div style='background-color: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.5); border-radius: 8px; padding: 16px; margin-bottom: 16px;'>"
        "⚠️ <b>Warning:</b> This action cannot be undone. All your indexed data will be permanently deleted."
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🗑️ Delete All Data", type="primary", use_container_width=True):
            with st.spinner("Deleting data..."):
                if clear_data():
                    st.success("✅ All data deleted successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to delete data")

    st.markdown("---")
    st.subheader("📊 Current Data Status")
    stats = get_stats()
    if stats:
        doc_count = stats.get("document_chunks", 0)
        table_count = len(stats.get("structured_tables", []))

        c1, c2 = st.columns(2)
        c1.metric("Document Chunks", doc_count)
        c2.metric("Data Tables", table_count)

        if stats.get("structured_tables"):
            st.subheader("📋 Loaded Tables")
            for t in stats["structured_tables"]:
                st.code(t, language="sql")
    else:
        st.info("ℹ️ No data currently loaded.")


# ===== TAB 6: HELP =====
with tab6:
    st.header("💡 How to Use")
    st.markdown("Step-by-step guide to using the Scientific RAG Pipeline.")

    st.markdown("""
    ### 📤 Step 1: Upload Your Data
    Go to the **Ingest** tab and upload a ZIP file containing your documents or data files.
    - Supported formats: PDF, TXT, CSV, DOCX, XLSX
    - The system will index your files for retrieval
    
    ### 💬 Step 2: Ask Questions
    Use the **Query** tab to ask questions about your uploaded data.
    - The AI will search through your documents and data tables
    - Results include sources and verification
    
    ### 📈 Step 3: Test Performance
    Use **Evaluate** to check how well the system is working.
    - Run test queries to measure accuracy
    - Check cache performance and hallucination rates
    
    ### 🧮 Step 4: Run Code
    Use **Execute** to run Python code in a safe sandbox environment.
    - Great for data analysis and calculations
    - Results are displayed immediately
    
    ### 🗑️ Step 5: Manage Data
    Use **Delete Data** to remove all indexed documents and tables.
    - Useful when you want to start fresh
    - Cannot be undone, so use with caution
    """)

    st.markdown("---")
    st.subheader("ℹ️ Tips")
    st.markdown("""
    - **ZIP files** are the recommended way to upload multiple documents
    - **Cache** your queries for faster repeated access
    - **Evaluation** helps you understand system performance
    - **Execute** tab provides a safe sandbox for code execution
    """)


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>🔬 Scientific RAG Pipeline</div>",
    unsafe_allow_html=True,
)
