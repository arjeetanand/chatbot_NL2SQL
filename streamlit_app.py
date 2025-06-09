import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sqlite3
import os
import logging
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any
import io
import base64
from main import NL2SQLChatbot, DatabaseManager

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Database path & creation
db_path = "integration_architecture.db"
if not os.path.exists(db_path):
    DatabaseManager.create_sample_db(db_path)

# Initialize chatbot
chatbot = NL2SQLChatbot(db_path)

# Streamlit page config
st.set_page_config(
    page_title="🤖 AI-Powered Integration Analytics", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for better styling and readability
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #667eea;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 6px solid #2196f3;
        color: #0d47a1;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 6px solid #9c27b0;
        color: #4a148c;
    }
    
    .sql-code {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #4caf50;
        font-family: 'Courier New', monospace;
        color: #1b5e20;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    
    .export-button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        cursor: pointer;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: white;
    }
    
    /* Fix button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sql" not in st.session_state:
    st.session_state.last_sql = None
if "last_result_df" not in st.session_state:
    st.session_state.last_result_df = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = {}

# Helper Functions
def format_sql_query(sql_query: str) -> str:
    """Format SQL query with syntax highlighting"""
    if not sql_query:
        return ""
    
    # SQL keywords for highlighting
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN',
        'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'COUNT', 'DISTINCT', 'AS',
        'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'IS', 'NULL',
        'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'INDEX'
    ]
    
    formatted = sql_query
    for keyword in keywords:
        pattern = f'\\b{keyword}\\b'
        replacement = f'<span style="color: #0066cc; font-weight: bold;">{keyword}</span>'
        formatted = re.sub(pattern, replacement, formatted, flags=re.IGNORECASE)
    
    return formatted

def parse_sql_result(result_str: str) -> pd.DataFrame:
    """Parse SQL result string to DataFrame"""
    try:
        if isinstance(result_str, str) and result_str.strip():
            # Try to parse as JSON first
            if result_str.startswith('[') or result_str.startswith('{'):
                data = json.loads(result_str)
                if isinstance(data, list) and len(data) > 0:
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    return pd.DataFrame([data])
            else:
                # Try to split by lines and create a simple DataFrame
                lines = result_str.strip().split('\n')
                if len(lines) > 0:
                    return pd.DataFrame([{"Result": line} for line in lines if line.strip()])
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error parsing SQL result: {e}")
        return pd.DataFrame([{"Result": str(result_str)}])

def create_interactive_visualization(df: pd.DataFrame, chart_type: str = "auto") -> go.Figure:
    """Create interactive visualization based on data"""
    if df.empty:
        return None
    
    # Determine chart type automatically
    if chart_type == "auto":
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) >= 2:
            chart_type = "scatter"
        elif len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            chart_type = "bar"
        elif len(categorical_cols) >= 1:
            chart_type = "pie"
        else:
            chart_type = "table"
    
    try:
        # Create visualization with enhanced interactivity
        if chart_type == "bar" and len(df.columns) >= 2:
            fig = px.bar(
                df, 
                x=df.columns[0], 
                y=df.columns[1], 
                title=f"{df.columns[1]} by {df.columns[0]}",
                hover_data=df.columns.tolist(),
                color=df.columns[1] if df[df.columns[1]].dtype in ['int64', 'float64'] else None
            )
        elif chart_type == "histogram" and len(df.columns) >= 1:
            fig = px.histogram(
                df, 
                x=df.columns[0], 
                title=f"Distribution of {df.columns[0]}",
                marginal="box"
            )
        elif chart_type == "pie" and len(df.columns) >= 2:
            fig = px.pie(
                df, 
                names=df.columns[0], 
                values=df.columns[1] if len(df.columns) > 1 else None,
                title=f"{df.columns[1] if len(df.columns) > 1 else 'Count'} by {df.columns[0]}",
                hover_data=df.columns.tolist()
            )
        elif chart_type == "line" and len(df.columns) >= 2:
            fig = px.line(
                df, 
                x=df.columns[0], 
                y=df.columns[1],
                title=f"{df.columns[1]} over {df.columns[0]}",
                hover_data=df.columns.tolist(),
                markers=True
            )
        elif chart_type == "scatter" and len(df.columns) >= 2:
            color_col = df.columns[2] if len(df.columns) > 2 else None
            fig = px.scatter(
                df,
                x=df.columns[0],
                y=df.columns[1],
                color=color_col,
                title=f"{df.columns[1]} vs {df.columns[0]}",
                hover_data=df.columns.tolist(),
                size=df.columns[1] if df[df.columns[1]].dtype in ['int64', 'float64'] else None
            )
        else:
            return None
        
        # Enhanced styling
        fig.update_layout(
            template="plotly_white",
            height=500,
            showlegend=True,
            hovermode='x unified',
            title={
                'text': fig.layout.title.text,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            font={'family': 'Arial, sans-serif', 'size': 12, 'color': '#2c3e50'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        # Add interactive features
        fig.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>' +
                         'X: %{x}<br>' +
                         'Y: %{y}<br>' +
                         '<extra></extra>'
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return None

def get_database_insights() -> Dict[str, Any]:
    """Get database insights and metrics"""
    try:
        conn = sqlite3.connect(db_path)
        insights = {}
        
        # Get table counts
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            insights[f"{table_name}_count"] = count
        
        conn.close()
        return insights
    except Exception as e:
        logger.error(f"Error getting database insights: {e}")
        return {}

def create_download_link(df: pd.DataFrame, filename: str = "query_result") -> str:
    """Create download link for DataFrame"""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" style="text-decoration: none; background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 8px 16px; border-radius: 8px; display: inline-block; margin: 8px 0;">📥 Download CSV</a>'
        return href
    except Exception as e:
        logger.error(f"Error creating download link: {e}")
        return ""

# Main App Layout
st.markdown('''
<div class="main-header">
    <h1>🤖 AI-Powered Integration Analytics Platform</h1>
    <p style="font-size: 1.2rem; margin-bottom: 0;">Natural Language to SQL with Advanced Analytics & Interactive Visualizations</p>
</div>
''', unsafe_allow_html=True)

# Sidebar for controls and analytics
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("📊 Dashboard Controls")
    
    # Database insights
    st.subheader("Database Overview")
    insights = get_database_insights()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Total Projects", 
            insights.get("AA_Projects_count", 0),
            delta=None,
            help="Number of projects in the database"
        )
    with col2:
        st.metric(
            "Total Integrations", 
            insights.get("AA_Integrations_count", 0),
            delta=None,
            help="Number of integrations in the database"
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Query history
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("📈 Query Analytics")
    if st.session_state.query_history:
        st.success(f"✅ Queries executed: {len(st.session_state.query_history)}")
        
        # Show recent queries
        st.write("**Recent Queries:**")
        for i, query in enumerate(st.session_state.query_history[-3:]):
            with st.expander(f"Query {len(st.session_state.query_history) - i}", expanded=False):
                st.write(f"**Question:** {query.get('question', '')}")
                st.code(query.get('sql_query', ''), language="sql")
                st.write(f"**Time:** {query.get('timestamp', '')}")
    else:
        st.info("💡 No queries executed yet")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Export functionality
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("📥 Export Options")
    if st.session_state.last_result_df is not None and not st.session_state.last_result_df.empty:
        # Download button
        csv = st.session_state.last_result_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Last Result as CSV",
            data=csv,
            file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the last query result as CSV file"
        )
        
        # JSON download
        json_str = st.session_state.last_result_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📄 Download as JSON",
            data=json_str,
            file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Download the last query result as JSON file"
        )
        
        st.success("✅ Export ready!")
    else:
        st.info("💡 Execute a query to enable export")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization controls
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("📈 Visualization Settings")
    viz_type = st.selectbox(
        "Chart Type",
        ["auto", "bar", "line", "histogram", "pie", "scatter"],
        help="Select the type of chart for data visualization"
    )
    
    show_data_table = st.checkbox("Show Data Table", value=True, help="Display the raw data table")
    show_summary = st.checkbox("Show Data Summary", value=True, help="Display statistical summary")
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.header("💬 Interactive Chat Interface")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history with improved formatting
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f'''
                <div class="chat-message user-message">
                    <strong>🧑‍💼 You:</strong><br>
                    {msg["content"]}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="chat-message assistant-message">
                    <strong>🤖 AI Assistant:</strong><br>
                    {msg["content"]}
                </div>
                ''', unsafe_allow_html=True)
    
    # User input
    prompt = st.chat_input("💭 Ask your question about integrations, projects, or data analytics...")
    
    # Process user input
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show user message immediately
        st.markdown(f'''
        <div class="chat-message user-message">
            <strong>🧑‍💼 You:</strong><br>
            {prompt}
        </div>
        ''', unsafe_allow_html=True)
        
        try:
            # Get AI response
            with st.spinner("🔄 Processing your query... Please wait"):
                result = chatbot.query(prompt)
            
            if result.get("success", False):
                # Parse and store result
                result_data = result.get("result", "")
                df = parse_sql_result(result_data)
                
                st.session_state.last_sql = result.get("sql_query", "")
                st.session_state.last_result_df = df
                
                # Add to query history
                st.session_state.query_history.append({
                    "question": prompt,
                    "sql_query": result.get("sql_query", ""),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "result_rows": len(df) if not df.empty else 0
                })
                
                # Show AI response
                response_text = result.get("summary", "Query executed successfully!")
                st.markdown(f'''
                <div class="chat-message assistant-message">
                    <strong>🤖 AI Assistant:</strong><br>
                    {response_text}
                </div>
                ''', unsafe_allow_html=True)
                
                # Show formatted SQL with syntax highlighting
                if result.get("sql_query"):
                    formatted_sql = format_sql_query(result.get("sql_query", ""))
                    st.markdown(f'''
                    <div class="sql-code">
                        <strong>🔧 Generated SQL Query:</strong><br><br>
                        <code>{formatted_sql}</code>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Add to session state
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Show success message
                if not df.empty:
                    st.markdown(f'''
                    <div class="success-box">
                        ✅ <strong>Query executed successfully!</strong><br>
                        📊 Retrieved {len(df)} rows with {len(df.columns)} columns
                    </div>
                    ''', unsafe_allow_html=True)
                
            else:
                error_msg = f"❌ Error: {result.get('error', 'Unknown error occurred')}"
                st.markdown(f'''
                <div class="error-box">
                    {error_msg}
                </div>
                ''', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_msg = f"❌ Exception occurred: {str(e)}"
            st.markdown(f'''
            <div class="error-box">
                {error_msg}
            </div>
            ''', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

with col2:
    st.header("📊 Interactive Data Visualization")
    
    # Show visualization if we have recent result
    if st.session_state.last_result_df is not None and not st.session_state.last_result_df.empty:
        df = st.session_state.last_result_df
        
        st.subheader("📈 Dynamic Chart")
        
        # Create and show interactive chart
        fig = create_interactive_visualization(df, viz_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
            })
        
        # Show data table if enabled
        if show_data_table:
            st.subheader("📋 Data Table")
            st.dataframe(
                df, 
                use_container_width=True,
                height=300
            )
        
        # Show data summary if enabled
        if show_summary:
            st.subheader("📊 Data Insights")
            
            col_summary1, col_summary2 = st.columns(2)
            with col_summary1:
                st.metric("Total Rows", len(df))
                st.metric("Total Columns", len(df.columns))
            
            with col_summary2:
                numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                st.metric("Numeric Columns", len(numeric_cols))
                st.metric("Text Columns", len(categorical_cols))
            
            # Statistical summary for numeric columns
            if len(numeric_cols) > 0:
                st.write("**📈 Statistical Summary:**")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            # Value counts for categorical columns
            if len(categorical_cols) > 0 and len(categorical_cols) <= 3:
                st.write("**📝 Categorical Data Distribution:**")
                for col in categorical_cols[:2]:  # Show max 2 categorical columns
                    if df[col].nunique() <= 10:  # Only show if unique values <= 10
                        st.write(f"**{col}:**")
                        value_counts = df[col].value_counts()
                        st.bar_chart(value_counts)
    
    else:
        st.info("💡 Execute a query to see visualizations and data insights")
        
        # Show sample query suggestions
        st.subheader("💭 Sample Queries to Try:")
        sample_queries = [
            "Show me all projects by region",
            "What are the top 5 technologies used in integrations?",
            "Count integrations by security level",
            "Show me projects in the Healthcare industry",
            "What's the average data volume by technology?"
        ]
        
        for query in sample_queries:
            if st.button(f"🔄 {query}", key=f"sample_{hash(query)}"):
                st.session_state.messages.append({"role": "user", "content": query})
                st.experimental_rerun()

# Advanced Features Section
st.header("🚀 Advanced Features")

# Create tabs for different advanced features
tab1, tab2, tab3 = st.tabs(["🔍 SQL Query Analyzer", "🗂️ Database Schema", "📊 Query Performance"])

with tab1:
    if st.session_state.last_sql:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Explain SQL Query", key="explain_btn"):
                with st.spinner("Generating detailed explanation..."):
                    try:
                        explanation = chatbot.explain_sql(st.session_state.last_sql)
                        st.markdown("### 💡 SQL Query Explanation")
                        st.markdown(explanation)
                    except Exception as e:
                        st.error(f"Error generating explanation: {e}")
        
        with col2:
            if st.button("⚡ Optimize Query", key="optimize_btn"):
                st.info("🚧 Query optimization feature coming soon!")
    else:
        st.info("💡 Execute a SQL query first to access analysis features")

with tab2:
    try:
        st.subheader("🗂️ Complete Database Schema")
        
        # Get and display schema information
        schema_info = chatbot.get_table_info()
        st.code(schema_info, language="sql")
        
        # Enhanced schema diagram
        st.subheader("📊 Visual Schema Overview")
        
        schema_diagram = """
        🏢 AA_Projects (Main Projects Table)
        ├── 🔑 ID (Primary Key) - Unique project identifier
        ├── 📝 Name - Project name
        ├── 🌍 Region - Geographic region
        ├── 🏳️ Country - Country location
        └── 🏭 Industry - Business industry sector
        
        🔗 AA_Integrations (Integration Details)
        ├── 🔑 ID (Primary Key) - Unique integration identifier
        ├── 🔗 Project_ID (Foreign Key → AA_Projects.ID)
        ├── 📝 Integration_Name - Name of the integration
        ├── ⚙️ Technology - Technology stack used
        ├── 📊 Data_Volume - Volume of data processed
        └── 🔒 Security_Level - Security classification level
        
        🔄 Relationship: AA_Projects (1) ←→ (Many) AA_Integrations
        """
        
        st.code(schema_diagram, language="text")
        
        # Table statistics
        st.subheader("📈 Table Statistics")
        insights = get_database_insights()
        
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.metric("Projects Table", f"{insights.get('AA_Projects_count', 0)} records")
        with stats_col2:
            st.metric("Integrations Table", f"{insights.get('AA_Integrations_count', 0)} records")
        
    except Exception as e:
        st.error(f"❌ Error loading schema: {e}")

with tab3:
    st.subheader("📊 Query Performance Dashboard")
    
    if st.session_state.query_history:
        # Create performance metrics
        query_df = pd.DataFrame(st.session_state.query_history)
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Queries Over Time**")
            query_df['timestamp'] = pd.to_datetime(query_df['timestamp'])
            queries_per_hour = query_df.groupby(query_df['timestamp'].dt.floor('h')).size()
            st.line_chart(queries_per_hour)
        
        with col2:
            st.write("**📊 Result Set Sizes**")
            if 'result_rows' in query_df.columns:
                st.bar_chart(query_df['result_rows'].head(10))
        
        # Query complexity analysis
        st.write("**🔍 Query Complexity Analysis**")
        complexity_data = []
        for query in st.session_state.query_history:
            sql = query.get('sql_query', '')
            complexity = len(sql.upper().split(' WHERE ')) + len(sql.upper().split(' JOIN ')) + len(sql.upper().split(' GROUP BY '))
            complexity_data.append({
                'Query': query.get('question', '')[:50] + '...',
                'Complexity Score': complexity,
                'Length': len(sql)
            })
        
        if complexity_data:
            complexity_df = pd.DataFrame(complexity_data)
            st.dataframe(complexity_df, use_container_width=True)
    else:
        st.info("💡 Execute some queries to see performance analytics")

# Footer section with additional features
st.markdown("---")

# Quick Actions Section
st.header("⚡ Quick Actions")

quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)

with quick_col1:
    if st.button("🔄 Clear Chat History", help="Clear all chat messages"):
        st.session_state.messages = []
        st.success("✅ Chat history cleared!")

with quick_col2:
    if st.button("📊 Sample Data Preview", help="Show sample data from tables"):
        try:
            conn = sqlite3.connect(db_path)
            
            # Show sample from AA_Projects
            st.subheader("📋 Sample Projects Data")
            projects_df = pd.read_sql_query("SELECT * FROM AA_Projects LIMIT 5", conn)
            st.dataframe(projects_df, use_container_width=True)
            
            # Show sample from AA_Integrations
            st.subheader("🔗 Sample Integrations Data")
            integrations_df = pd.read_sql_query("SELECT * FROM AA_Integrations LIMIT 5", conn)
            st.dataframe(integrations_df, use_container_width=True)
            
            conn.close()
        except Exception as e:
            st.error(f"❌ Error loading sample data: {e}")

with quick_col3:
    if st.button("📈 Generate Report", help="Generate summary report"):
        try:
            conn = sqlite3.connect(db_path)
            
            # Generate comprehensive report
            st.subheader("📊 Analytics Report")
            
            # Projects by region
            region_query = """
            SELECT Region, COUNT(*) as Project_Count 
            FROM AA_Projects 
            GROUP BY Region 
            ORDER BY Project_Count DESC
            """
            region_df = pd.read_sql_query(region_query, conn)
            
            # Technology distribution
            tech_query = """
            SELECT Technology, COUNT(*) as Integration_Count 
            FROM AA_Integrations 
            GROUP BY Technology 
            ORDER BY Integration_Count DESC
            """
            tech_df = pd.read_sql_query(tech_query, conn)
            
            # Create report visualizations
            report_col1, report_col2 = st.columns(2)
            
            with report_col1:
                st.write("**Projects by Region**")
                fig_region = px.pie(region_df, names='Region', values='Project_Count', 
                                  title="Project Distribution by Region")
                fig_region.update_layout(height=300)
                st.plotly_chart(fig_region, use_container_width=True)
            
            with report_col2:
                st.write("**Technology Usage**")
                fig_tech = px.bar(tech_df, x='Technology', y='Integration_Count',
                                title="Integration Count by Technology")
                fig_tech.update_layout(height=300)
                st.plotly_chart(fig_tech, use_container_width=True)
            
            # Summary statistics
            st.write("**📊 Summary Statistics**")
            summary_stats = {
                "Total Projects": len(pd.read_sql_query("SELECT * FROM AA_Projects", conn)),
                "Total Integrations": len(pd.read_sql_query("SELECT * FROM AA_Integrations", conn)),
                "Unique Regions": len(region_df),
                "Unique Technologies": len(tech_df),
                "Report Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
            summary_df['Value'] = summary_df['Value'].astype(str) 
            st.table(summary_df)
            
            conn.close()
            
        except Exception as e:
            st.error(f"❌ Error generating report: {e}")

with quick_col4:
    if st.button("💾 Export All Data", help="Export complete database"):
        try:
            conn = sqlite3.connect(db_path)
            
            # Export all tables
            with st.spinner("Preparing complete data export..."):
                # Get all projects
                projects_df = pd.read_sql_query("SELECT * FROM AA_Projects", conn)
                
                # Get all integrations
                integrations_df = pd.read_sql_query("SELECT * FROM AA_Integrations", conn)
                
                # Create combined export
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Offer multiple download formats
                st.success("✅ Data export ready!")
                
                # Projects CSV
                projects_csv = projects_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Projects Data (CSV)",
                    data=projects_csv,
                    file_name=f"projects_export_{timestamp}.csv",
                    mime="text/csv"
                )
                
                # Integrations CSV
                integrations_csv = integrations_df.to_csv(index=False)
                st.download_button(
                    label="🔗 Download Integrations Data (CSV)",
                    data=integrations_csv,
                    file_name=f"integrations_export_{timestamp}.csv",
                    mime="text/csv"
                )
                
                # Combined JSON export
                combined_data = {
                    "projects": projects_df.to_dict('records'),
                    "integrations": integrations_df.to_dict('records'),
                    "export_timestamp": timestamp,
                    "total_records": len(projects_df) + len(integrations_df)
                }
                
                combined_json = json.dumps(combined_data, indent=2, default=str)
                st.download_button(
                    label="📄 Download Complete Dataset (JSON)",
                    data=combined_json,
                    file_name=f"complete_export_{timestamp}.json",
                    mime="application/json"
                )
            
            conn.close()
            
        except Exception as e:
            st.error(f"❌ Error exporting data: {e}")

# AI Assistant Tips Section
st.header("💡 AI Assistant Tips")

tips_col1, tips_col2 = st.columns(2)

with tips_col1:
    st.markdown("""
    ### 🎯 **Effective Query Tips**
    
    **✅ Good Examples:**
    - "Show me all projects in the Healthcare industry"
    - "What are the top 5 technologies by integration count?"
    - "List projects with high security level integrations"
    - "Compare data volumes across different technologies"
    
    **📊 Analytics Queries:**
    - "Calculate average data volume by region"
    - "Show integration distribution by security level"
    - "Find projects with the most integrations"
    """)

with tips_col2:
    st.markdown("""
    ### 🔧 **Advanced Features**
    
    **📈 Visualizations:**
    - Charts automatically adapt to your data
    - Use the sidebar to change chart types
    - Interactive charts with hover details
    
    **💾 Export Options:**
    - Download query results as CSV or JSON
    - Export complete database
    - Generate comprehensive reports
    
    **🔍 Query Analysis:**
    - Get explanations of generated SQL
    - View query performance metrics
    - Analyze query complexity
    """)

# System Status
st.header("🔧 System Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    # Check database connection
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1")
        conn.close()
        st.success("✅ Database: Connected")
    except:
        st.error("❌ Database: Error")

with status_col2:
    # Check chatbot status
    try:
        if chatbot:
            st.success("✅ AI Chatbot: Active")
        else:
            st.error("❌ AI Chatbot: Error")
    except:
        st.error("❌ AI Chatbot: Error")

with status_col3:
    # Session info
    query_count = len(st.session_state.query_history)
    st.info(f"📊 Session: {query_count} queries")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-top: 2rem;">
    <h3 style="color: #495057; margin-bottom: 1rem;">🚀 Powered by Advanced AI Technology</h3>
    <p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 0;">
        <strong>Oracle GenAI</strong> • <strong>LangChain</strong> • <strong>Streamlit</strong> • <strong>Plotly Interactive Charts</strong>
    </p>
    <p style="color: #6c757d; font-size: 0.9rem; margin-top: 1rem; margin-bottom: 0;">
        Built for Advanced Analytics & Real-time Business Intelligence
    </p>
</div>
""", unsafe_allow_html=True)