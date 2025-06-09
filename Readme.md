# NL2SQL Analytics API & Streamlit App

This folder contains the API and Streamlit web application for the NL2SQL Analytics Platform. The system enables users to query a database using natural language, automatically generates SQL, and provides advanced analytics and interactive visualizations.

## 📦 Features

- **Natural Language to SQL**: Converts user questions into SQL queries using LLMs.
- **Interactive Streamlit Dashboard**: Chat interface, visualizations, and analytics.
- **Data Export**: Download results as CSV or JSON.
- **Comprehensive Analytics**: Query history, performance metrics, and database insights.
- **Advanced Visualizations**: Dynamic charts using Plotly and Matplotlib.
- **Sample Database**: Auto-generated SQLite database for integration architecture.

## 🚀 Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**

   Create a `.env` file in this folder with your API keys:

   ```
   COHERE_API_KEY=your_cohere_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run streamlit_app.py
   ```

   The app will launch in your browser.

## 🗂️ File Structure

- [`main.py`](main.py): Core logic for NL2SQL chatbot and database manager.
- [`streamlit_app.py`](streamlit_app.py): Streamlit web UI for chat, analytics, and visualization.
- [`requirements.txt`](requirements.txt): Python dependencies for this module.
- `.env`: API keys and environment variables (not committed to git).

## 📝 Example Queries

- "Show me all projects by region"
- "What are the top 5 technologies used in integrations?"
- "Count integrations by security level"
- "Show me projects in the Healthcare industry"
- "What's the average data volume by technology?"

## ⚙️ Configuration

- The app uses a local SQLite database (`integration_architecture.db`). If it does not exist, it will be created automatically with sample data.
- API keys for LLMs (Cohere, Groq, etc.) must be set in `.env`.

## 📊 Visualization

- Results are visualized using Plotly charts.
- Data tables and summary statistics are shown for each query.

## 💾 Export

- Download query results as CSV or JSON.
- Export the entire database from the sidebar.

## 🛠️ Advanced

- SQL query explanations and optimization (coming soon).
- Query performance dashboard and analytics.

## 🤝 Contributing

Pull requests and suggestions are welcome!

---

**Note:** This app requires valid API keys for LLM services. Do not share your keys publicly.