import os
import re
import ast    
import sqlite3
import datetime
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv
import pandas as pd
import requests
import oci
import matplotlib.pyplot as plt
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate

from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class CohereLLM(LLM):
    """Custom LLM wrapper for Oracle Generative AI"""
    
    
    _api_key: str = None
    _base_url: str = None

    def __init__(self, api_key: str, base_url: str = "https://api.cohere.ai/v1/generate", **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._base_url = base_url
        logger.info("Cohere LLM initialized successfully")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Cohere API model"""
        try:
            return self._generate_response_for_sql(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error calling Cohere API: {e}")
            raise

    def _call_cohere(self, prompt: str, max_tokens: int = 4000, temperature: float = 0.1) -> str:
        """Generate response using Cohere API"""
        try:
            headers = {
                'Authorization': f'Bearer {self._api_key}',
                'Content-Type': 'application/json',
            }

            data = {
                "model": "command-r-plus",  # You can change this based on your subscription
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "frequency_penalty": 0,
                "top_p": 0.75,
                "top_k": 0,
                "stop_sequences": [],  # Optional
                "return_likelihoods": "NONE"
            }

            response = requests.post(self._base_url, json=data, headers=headers)
            response.raise_for_status()

            raw_response = response.json()["generations"][0]["text"]
            logger.debug(f"Raw Cohere response: {raw_response}")
            return raw_response

        except Exception as e:
            logger.error(f"Error generating response from Cohere: {e}")
            raise
    
    def _generate_response_for_sql(self, prompt: str, max_tokens: int = 4000, temperature: float = 0.1) -> str:
        """Generate SQL response using Oracle GenAI"""
        raw_response = self._call_cohere(prompt, max_tokens, temperature)
        cleaned_response = self._extract_sql_from_response(raw_response)
        return cleaned_response

    def _generate_response_for_summary(self, prompt: str, max_tokens: int = 4000, temperature: float = 0.7) -> str:
        """Generate natural language summary using Oracle GenAI"""
        raw_response = self._call_cohere(prompt, max_tokens, temperature)
        return raw_response.strip()

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from response, handling markdown code blocks"""

        # Remove markdown code blocks (```sql ... ``` or ``` ... ```)
        sql_block_pattern = r'```(?:sql)?\s*\n?(.*?)\n?```'
        match = re.search(sql_block_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            # Extract the SQL from the code block
            sql_query = match.group(1).strip()
            logger.debug(f"Extracted SQL from code block: {sql_query}")
        else:
            # If no code blocks found, return the original response stripped
            sql_query = response.strip()
            logger.debug(f"No code blocks found, returning cleaned response: {sql_query}")
        
        # Validate and clean the SQL query
        cleaned_sql = self._validate_and_clean_sql(sql_query)
        return cleaned_sql
    
    def _validate_and_clean_sql(self, sql_query: str) -> str:
        """Validate and clean SQL query, handling common issues"""

        # Remove any leading/trailing whitespace
        sql_query = sql_query.strip()
        
        # Check for SQLite command-line commands and convert them
        if sql_query.startswith('.schema'):
            # Convert .schema to a proper SQL query showing table structure
            return "SELECT name, sql FROM sqlite_master WHERE type='table';"
        
        if sql_query.startswith('.tables'):
            # Convert .tables to a proper SQL query
            return "SELECT name FROM sqlite_master WHERE type='table';"
        
        # Check for other invalid patterns
        invalid_patterns = [
            r'^\s*\..*',  # Any SQLite dot commands
            r'^\s*DESCRIBE\s+',  # MySQL/PostgreSQL DESCRIBE
            r'^\s*SHOW\s+TABLES',  # MySQL/PostgreSQL SHOW TABLES
            r'^\s*\\d\s+',  # PostgreSQL \d commands
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, sql_query, re.IGNORECASE):
                logger.warning(f"Invalid SQL pattern detected: {sql_query}")
                # For table listing requests, return a standard query
                if any(word in sql_query.lower() for word in ['table', 'schema', 'describe']):
                    return "SELECT name FROM sqlite_master WHERE type='table';"
        
        # If it looks like a valid SQL query, return it
        if sql_query.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
            return sql_query
        
        # If we can't identify it as valid SQL, log a warning and return as-is
        logger.warning(f"Potentially invalid SQL query: {sql_query}")
        return sql_query

    @property
    def _llm_type(self) -> str:
        return "oracle_gen_ai"

class DatabaseManager:
    """Handles database operations and setup for Integration Architecture"""
    
    @staticmethod
    def create_sample_db(db_path: str = 'integration_architecture.db'):
        """Create a sample SQLite database with integration architecture data"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create AA_Projects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS AA_Projects (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Name TEXT,
                    Client TEXT,
                    Description TEXT,
                    Region TEXT,
                    Country TEXT,
                    Industry TEXT,
                    Owner_Vertical TEXT,
                    PM_Email TEXT,
                    Tech_Lead_Email TEXT
                )
            ''')
            
            # Create AA_Integrations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS AA_Integrations (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Project_ID INTEGER,
                    Name TEXT,
                    Description TEXT,
                    Data_Entity TEXT,
                    Technology TEXT,
                    Source_Application TEXT,
                    Source_Data_Format TEXT,
                    Source_Interface TEXT,
                    Target_Application TEXT,
                    Target_Data_Format TEXT,
                    Target_Interface TEXT,
                    Transformation_Flag BOOLEAN,
                    Trigger_Method TEXT,
                    Mode TEXT,
                    Frequency TEXT,
                    Data_Volume TEXT,
                    Compression_Flag BOOLEAN,
                    Encryption_Flag BOOLEAN,
                    Security_Policy TEXT,
                    Complexity TEXT,
                    Lifecycle TEXT,
                    Reusable_Flag BOOLEAN,
                    CIF_Onboarding_Flag BOOLEAN,
                    IP_Status TEXT,
                    FOREIGN KEY (Project_ID) REFERENCES AA_Projects (ID)
                )
            ''')
            
            # Create AA_Services table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS AA_Services (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Name TEXT,
                    Description TEXT,
                    Technology TEXT,
                    Version TEXT,
                    Build_Status TEXT,
                    Pattern TEXT,
                    Trigger_Type TEXT,
                    Outbound_Connection_Type TEXT,
                    CIF_Onboarding_Flag BOOLEAN
                )
            ''')
            
            # Create AA_Connections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS AA_Connections (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Name TEXT,
                    Type TEXT,
                    System_Name TEXT,
                    Hostname TEXT,
                    Port INTEGER,
                    Properties TEXT,
                    Agent TEXT
                )
            ''')
            
            # Create AA_Integration_to_Service_Mapping table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS AA_Integration_to_Service_Mapping (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Integration_ID INTEGER,
                    Service_ID INTEGER,
                    FOREIGN KEY (Integration_ID) REFERENCES AA_Integrations (ID),
                    FOREIGN KEY (Service_ID) REFERENCES AA_Services (ID)
                )
            ''')
            
            # Create AA_Service_to_Connection_Mapping table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS AA_Service_to_Connection_Mapping (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Service_ID INTEGER,
                    Connection_ID INTEGER,
                    FOREIGN KEY (Service_ID) REFERENCES AA_Services (ID),
                    FOREIGN KEY (Connection_ID) REFERENCES AA_Connections (ID)
                )
            ''')
            
            # Create AA_Lookups table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS AA_Lookups (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Type TEXT,
                    Code TEXT,
                    Value TEXT,
                    Description TEXT,
                    Enabled_Flag BOOLEAN
                )
            ''')
            
            # Insert sample projects data
            projects_data = [
                ('EIC Project Alpha', 'ClientX', 'Integration project for ClientX retail systems', 'APAC', 'India', 'Retail', 'Consumer Vertical', 'pm_clientx@example.com', 'techlead_clientx@example.com'),
                ('Digital Banking Hub', 'BankCorp', 'Core banking integration platform', 'NA', 'USA', 'Financial Services', 'Banking Vertical', 'pm_banking@example.com', 'techlead_banking@example.com'),
                ('Supply Chain Connect', 'ManufacturingCo', 'Supply chain visibility platform', 'EMEA', 'Germany', 'Manufacturing', 'Industrial Vertical', 'pm_supply@example.com', 'techlead_supply@example.com'),
                ('Healthcare Data Exchange', 'MedSystem', 'Patient data integration system', 'APAC', 'Singapore', 'Healthcare', 'Healthcare Vertical', 'pm_health@example.com', 'techlead_health@example.com'),
                ('E-commerce Analytics', 'RetailGiant', 'Real-time analytics integration', 'NA', 'Canada', 'E-commerce', 'Consumer Vertical', 'pm_ecom@example.com', 'techlead_ecom@example.com')
            ]
            
            cursor.executemany('INSERT INTO AA_Projects (Name, Client, Description, Region, Country, Industry, Owner_Vertical, PM_Email, Tech_Lead_Email) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', projects_data)
            
            # Insert sample integrations data
            integrations_data = [
                (1, 'Order Sync Integration', 'Sync orders from POS to ERP', 'Order', 'MuleSoft', 'POS System', 'JSON', 'REST API', 'ERP System', 'XML', 'SOAP API', True, 'Event-based', 'Asynchronous', 'Real-time', 'High', True, True, 'High Security', 'Medium', 'Production', True, True, 'Active'),
                (1, 'Customer Data Sync', 'Sync customer data across systems', 'Customer', 'Apache Camel', 'CRM System', 'CSV', 'File Transfer', 'Data Warehouse', 'Parquet', 'Batch Load', True, 'Scheduled', 'Batch', 'Daily', 'Medium', True, False, 'Standard', 'Low', 'Production', True, False, 'Active'),
                (2, 'Transaction Processing', 'Real-time transaction processing', 'Transaction', 'IBM Integration Bus', 'ATM Network', 'ISO8583', 'TCP/IP', 'Core Banking', 'SWIFT', 'MQ', True, 'Real-time', 'Synchronous', 'Real-time', 'Very High', False, True, 'High Security', 'High', 'Production', False, True, 'Active'),
                (2, 'Account Balance Sync', 'Account balance synchronization', 'Account', 'Spring Integration', 'Mobile App', 'JSON', 'REST API', 'Core Banking', 'JSON', 'REST API', False, 'Event-based', 'Asynchronous', 'Real-time', 'Medium', False, True, 'High Security', 'Low', 'Production', True, True, 'Active'),
                (3, 'Inventory Updates', 'Real-time inventory updates', 'Inventory', 'Apache Kafka', 'Warehouse System', 'Avro', 'Message Queue', 'ERP System', 'JSON', 'REST API', True, 'Event-based', 'Asynchronous', 'Real-time', 'High', True, False, 'Standard', 'Medium', 'Production', True, False, 'Active'),
                (3, 'Supplier Integration', 'Supplier data integration', 'Supplier', 'Talend', 'Supplier Portal', 'XML', 'Web Service', 'Procurement System', 'JSON', 'REST API', True, 'Scheduled', 'Batch', 'Hourly', 'Low', True, True, 'Standard', 'Medium', 'Production', True, True, 'Active'),
                (4, 'Patient Record Sync', 'Patient record synchronization', 'Patient', 'FHIR Gateway', 'EMR System', 'FHIR', 'REST API', 'Central Registry', 'HL7', 'Message Queue', True, 'Event-based', 'Asynchronous', 'Real-time', 'Medium', True, True, 'High Security', 'High', 'Production', False, True, 'Active'),
                (5, 'Sales Analytics Feed', 'Real-time sales data feed', 'Sales', 'AWS Kinesis', 'E-commerce Platform', 'JSON', 'Stream', 'Analytics Platform', 'Parquet', 'Batch Load', True, 'Stream', 'Asynchronous', 'Real-time', 'Very High', True, False, 'Standard', 'Low', 'Production', True, False, 'Active')
            ]
            
            cursor.executemany('INSERT INTO AA_Integrations (Project_ID, Name, Description, Data_Entity, Technology, Source_Application, Source_Data_Format, Source_Interface, Target_Application, Target_Data_Format, Target_Interface, Transformation_Flag, Trigger_Method, Mode, Frequency, Data_Volume, Compression_Flag, Encryption_Flag, Security_Policy, Complexity, Lifecycle, Reusable_Flag, CIF_Onboarding_Flag, IP_Status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', integrations_data)
            
            # Insert sample services data
            services_data = [
                ('Order Processing Service', 'Processes incoming order data and applies business rules', 'Java Spring Boot', '1.2.0', 'Deployed', 'Event-driven', 'Message Queue', 'REST', True),
                ('Customer Data Transformer', 'Transforms customer data between formats', 'Python Flask', '2.1.0', 'Deployed', 'Request-Response', 'HTTP Request', 'REST', False),
                ('Transaction Validator', 'Validates banking transactions', 'C# .NET Core', '3.0.1', 'Deployed', 'Synchronous', 'Direct Call', 'SOAP', True),
                ('Balance Inquiry Service', 'Handles account balance inquiries', 'Node.js Express', '1.5.2', 'Deployed', 'Event-driven', 'Message Queue', 'REST', True),
                ('Inventory Monitor', 'Monitors inventory levels and triggers alerts', 'Java Spring Cloud', '2.0.0', 'Deployed', 'Event-driven', 'Stream Processing', 'Message Queue', False),
                ('Supplier Onboarding', 'Handles supplier registration process', 'Python Django', '1.8.0', 'Testing', 'Request-Response', 'HTTP Request', 'REST', True),
                ('Patient Data Aggregator', 'Aggregates patient data from multiple sources', 'Java Spring Boot', '1.4.0', 'Deployed', 'Event-driven', 'Message Queue', 'REST', True),
                ('Sales Analytics Processor', 'Processes sales data for analytics', 'Scala Akka', '1.0.5', 'Deployed', 'Stream Processing', 'Stream', 'Message Queue', False)
            ]
            
            cursor.executemany('INSERT INTO AA_Services (Name, Description, Technology, Version, Build_Status, Pattern, Trigger_Type, Outbound_Connection_Type, CIF_Onboarding_Flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', services_data)
            
            # Insert sample connections data
            connections_data = [
                ('ERP Database Connection', 'Database', 'ERP_DB', 'erp.clientx.com', 5432, 'SSL=true;Timeout=30;ConnectionPool=10', 'PostgreSQL JDBC'),
                ('CRM System Connection', 'HTTP', 'CRM_API', 'api.crm.clientx.com', 443, 'SSL=true;AuthType=OAuth2;Timeout=60', 'HTTP Client'),
                ('Core Banking DB', 'Database', 'CORE_BANKING', 'core.bankcorp.com', 1521, 'SSL=true;ServiceName=PROD;ConnectionPool=20', 'Oracle JDBC'),
                ('ATM Network Gateway', 'TCP', 'ATM_GATEWAY', 'atm-gw.bankcorp.com', 8080, 'KeepAlive=true;Timeout=30;MaxConnections=100', 'TCP Socket'),
                ('Warehouse System API', 'HTTP', 'WMS_API', 'wms.manufacturing.com', 8443, 'SSL=true;AuthType=ApiKey;RateLimit=1000', 'HTTP Client'),
                ('Supplier Portal DB', 'Database', 'SUPPLIER_DB', 'supplier.manufacturing.com', 3306, 'SSL=true;Timeout=45;ConnectionPool=15', 'MySQL JDBC'),
                ('EMR System Interface', 'HTTP', 'EMR_FHIR', 'fhir.medsystem.com', 443, 'SSL=true;AuthType=Bearer;FHIR_Version=R4', 'FHIR Client'),
                ('Analytics Platform', 'HTTP', 'ANALYTICS_API', 'analytics.retailgiant.com', 443, 'SSL=true;AuthType=JWT;Timeout=120', 'HTTP Client'),
                ('Message Broker', 'Message Queue', 'KAFKA_CLUSTER', 'kafka.internal.com', 9092, 'SecurityProtocol=SASL_SSL;SaslMechanism=PLAIN', 'Kafka Client'),
                ('Data Warehouse', 'Database', 'DW_CLUSTER', 'warehouse.internal.com', 5439, 'SSL=true;Timeout=300;ConnectionPool=50', 'Redshift JDBC')
            ]
            
            cursor.executemany('INSERT INTO AA_Connections (Name, Type, System_Name, Hostname, Port, Properties, Agent) VALUES (?, ?, ?, ?, ?, ?, ?)', connections_data)
            
            # Insert Integration-to-Service mappings
            integration_service_mappings = [
                (1, 1), (1, 2),  # Order Sync uses Order Processing and Data Transformer
                (2, 2),         # Customer Data Sync uses Data Transformer
                (3, 3),         # Transaction Processing uses Validator
                (4, 4),         # Account Balance uses Balance Inquiry
                (5, 5),         # Inventory Updates uses Inventory Monitor
                (6, 6),         # Supplier Integration uses Supplier Onboarding
                (7, 7),         # Patient Record uses Data Aggregator
                (8, 8)          # Sales Analytics uses Analytics Processor
            ]
            
            cursor.executemany('INSERT INTO AA_Integration_to_Service_Mapping (Integration_ID, Service_ID) VALUES (?, ?)', integration_service_mappings)
            
            # Insert Service-to-Connection mappings
            service_connection_mappings = [
                (1, 1), (1, 9),  # Order Processing uses ERP DB and Message Broker
                (2, 2), (2, 10), # Data Transformer uses CRM and Data Warehouse
                (3, 3), (3, 4),  # Transaction Validator uses Core Banking and ATM Gateway
                (4, 3), (4, 9),  # Balance Inquiry uses Core Banking and Message Broker
                (5, 5), (5, 9),  # Inventory Monitor uses Warehouse API and Message Broker
                (6, 6),          # Supplier Onboarding uses Supplier DB
                (7, 7), (7, 9),  # Patient Aggregator uses EMR and Message Broker
                (8, 8), (8, 9)   # Analytics Processor uses Analytics API and Message Broker
            ]
            
            cursor.executemany('INSERT INTO AA_Service_to_Connection_Mapping (Service_ID, Connection_ID) VALUES (?, ?)', service_connection_mappings)
            
            # Insert lookup data
            lookups_data = [
                ('Security_Policy', 'HIGH_SEC', 'High Security', 'High level security with encryption and advanced authentication', True),
                ('Security_Policy', 'STD_SEC', 'Standard', 'Standard security with basic authentication', True),
                ('Security_Policy', 'LOW_SEC', 'Low Security', 'Basic security for internal systems', True),
                ('Technology', 'MULESOFT', 'MuleSoft', 'MuleSoft Anypoint Platform', True),
                ('Technology', 'CAMEL', 'Apache Camel', 'Apache Camel Integration Framework', True),
                ('Technology', 'SPRING', 'Spring Integration', 'Spring Integration Framework', True),
                ('Technology', 'KAFKA', 'Apache Kafka', 'Apache Kafka Streaming Platform', True),
                ('Data_Format', 'JSON', 'JSON', 'JavaScript Object Notation', True),
                ('Data_Format', 'XML', 'XML', 'Extensible Markup Language', True),
                ('Data_Format', 'CSV', 'CSV', 'Comma Separated Values', True),
                ('Data_Format', 'AVRO', 'Avro', 'Apache Avro Data Serialization', True),
                ('Interface_Type', 'REST', 'REST API', 'RESTful Web Service', True),
                ('Interface_Type', 'SOAP', 'SOAP API', 'Simple Object Access Protocol', True),
                ('Interface_Type', 'MQ', 'Message Queue', 'Message Queue Interface', True),
                ('Interface_Type', 'FILE', 'File Transfer', 'File-based Interface', True),
                ('Complexity', 'LOW', 'Low', 'Simple integration with minimal transformation', True),
                ('Complexity', 'MEDIUM', 'Medium', 'Moderate complexity with some business logic', True),
                ('Complexity', 'HIGH', 'High', 'Complex integration with extensive transformation', True),
                ('Frequency', 'REALTIME', 'Real-time', 'Real-time processing', True),
                ('Frequency', 'HOURLY', 'Hourly', 'Processed every hour', True),
                ('Frequency', 'DAILY', 'Daily', 'Processed once per day', True),
                ('Frequency', 'WEEKLY', 'Weekly', 'Processed once per week', True)
            ]
            
            cursor.executemany('INSERT INTO AA_Lookups (Type, Code, Value, Description, Enabled_Flag) VALUES (?, ?, ?, ?, ?)', lookups_data)
            
            conn.commit()
            conn.close()
            logger.info(f"Integration architecture database created successfully at {db_path}")
            
        except Exception as e:
            logger.error(f"Error creating integration architecture database: {e}")
            raise
    

class NL2SQLChatbot:
    """Enhanced chatbot class"""
    
    def __init__(self, db_path: str):
        """Initialize the NL2SQL chatbot"""
        self.db_path = db_path
        # self.oci_config = oci_config
        self.conn = sqlite3.connect(self.db_path)
        self.query_cache = {}  # Simple in-memory cache
        self.performance_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_response_time": 0.0,
            "cache_hits": 0
        }
        self.session_context = {}  # session_id → last question, last sql_query, last sql_rows

        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize database and LLM components"""
        try:
            # Initialize database connection
            self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            logger.info(f"Connected to database: {self.db_path}")
            
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS AA_Analytics (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Query_Text TEXT,
                    SQL_Generated TEXT,
                    Execution_Time REAL,
                    Success INTEGER,
                    Timestamp DATETIME,
                    Session_ID TEXT
                )
                ''')
                conn.commit()
                conn.close()
                logger.info("✅ AA_Analytics table ensured.")
            except Exception as e:
                logger.error(f"Error creating AA_Analytics table: {e}")


            # Initialize Oracle GenAI LLM
            self.llm = CohereLLM(api_key=os.getenv("COHERE_API_KEY"))
            
            # SQL prompt template
            self.sql_prompt = PromptTemplate(
                input_variables=["input", "table_info", "dialect"],
                template="""You are an expert SQL assistant. Given a natural language question, generate a syntactically correct {dialect} query.

Available tables and their schema:
{table_info}

IMPORTANT RULES:
1. Generate ONLY valid SQL SELECT statements
2. DO NOT use SQLite command-line commands like .schema, .tables, .describe
3. Use standard SQL syntax that works with {dialect}
4. Use only tables and columns that exist in the schema above
5. If a required column does not exist, do not invent it — instead, use an appropriate existing column or skip it.
6. If a required JOIN is needed, use only columns that exist in the joined tables.
7. For aggregations, use appropriate GROUP BY clauses.
8. Use LIMIT when appropriate to avoid huge result sets.
9. Return ONLY the SQL query without any markdown formatting, explanations, or code blocks.

STRICT INSTRUCTIONS FOR COLUMN USAGE:
- You MUST use only columns present in the schema above.
- You MUST NOT invent columns (example of invalid behavior: using a column called Pattern in AA_Integrations if it does not exist).
- If unsure, leave that part out of the query.

IMPORTANT NOTE ABOUT JOINS:
- AA_Integrations table does NOT have Region or Country directly.
- To access Region or Country, JOIN AA_Integrations.Project_ID → AA_Projects.ID and use AA_Projects.Region or AA_Projects.Country.
- If you need Region, always use JOIN with AA_Projects.

SQLITE-SPECIFIC RULES:
- When using GROUP_CONCAT with DISTINCT, do NOT include a separator argument.
- Correct: GROUP_CONCAT(DISTINCT column_name)
- Incorrect: GROUP_CONCAT(DISTINCT column_name, ',')

Examples of VALID queries:
- SELECT * FROM table_name;
- SELECT column1, column2 FROM table_name WHERE condition;
- SELECT COUNT(*) FROM table_name;
- SELECT GROUP_CONCAT(DISTINCT column_name) FROM table_name;
- SELECT P.Region, COUNT(I.ID) FROM AA_Integrations AS I JOIN AA_Projects AS P ON I.Project_ID = P.ID GROUP BY P.Region;

Examples of INVALID queries (DO NOT USE):
- .schema
- .tables  
- DESCRIBE table_name
- SHOW TABLES
- GROUP_CONCAT(DISTINCT column_name, ',')

Question: {input}

SQL Query:
"""
)
            
            # Initialize SQL database chain
            self.db_chain = SQLDatabaseChain.from_llm(
                llm=self.llm,
                db=self.db,
                verbose=True,
                return_intermediate_steps=True,
                prompt=self.sql_prompt
            )
            
            logger.info("NL2SQL Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing chatbot components: {e}")
            raise
    
    def log_audit(self, question: str, result: Any, summary: str, execution_time: float = 0.0, session_id: str = None):
        """Enhanced audit logging with performance metrics"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            log_entry = f"""
                    ============================================================
                    TIMESTAMP: {timestamp}
                    SESSION: {session_id or 'N/A'}
                    EXECUTION TIME: {execution_time:.3f}s
                    QUESTION: {question}

                    SQL QUERY:
                    {result}

                    SUMMARY:
                    {summary}
                    ============================================================
                    """
            with open("chat_log.txt", "a", encoding="utf-8") as f:
                f.write(log_entry)

            # Log to database if analytics table exists
            self._log_to_analytics_db(question, str(result), execution_time, True, session_id)
            logger.info("✅ Audit log updated.")

        except Exception as e:
            logger.error(f"Error writing audit log: {e}")

    def _log_to_analytics_db(self, query_text: str, sql_generated: str, execution_time: float, 
                            success: bool, session_id: str = None):
        """Log query analytics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO AA_Analytics (Query_Text, SQL_Generated, Execution_Time, Success, Timestamp, Session_ID)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (query_text, sql_generated, execution_time, success, datetime.datetime.now(), session_id or ''))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging to analytics DB: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get chatbot performance metrics"""
        return self.performance_metrics.copy()
    
    def get_query_analytics(self) -> Dict[str, Any]:
        """Get query analytics from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('''
            SELECT 
                COUNT(*) as total_queries,
                SUM(CASE WHEN Success = 1 THEN 1 ELSE 0 END) as successful_queries,
                AVG(Execution_Time) as avg_execution_time,
                DATE(Timestamp) as query_date,
                COUNT(*) as daily_count
            FROM AA_Analytics 
            GROUP BY DATE(Timestamp)
            ORDER BY query_date DESC
            LIMIT 30
            ''', conn)
            conn.close()
            
            return {
                "daily_metrics": df.to_dict('records'),
                "summary": {
                    "total_queries": int(df['total_queries'].sum()) if not df.empty else 0,
                    "avg_success_rate": float(df['successful_queries'].sum() / df['total_queries'].sum()) if not df.empty and df['total_queries'].sum() > 0 else 0,
                    "avg_execution_time": float(df['avg_execution_time'].mean()) if not df.empty else 0
                }
            }
        except Exception as e:
            logger.error(f"Error getting query analytics: {e}")
            return {"daily_metrics": [], "summary": {}}
    
    # Helper method to invoke LLM with retry
    def _invoke_with_retry(self, question, max_retries=2):
        for attempt in range(max_retries):
            try:
                print(f"👉 Attempt {attempt + 1}/{max_retries}...")
                result = self.db_chain.invoke({"query": question})
                # If no exception, return result
                return result
            except Exception as e:
                print(f"⚠️ LLM attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    # Optionally adjust prompt with feedback
                    question = f"{question}. NOTE: Previous SQL had error: {e}. Please correct it."
                else:
                    raise e

    def query(self, question: str, session_id: str = None) -> Dict[str, Any]:
        """Enhanced query processing with analytics and caching"""
        start_time = datetime.datetime.now()
        sql_query = None   # Initialize early, to avoid unbound error later
        sql_rows = []

        try:
            # Update metrics
            self.performance_metrics["total_queries"] += 1
            
            logger.info(f"Processing query: {question}")
            
            # Check cache first
            cache_key = question.lower().strip()
            if cache_key in self.query_cache:
                self.performance_metrics["cache_hits"] += 1
                cached_result = self.query_cache[cache_key].copy()
                cached_result["from_cache"] = True
                return cached_result
            
            # Handle common queries
            question_lower = question.lower()
            if any(phrase in question_lower for phrase in ['list table', 'show table', 'table name', 'what table']):
                table_names = self.get_table_names()
                result = {
                    "question": question,
                    "sql_query": "SELECT name FROM sqlite_master WHERE type='table';",
                    "result": f"Available tables: {', '.join(table_names)}",
                    "summary": f"Found {len(table_names)} tables in the database: {', '.join(table_names)}",
                    "success": True,
                    "execution_time": 0.1
                }
                return result
            
            # OPTIONAL: Use prior context if question is vague:
            if session_id:
                if session_id not in self.session_context:
                    self.session_context[session_id] = {
                        "last_questions": []
                    }

                if "last_questions" not in self.session_context[session_id]:
                    self.session_context[session_id]["last_questions"] = []
                
                context = ""
                # If vague question (<= 4 words), add prior question as context
                if len(question.split()) <= 4 and self.session_context[session_id]["last_questions"]:
                    # prior_question = self.session_context[session_id]["last_questions"][-1]
                    recent_questions = self.session_context[session_id]["last_questions"][-2:]
                    context_text = " → ".join(recent_questions)
                    context = context_text
                    print("🔁 Using prior context from:", context_text)
                    question = f"{question}  (context: {context_text}) "
                    print("Updated question with context:", question)

                # Store raw user question, strip context part if present
                if " (context:" in question:
                    raw_question = question.split(" (context:")[0].strip()
                else:
                    raw_question = question.strip()

                self.session_context[session_id]["last_questions"].append(raw_question)
                self.session_context[session_id]["last_sql_query"] = sql_query
                self.session_context[session_id]["last_result"] = sql_rows


            # 1️⃣ Generate SQL
            try:
                result = self._invoke_with_retry(question, max_retries=2)
    #             print("DEBUG BEFORE SUMMARY CALL — result['result'] type:", type(result.get("result")), 
    #   ", value example:", result.get("result")[:3] if isinstance(result.get("result"), (list, tuple)) else result.get("result"))

            except Exception as e:
                execution_time = (datetime.datetime.now() - start_time).total_seconds()
                logger.error(f"Error processing query '{question}': {e}")

                error_response = {
                    "question": question,
                    "error": str(e),
                    "success": False,
                    "execution_time": execution_time,
                    "from_cache": False
                }

                # Log error
                self._log_to_analytics_db(question, "", execution_time, False, session_id)

                return error_response


            # Fallback rows list
            sql_rows = []

            # Step 1 — try to parse SQLResult first
            intermediate_step = result.get("intermediate_steps", [{}])[0]
            sql_input_text = intermediate_step.get("input", "")

            # Use regex to extract SQLResult from intermediate_steps
            sql_result_match = re.search(r'SQLResult:\s*\n*(\[.*?\])', sql_input_text, re.DOTALL)

            if sql_result_match:
                sql_result_raw = sql_result_match.group(1)
                try:
                    sql_rows = ast.literal_eval(sql_result_raw)
                    print(f"✅ Parsed SQLResult with {len(sql_rows)} rows.")
                except Exception as e:
                    print(f"⚠️ Failed to parse SQLResult: {e}")
                    sql_rows = []

            # Step 2 — fallback: execute sql_cmd if SQLResult was not found or parsing failed
            if not sql_rows:
                sql_query = intermediate_step.get("sql_cmd", "")
                if not sql_query:
                    # As fallback if sql_cmd missing, fallback to result['result'] (only if it is SELECT)
                    sql_query = result.get("result", "")
                    if sql_query.lower().strip().startswith("select"):
                        sql_query_fallback_used = True
                        # print("⚠️ No sql_cmd, fallback to result['result'] as SQL.")
                    else:
                        sql_query_fallback_used = False
                        print("❌ No valid SQL to execute. Skipping fallback execution.")
                        sql_query = None
                
                # if sql_query_fallback_used:
                #     print(f"Fallback used: executing SELECT from result['result']")

                # Execute only if valid sql_query
                if sql_query:
                    # print(f"⚠️ No parsed SQLResult — fallback to execute sql_query:\n{sql_query}")
                    cursor = self.conn.cursor()
                    cursor.execute(sql_query)
                    sql_rows = cursor.fetchall()
                    # print(f"✅ SQL execution fallback: Retrieved {len(sql_rows)} rows.")

            # 3️⃣ Generate Summary Prompt
            summary_prompt = f"""You are a data analyst assistant and are a helpful assistant. The user asked the following question about their database as a sql response:.

                Question: {question}
                You also have the result for this question as a sql response:

                
                SQL Result: {sql_rows}


                Please provide a short and direct answer to the user’s question based on the result.
                IMPORTANT:
                - Read the user’s question carefully.
                - Analyze the actual result. Dont assume anything. what  ever is the result, use it.
                - Use the result to provide a concise answer.
                - Provide a short, natural language answer to the question — do not repeat the question.
                - Do not explain the SQL query.
                - Use ONLY the provided result — do not add or invent values.
                - If the result is empty, say "No results found."
                -Dont print the sql query or any other information apart from the answer.
                Answer:"""

            summary = self.llm._generate_response_for_summary(summary_prompt)
            
            # After calling LLM:
            # print("DEBUG AFTER SUMMARY CALL — result['result'] type:", type(result.get("result")), 
    #   ", value example:", result.get("result")[:3] if isinstance(result.get("result"), (list, tuple)) else result.get("result"))
            # Calculate execution time
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Prepare response
            response = {
                "question": question,
                "sql_query": sql_query,
                "result": sql_rows,  
                "summary": summary,
                "success": True,
                "execution_time": execution_time,
                "from_cache": False
            }
           
            # Cache successful results
            self.query_cache[cache_key] = response.copy()
            
            # Update metrics
            self.performance_metrics["successful_queries"] += 1
            self.performance_metrics["average_response_time"] = (
                (self.performance_metrics["average_response_time"] * (self.performance_metrics["total_queries"] - 1) + execution_time) 
                / self.performance_metrics["total_queries"]
            )
            
            # Log with analytics
            self.log_audit(question, result["result"], summary, execution_time, session_id)
            
            return response
            
        except Exception as e:
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            logger.error(f"Error processing query '{question}': {e}")
            
            error_response = {
                "question": question,
                "error": str(e),
                "success": False,
                "execution_time": execution_time,
                "from_cache": False
            }
            
            # Log error
            self._log_to_analytics_db(question, "", execution_time, False, session_id)
            
            return error_response
    
    def get_table_info(self) -> str:
        """Get detailed information about database tables"""
        try:
            return self.db.get_table_info()
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return "Error retrieving table information"
    
    def get_table_names(self) -> List[str]:
        """Get list of table names"""
        try:
            return self.db.get_usable_table_names()
        except Exception as e:
            logger.error(f"Error getting table names: {e}")
            return []
    

    def explain_sql(self, result: str) -> str:
        """Generate explanation for SQL query"""
        try:
            explanation_prompt = f"""
    You are a helpful SQL expert and data analyst assistant. Explain the following SQL query, You must explain the EXACT SQL query provided below in simple, clear terms in short bullet points.:

    SQL Query:
    {result}

    IMPORTANT:
    - You MUST explain THIS SQL query and THIS query only.
    - You MUST NOT refer to any 'products' or 'sales' or any generic tables — use only the actual tables present.
    - You MUST explain the correct meaning of this query.
    - Use bullet points if helpful.
    
    Example:
    SQL Query:
    SELECT COUNT(I.ID)
    FROM AA_Integrations AS I
    JOIN AA_Projects AS P ON I.Project_ID = P.ID
    WHERE P.Region = 'APAC';

    Explanation:
    - This query counts the number of integrations in the APAC region.

    Explanation:
    """
            explanation = self.llm._generate_response_for_summary(explanation_prompt)
            return explanation.strip()
        
        except Exception as e:
            logger.error(f"Error generating SQL explanation: {e}")
            return f"Error explaining SQL: {e}"

    def _show_graph_examples(self):
        print("\n💡 Example graph prompts you can try:")
        print("1. Show me how many projects are there in each region.")
        print("2. Show me a breakdown of integrations by complexity.")
        print("3. List all countries where we have projects.")
        print("4. Analyze connection usage patterns — show connection system name, type, and services.")
        # print()

    def chat(self):
        """Enhanced interactive chat session with analytics"""
        print("🤖 Enhanced NL2SQL Analytics Chatbot!")
        print(f"📊 Available tables: {', '.join(self.get_table_names())}")
        print("\n" + "="*60)
        print("Ask me questions about the database for insights and analytics!")
        print("Commands:")
        print("  - 'quit' or 'exit': Exit the chatbot")
        print("  - 'tables': Show table information")
        print("  - 'metrics': Show performance metrics")
        print("  - 'analytics': Show query analytics")
        print("="*60 + "\n")
        
        session_id = f"chat_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'tables':
                    print("\n📋 Table Information:")
                    print(self.get_table_info())
                    continue

                if user_input.lower() == 'examples':
                    self._show_examples()
                    continue  

                if user_input.lower() == 'metrics':
                    metrics = self.get_performance_metrics()
                    print(f"\n📊 Performance Metrics:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
                    continue
                
                if user_input.lower() == 'analytics':
                    analytics = self.get_query_analytics()
                    print(f"\n📈 Query Analytics:")
                    print(f"  Summary: {analytics['summary']}")

                    # Plot daily query count
                    df = pd.DataFrame(analytics['daily_metrics'])
                    if not df.empty:
                        df = df.sort_values('query_date')

                        plt.figure(figsize=(10, 6))
                        plt.plot(df['query_date'], df['total_queries'], marker='o', label='Total Queries')
                        plt.plot(df['query_date'], df['successful_queries'], marker='x', label='Successful Queries')
                        plt.title('Daily Query Analytics')
                        plt.xlabel('Date')
                        plt.ylabel('Count')
                        plt.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.show()
                    else:
                        print("No analytics data to plot.")
                    continue
                
                if user_input.lower() == 'graph_examples':
                    self._show_graph_examples()
                    continue

                if not user_input:
                    continue
                
                print("\n🔍 Processing your question...")
                result = self.query(user_input, session_id)

                if result["success"]:
                    print(f"✅ SQL Query: {result.get('sql_query', 'N/A')}")
                    print(f"✅ SQL Result: {result['result']}")
                    print(f"📝 Summary: {result['summary']}")
                    print(f"⏱️ Execution time: {result.get('execution_time', 0):.3f}s")
                    if result.get('from_cache'):
                        print("🚀 (Retrieved from cache)")

                    try:
                        # Guard: ensure result['result'] is list of tuples or list of lists, not string
                        # print(f"DEBUG OUTER: result['result'] type = {type(result['result'])}, value = {result['result'][:5] if isinstance(result['result'], (list, tuple)) else result['result']}")
                        
                        if result['result'] and isinstance(result['result'], (list, tuple)):
                            first_item = result['result'][0]
                            if isinstance(first_item, (list, tuple)):
                                # Proceed with normalization + visualization
                                max_columns = max(len(row) for row in result['result']) if result['result'] else 0
                                # print("columns", max_columns)
                                normalized_result = [tuple(list(row) + [None] * (max_columns - len(row))) for row in result['result']]
                                # print("normalized result", normalized_result)
                                df = pd.DataFrame(normalized_result, columns=[f'Col{i+1}' for i in range(max_columns)])

                                # DEBUG: show DataFrame shape and columns
                                # print(f"DEBUG: df.columns = {df.columns.tolist()}, df.shape = {df.shape}")

                                if not df.empty and len(df.columns) >= 1:
                                    print("📊 Attempting auto-visualization of query result...")

                                    # Heuristic:

                                    # 1 column → bar chart of frequencies
                                    if len(df.columns) == 1:
                                        value_counts = df[df.columns[0]].value_counts()
                                        plt.figure(figsize=(10, 6))
                                        plt.bar(value_counts.index, value_counts.values)
                                        plt.title(f'Frequency of {df.columns[0]}')
                                        plt.xlabel(df.columns[0])
                                        plt.ylabel('Frequency')
                                        plt.xticks(rotation=45)
                                        plt.tight_layout()
                                        plt.show()

                                    # 2 columns:
                                    elif len(df.columns) == 2:
                                        if pd.api.types.is_numeric_dtype(df[df.columns[1]]):
                                            # Standard bar chart
                                            df = df.sort_values(df.columns[1], ascending=False)
                                            plt.figure(figsize=(10, 6))
                                            plt.bar(df[df.columns[0]], df[df.columns[1]])
                                            plt.title('Auto-Generated Bar Chart')
                                            plt.xlabel(df.columns[0])
                                            plt.ylabel(df.columns[1])
                                            plt.xticks(rotation=45)
                                            plt.tight_layout()
                                            plt.show()
                                        elif df[df.columns[1]].nunique() <= 10:
                                            # Pie chart if small number of categories
                                            value_counts = df[df.columns[1]].value_counts()
                                            plt.figure(figsize=(8, 8))
                                            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
                                            plt.title(f'Pie Chart of {df.columns[1]}')
                                            plt.tight_layout()
                                            plt.show()
                                        else:
                                            # Fallback bar chart
                                            value_counts = df[df.columns[1]].value_counts()
                                            plt.figure(figsize=(10, 6))
                                            plt.bar(value_counts.index, value_counts.values)
                                            plt.title(f'Frequency of {df.columns[1]}')
                                            plt.xlabel(df.columns[1])
                                            plt.ylabel('Frequency')
                                            plt.xticks(rotation=45)
                                            plt.tight_layout()
                                            plt.show()

                                    # 3 columns:
                                    elif len(df.columns) == 3:
                                        # Try pie chart if last column looks like count
                                        if 'count' in df.columns[-1].lower() or pd.api.types.is_numeric_dtype(df[df.columns[2]]):
                                            try:
                                                # If numeric → pie chart
                                                df.columns = ['Label', 'Value', 'Count']
                                                plt.figure(figsize=(8, 8))
                                                plt.pie(df['Count'], labels=df['Label'], autopct='%1.1f%%', startangle=140)
                                                plt.title('Pie Chart')
                                                plt.tight_layout()
                                                plt.show()
                                            except Exception as pie_error:
                                                print(f"⚠️ Could not plot pie chart: {pie_error}")

                                        else:
                                            # Fallback: bar chart of Col3 frequencies
                                            value_counts = df[df.columns[2]].value_counts()
                                            plt.figure(figsize=(10, 6))
                                            plt.bar(value_counts.index, value_counts.values)
                                            plt.title(f'Frequency of {df.columns[2]}')
                                            plt.xlabel(df.columns[2])
                                            plt.ylabel('Frequency')
                                            plt.xticks(rotation=45)
                                            plt.tight_layout()
                                            plt.show()

                                    else:
                                        print("ℹ️ Query result not suitable for automatic chart.")
                                else:
                                    print("ℹ️ Query result not suitable for automatic chart.")
                            else:
                                print("⚠️ Skipping auto-visualization: result is list but not of tuples (possibly LLM text fallback).")
                        else:
                            print("⚠️ Skipping auto-visualization: result format not suitable (likely not SQL result rows).")

                    except Exception as e:
                        print(f"⚠️ Auto-visualization error: {e}")

                    # Show explanation option
                    if input("\nWould you like an explanation of the SQL? (y/n): ").lower() == 'y':
                        explanation = self.explain_sql(result['result'])
                        print(f"💡 Explanation: {explanation}")
                        
                else:
                    # print(f"❌ Error: {result['error']}")
                    continue
                
                print("-" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error in chat loop: {e}")
                print(f"❌ An unexpected error occurred: {e}")
    
    def _show_examples(self):
        """Show example queries"""
        examples = [
            "Get all projects with their basic details",
            "Find all real-time integrations with high data volume",
            "Count integrations by technology",
            "Find integrations with encryption and security policies",
            "Find reusable integrations and their usage patterns",
            "Find employees who earn more than $70,000",
            "Identify potentially problematic integration patterns",
            "Get project team contacts for coordination",
            "Analyze data format transformations across integrations",
            "Monitor service deployment and build status",
            "Analyze integration distribution by region and industry",
            "Analyze connection usage patterns - get connections and their services"
        ]
        
        print("\n💡 Example queries you can try:")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
        print()

def main():
    """Main function to run the chatbot"""
    try:
        db_path = "integration_architecture.db"
        
        # Create sample database if it doesn't exist
        if not os.path.exists(db_path):
            logger.info("Creating sample database...")
            DatabaseManager.create_sample_db(db_path)
        
        # Initialize chatbot
        chatbot = NL2SQLChatbot(db_path)
        
        # Test with example queries
        # example_queries = [
            # "Show integration count + encrypted % per region",
            # "List connections NOT used by any service",
            # "How many integrations are there in APAC region?"
            # "Projects where ALL integrations are encrypted",
            # "Find services that support more than one connection with the number of connections",
            # "Find top 2 most connected systems (by connection usage)",
            # "Services with no outbound connection defined"
            # "Get all projects with their basic details",
            # "Find all real-time integrations with high data volume",
            # "Count integrations by technology",
            # "Find integrations with encryption and security policies",
            # "Find reusable integrations and their usage patterns",
            # "Find employees who earn more than $70,000",
            # "Identify potentially problematic integration patterns",
            # "Get project team contacts for coordination",
            # "Analyze data format transformations across integrations",
            # "Monitor service deployment and build status",
            # "Analyze integration distribution by region and industry",
            # "Analyze connection usage patterns - get connections and their services"
        # ]
        
        # print("🔧 Testing with example queries:")
        # print("-" * 40)
        
        # for query in example_queries:
        #     print(f"\nQ: {query}")
        #     result = chatbot.query(query)
        #     explanation = chatbot.explain_sql(result['result']) if result["success"] else "N/A"
        #     if result["success"]:
                # print(f"✅ SQL Query: {result.get('sql_query', 'N/A')}")
                # print(f"✅ SQL Result: {result['result']}")
        #         print(f"Summary: {result['summary']}")
        #         print(f"Explanation: {explanation}")
        #     else:
        #         print(f"Error: {result['error']}")
        
        # print("\n" + "="*60)
        
        # Start interactive chat
        chatbot.chat()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error initializing chatbot: {e}")

if __name__ == "__main__":
    main()
