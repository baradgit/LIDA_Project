# Importing the necessary libraries
# pandas - for handling data in DataFrame structure
# sqlite3 - for interacting with SQLite databases
# difflib - for comparing and finding closest matches to strings
# openai - for OpenAI's API interactions
# langchain and langchain_experimental - for agent-based LLMs with CSV handling
# lida - used for text generation and visualizations
# dotenv - for managing environment variables
# PIL - Python Imaging Library to handle image manipulation
# io and base64 - for handling image conversions
import pandas as pd
import sqlite3
import difflib
from openai import OpenAI
from langchain.llms import OpenAI as LangOpenAI
from langchain_experimental.agents import create_csv_agent
from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
import os

# Load environment variables, particularly the OpenAI API key. This ensures
# the sensitive key is never hardcoded but is accessed securely from the environment.
# load_dotenv()  # Uncomment if .env file is used
# openai_api_key = os.getenv("OPENAI_API_KEY")

# This function converts a base64-encoded string back into an image.
# It decodes the base64 data, then opens it as an image using PIL.
def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

# This function initializes LIDA, a tool that manages text generation.
# It sets up the manager using the OpenAI key.
def initialize_lida(api_key):
    return Manager(text_gen=llm("openai", api_key=api_key))

# Function to generate visualizations based on a user's query.
# It first summarizes the given file and then generates visual charts using the Seaborn library.
def generate_visualization(file_path, user_query, api_key):
    # Configuring text generation settings for LIDA
    textgen_config = TextGenerationConfig(n=1, temperature=0.2, model="gpt-3.5-turbo", use_cache=True)
    # Initialize LIDA Manager
    lida = Manager(text_gen=llm("openai", api_key=api_key))
    # Generate a summary of the file based on the default method
    summary = lida.summarize(file_path, summary_method="default", textgen_config=textgen_config)
    # Create visualizations from the summary and based on user goal/query
    charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config, library="seaborn")

     # If a chart is generated, decode the image from its base64 format
    if charts:
        img_base64_string = charts[0].raster
        img = base64_to_image(img_base64_string)
        return img
    else:
        return None

# Function to detect if the user's query is asking for any form of visualization.
# Checks for keywords like 'plot', 'chart', or 'visualize' in the query.
def is_visualization_query(query):
    keywords = ["plot", "chart", "graph", "visualize", "visualization", "visual"]
    return any(keyword in query.lower() for keyword in keywords)

# Function to detect if the user is asking for table-related information.
# Checks for keywords like 'table', 'list', or 'structured'.
def is_table_query(query):
    keywords = ["table", "structured", "draw table", "create table", "show table", "list"]
    return any(keyword in query.lower() for keyword in keywords)

# Column names that we expect in our dataset.
COLUMN_NAMES = ['UDI', 'Product_ID', 'Type', 'Air_temperature__K_', 'Process_temperature__K_',
                'Rotational_speed__rpm_', 'Torque__Nm_', 'Tool_wear__min_', 'Machine_failure',
                'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Function to correct any misspelled column names in user input by comparing them to the actual column names.
# Uses difflib to find the closest match.
def correct_column_name(user_input_column):
    matches = difflib.get_close_matches(user_input_column, COLUMN_NAMES, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    else:
        return user_input_column

# This function stores a CSV file into an SQLite database.
# The CSV is read into a pandas DataFrame and then stored into a local SQLite database.
def store_csv_in_db(csv_file):
    df = pd.read_csv(csv_file)
    conn = sqlite3.connect("local_database.db")
    df.to_sql('data_table', conn, if_exists='replace', index=False)
    conn.close()
    return df

# Function to generate an SQL query based on user input.
# It corrects column names where necessary and uses OpenAI to generate the SQL query.
def generate_sql_query(user_input, api_key):
    words = user_input.split()
    corrected_words = [correct_column_name(word) for word in words]
    corrected_input = ' '.join(corrected_words)
    client = OpenAI(api_key=api_key)
    prompt = (
        f"Generate an SQL query based on this user request: '{corrected_input}'. "
        f"Use the table name 'data_table' in the query. "
        f"TWF = Tool Wear Failure, HDF = Heat Dissipation Failure, PWF = Power Failure, OSF = Overstrain Failure, RNF = Random Failures. "
        f"Talking about failure or failed always mention='1' and if not failed means always mention='0'."
        f"Note: Ensure that the SQL query string does not include ```sql. It should only contain valid SQL syntax not even extra unnecessary character"
    )
     # API call to OpenAI's model for SQL generation
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.1
    )
    # Extract the SQL query from the response
    sql_query = response.choices[0].message.content.strip()
    return sql_query

# Function to run the generated SQL query on the SQLite database.
# Executes the query and returns the result as a pandas DataFrame.
def run_sql_query(sql_query):
    conn = sqlite3.connect("local_database.db")
    try:
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return result_df
    except Exception as e:
        conn.close()
        return str(e)

# Function to split a complex user query into three distinct parts: Visualization, Table, and Summary.
# It generates SQL queries or Python code as required and provides appropriate summaries.
def split_query_into_parts(user_query, api_key):
    client = OpenAI(api_key=api_key)
    prompt = (
        f"Analyze the user's query: '{user_query}', and break it down into three distinct sections: Visualization, Table, and Summary. "
        f"Ensure each section is correctly handled based on the dataset. The available columns from the dataset are: "
        f"['UDI', 'Product_ID', 'Type', 'Air_temperature__K_', 'Process_temperature__K_', "
        f"'Rotational_speed__rpm_', 'Torque__Nm_', 'Tool_wear__min_', 'Machine_failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']. "
        f"TWF = Tool Wear Failure, HDF = Heat Dissipation Failure, PWF = Power Failure, OSF = Overstrain Failure, RNF = Random Failures. "
        f"Talking about failure or failed always mention='1' and if not failed means always mention='0'. "
        f"Here are detailed instructions for each section: "
        f"1) **Visualization Request**: Detect phrases indicating the user wants a 'chart', 'graph','plot' or any visual representation of the data. Look for words like 'show a graph', 'plot', 'visualize', 'bar chart', 'scatter plot', etc. "
        f"Also handle implicit requests like 'compare' 'univariate' 'bivariate' 'Multivariate' use mention any you add any columns use column names mentioned , which suggests the user wants a plot. "
        f"univariate means plots has only has one column ,bivariate means 2 columns ,Multivariate means more than 2 columns "
        f"'plot', 'chart', 'graph', 'visualize', 'visualization, 'visual' if this key words not there user is not requestion for visualization ignore visualization "
        f"check if user is asking  multiple plot  create detail prompt of multiple plots different plots with type of plot plot means visualization"
        f"If multiple variables are mentioned, infer the correct type of chart. Provide the result in the format: 'Visualization: <description of chart>'. "
        f"2) **Table Request (SQL Query or Python Code)**: For structured data requests, create a valid SQL query or Python code to match the user's request. "
        f"Pay attention to words like 'list', 'show table', 'retrieve', 'filter', 'order by', etc. For simple requests, generate an SQL query. "
        f"For more complex queries involving calculations, multiple filters, or conditions, generate a Python code snippet using Pandas. "
        f"Make sure to handle advanced queries requiring operations that SQL cannot handle alone. Provide the result in the format: 'Table: <SQL query or Python code>'. "
        f"If no table is requested, return 'Table: None'. "
        f"3) **Summary Request**: Look for phrases indicating the user wants a summary, analysis, or statistical insight, such as 'summarize', 'describe', 'analyze', 'mean', 'median', 'standard deviation', etc. "
        f"Generate a text-based summary for such requests. For example, 'Summarize the relationship between Air_temperature__K_ and Process_temperature__K_' implies a statistical explanation. "
        f"Provide the result in the format: 'Summary: <text-based summary>'. If no summary is requested, return 'Summary: None'. "
        f"4) **Handling Multiple Requests**: If the user asks for more than one of the three sections (visualization, table, summary), generate the output for each as required. "
        f"If the user specifies only one section (e.g., 'only show me a table'), ensure the other sections are ignored. For ambiguous or complex queries, intelligently split the request and handle each part appropriately. "
        f"5) **Handling Complex Queries**: If the query is ambiguous or complex, split the operations and handle them individually. For instance, if the user asks for 'average temperature and visualize it over time', return both a summary and a relevant chart. "
        f"For queries beyond SQL's capability (e.g., involving advanced calculations or multiple conditions), generate Python code to handle the request. "
        f"Return the output in the following structured format: 1) Visualization: <description of chart> 2) Table: <SQL query or Python code> 3) Summary: <text-based summary>. "
        f"If any section does not apply, return 'None' for that section."
        f"Talking about failure or failed always mention='1' and if not failed means always mention='0'. "
        
    )
      # API call to OpenAI's model for splitting the query into parts
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.2
    )
     # Extract the structured response
    divided_query = response.choices[0].message.content.strip()
    return divided_query
