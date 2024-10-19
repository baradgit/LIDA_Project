# Import necessary modules for building the application
import streamlit as st  # Streamlit is used for creating the web-based interface
from openai import OpenAI  # OpenAI client for interacting with the API
import pandas as pd  # Pandas for handling and manipulating CSV data
import os  # OS module for environment variable operations
from chat_bot import initialize_lida, store_csv_in_db, generate_sql_query, run_sql_query, generate_visualization, split_query_into_parts, COLUMN_NAMES, is_visualization_query, is_table_query
from langchain.llms import OpenAI as LangOpenAI  # LangChain for more structured queries and agents
from langchain_experimental.agents import create_csv_agent  # CSV agent creation for answering queries from CSV data

# Set page configurations for the Streamlit app
st.set_page_config(page_title="CSV Agent Application", layout="wide")  # Page title and layout settings

# Adding some custom styles to beautify the UI
st.markdown(
    """
    <style>
    .main { background-image: url('https://wallpaper.dog/large/5452685.jpg'); background-size: cover; background-color: black; }
    .centered { text-align: center; }
    .header { font-size: 36px; font-weight: bold; color: #ffffff; }
    .subheader { font-size: 28px; font-weight: bold; color: #ffffff; }
    .text { font-size: 18px; font-weight: bold; color: #ffffff; }
    .example-question { font-size: 18px; font-weight: bold; color: #ffffff; margin-bottom: 5px; }
    .column-names { font-size: 18px; font-weight: bold; color: #ffffff; background-color: #4F8BF9; padding: 10px; border-radius: 10px; }
    .highlight-summary { background-color: #333333; padding: 15px; border-left: 5px solid #4F8BF9; font-size: 18px; font-weight: bold; color: #ffffff; }
    textarea { background-color: #2C2C2C; color: #ffffff; font-size: 18px; font-weight: bold; padding: 15px; border-radius: 10px; border: 2px solid #4F8BF9; }
    button { font-size: 18px; font-weight: bold; color: #ffffff; }
    .stAlert { color: #ffffff; }
    .stMarkdown { color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True  # Ensures the HTML and CSS code is executed for custom styling
)

# Displaying the app's title with a custom header style
st.markdown("<h1 class='centered header'>Predictive Maintenance Chatbot</h1>", unsafe_allow_html=True)

# Input field for the user to provide their OpenAI API key securely
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Ensure the API key is provided before proceeding with further functionalities
if api_key:
   
    client = OpenAI(api_key=api_key)  # Initialize the OpenAI client with the provided API key
    lida = initialize_lida(api_key)  # Initialize the LIDA manager for text generation tasks

    # Predefined file path for the CSV file to be used for the chatbot's data source
    file_path = "ai4i2020.csv"

    if file_path:
        st.write(f"Using file: {file_path}")  # Display which CSV file is being used
        
        df = store_csv_in_db(file_path)  # Store the CSV data into a local SQLite database
        # Inform the user about the dataset and what the chatbot can help with
        st.markdown(f"<div class='column-names'>This chatbot, built on the AI4I 2020 Predictive Maintenance Dataset, helps predict machine failures based on operational data like temperature, speed, torque, and tool wear. The chatbot allows users to query for visualizations, tables, and summaries using natural language input. It leverages LangChain to interpret queries and uses SQLite for data storage. The chatbot includes error correction for column names and generates visualizations using the LIDA library for charts. The user experience is streamlined through Streamlit, with continuous conversation capabilities, making the system efficient for predictive maintenance tasks.</div>", unsafe_allow_html=True)

        # Some example questions to guide the user on how to ask the chatbot for information
        EXAMPLE_QUESTIONS = [
            "1. How many power failed products what is the average air temperature of only power failed products?",
            "2. The count of products got machine failure and for that products what is the range of the air temperature?",
            "3. Which failure type seems to occur more times under high air temperature conditions?",
            "4. Provide a summary of failures by failure type (TWF, HDF, PWF, OSF, RNF) and the associated average operating conditions.",
            "5. How many machines experienced power failure?",
            "6. What is the average air temperature for products with power failure?",
            "7. Table: product ID, air temperature, and rotational speed for products with machine failure?",
            "8. How many products have both tool wear failure and power failure?",
            "9. What is the range (MIN and MAX) of air temperature for machines with machine failure (Machine_failure)?",
            "10. What is the total number of machines that experienced each type of failure (TWF, HDF, PWF, OSF, RNF)?",
            "11. Plot a histogram of the air temperature (Air_temperature__K_) for machines with machine failure (Machine_failure).",
            "12. Create a bar chart comparing the average tool wear time (Tool_wear__min_) for failed machines (Machine_failure) vs. non-failed machines (Machine_failure = 0).",
            "13. Show a bar plot of the number of machines with rotational speed above 2000 rpm for each failure type.",
            "14. Create a histogram showing the distribution of torque (Torque__Nm_) for machines with overstrain failure (OSF).",
            "15. Create a scatter plot of air temperature (Air_temperature__K_) versus process temperature (Process_temperature__K_) for machines that experienced machine failure (Machine_failure = 1).",
        ]

        # Display the example questions on the app interface
        st.markdown("<h2 class='subheader'>Example Questions</h2>", unsafe_allow_html=True)
        for question in EXAMPLE_QUESTIONS:
            st.markdown(f"<div class='example-question'>{question}</div>", unsafe_allow_html=True)

        # Display available column names from the dataset to guide users
        st.markdown("<h2 class='subheader'>Available Column Names</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='column-names'>UDI, Product_ID, Type, Air_temperature__K_, Process_temperature__K_, Rotational_speed__rpm_, Torque__Nm_, Tool_wear__min_, Machine_failure,TWF (Tool Wear Failure), HDF (Heat Dissipation Failure), PWF (Power Failure), OSF (Overstrain Failure), RNF (Random Failures).</div>", unsafe_allow_html=True)

        # Warning to the user about entering a correct API key
        st.markdown("**If You get authentication error api key is wrong!**")
        st.markdown("<h2 class='subheader'>Ask a Question</h2>", unsafe_allow_html=True)
        # Text input for the user to type their question or query
        query = st.text_area("Ask for a visualization, table, and summary in a single query:")

        # Execute when the submit button is clicked
        if st.button("Submit"):
            # Split the user's query into parts (visualization, table, summary)
            divided_queries = split_query_into_parts(query, api_key)

            # Check if the query contains any of the sections: Visualization, Table, or Summary
            if 'Visualization:' in divided_queries and 'Table:' in divided_queries and 'Summary:' in divided_queries:
                # Extract and clean the different parts of the query
                visualization_query = divided_queries.split('Visualization:')[1].split('Table:')[0].strip()
                table_query = divided_queries.split('Table:')[1].split('Summary:')[0].strip()
                summary_query = divided_queries.split('Summary:')[1].strip()
                visualization_query = "visualization:" + visualization_query

                # Generate a visualization if requested
                if is_visualization_query(query) or ('Visualization' in divided_queries and 'None' not in visualization_query):
                    st.markdown("<h2 class='subheader'>Generating Visualization...</h2>", unsafe_allow_html=True)
                    img = generate_visualization(file_path, visualization_query, api_key)
                    if img:
                        st.image(img)  # Display the generated image
                    else:
                        st.error("No chart was generated for the visualization query.")

                # Generate a table if requested
                if is_table_query(query) or ('Table' in divided_queries and 'None' not in table_query):
                    st.markdown("<h2 class='subheader'>Generating Table...</h2>", unsafe_allow_html=True)
                    sql_query = generate_sql_query(table_query, api_key)
                    try:
                        result_df = run_sql_query(sql_query)  # Execute the SQL query and get results
    
                        if isinstance(result_df, pd.DataFrame):
                            st.dataframe(result_df)  # Display the table as a DataFrame
                        else:
                            raise ValueError("rephrase the query try starting with table:")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")        

                # Generate a summary if requested
                if 'Summary' in divided_queries and 'None' not in summary_query:
                    st.markdown("<h2 class='subheader'>Fetching Summary...</h2>", unsafe_allow_html=True)

                    # Create a CSV agent using LangChain to handle CSV-based queries
                    agent = create_csv_agent(
                        LangOpenAI(temperature=0, api_key=api_key),
                        file_path,
                        verbose=True,
                        allow_dangerous_code=True
                    )

                    # Try executing the summary query
                    try:
                        result = agent.invoke({"input": summary_query})
                        summary_output = result["output"]

                        # Display the generated summary in the UI
                        st.markdown(f"<div class='highlight-summary'>{summary_output}</div>", unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
            else:
                st.error("Please try a clearer query.")  # Error message if the query is unclear
else:
    st.error("Please enter your OpenAI API key to proceed.")  # Error if no API key is provided
