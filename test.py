from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.chains import create_sql_query_chain
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from typing import Dict, Any
from langchain_community.agent_toolkits import create_sql_agent
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
import re
import json
import os

load_dotenv()

db_uri = os.getenv("DATABASE_URL")



db = SQLDatabase.from_uri(db_uri)

# Initialize Groq LLM instead of OpenAI
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",  # or "llama2-70b-4096"
    api_key=os.getenv("GROQ_API_KEY")
)

# chain = create_sql_query_chain(llm, db)
# response = chain.invoke({"question": "How many users are there. Output only the sql command"})

# if "SELECT" in response:
#     # Extract SQL if needed - this simple approach assumes SQL starts with SELECT
#     sql_to_execute = response[response.find("SELECT"):]
#     # Remove any trailing text that might not be part of the SQL
#     if ";" in sql_to_execute:
#         sql_to_execute = sql_to_execute[:sql_to_execute.find(";")+1]
    
#     print("\nExecuting SQL:", sql_to_execute)
#     result = db.run(sql_to_execute)
#     print(result)
# else:
#     print("Could not extract valid SQL from the response")



# Create the SQLDatabaseChain using the from_llm method
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_intermediate_steps=True)

# # Example usage
# query = "SELECT first_name FROM users LIMIT 1;"
# response = db_chain.invoke({"query": query})
template = '''Answer the question using the tools below. 

Tools:
{tools}

Format:
Question: the input question
Thought: reasoning step
Action: one of [{tool_names}]
Action Input: input to the action
Observation: result of the action
... (Repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: your question answer is ; no input query in final answer

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)


toolkit = SQLDatabaseToolkit(db=db, llm=llm)


agent_executor= create_sql_agent(llm=llm, toolkit=toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


query= "what is my complaint resolved status?"

def run_agent(user_input):
    name_pattern = r"\b(my name is|i am|this is|name is)\b"
    if not re.search(name_pattern, user_input.lower()):
        return "Please provide your name before I can process your request."
    return agent_executor.invoke(user_input)

response = run_agent(query)
print(response)

# from huggingface_hub import InferenceClient

# client = InferenceClient(
#     provider="hf-inference",
#     api_key="copy from env file",
# )

# result = client.text_classification(
#     text="My order arrived damaged, and no one is helping!",
#     model="nlptown/bert-base-multilingual-uncased-sentiment",
# )
# print(result)



# def sentiment_analysis(text): 
#     chat = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="llama-3.3-70b-versatile",
#     temperature=0
# ).bind(response_format={"type": "json_object"})
#     prompt = f"Analyze the sentiment of the following text, find which tone is greater and return the larger tone as a JSON object with keys 'sentiment' with value 'POSITIVE' OR 'NEGATIVE' and key 'confidence' with value 0-100:\n\n\"{text}\""
# # Send request
#     response = chat.invoke([
#     {"role": "system", "content": "You are a sentiment analysis assistant."},
#     {"role": "user", "content": prompt},
# ])
#     response_dict = json.loads(response.content)

#     return response_dict


# text = "My order arrived damaged, and no one is helping!"
# response = sentiment_analysis(text)
# print(response)
