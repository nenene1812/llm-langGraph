from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool, Tool
from langchain_openai import ChatOpenAI
from openai import OpenAI
import requests
import os


# Web Search
wrapper = DuckDuckGoSearchAPIWrapper(max_results=10)
duckduckgo_search = DuckDuckGoSearchResults(api_wrapper=wrapper)

# Image Gen
@tool
def generate_dalle_image(prompt: str):
    """
    Function to generate an image using OpenAI's DALL-E model.

    Parameters:
    - prompt (str): The prompt to generate the image.

    Returns:
    - str: The URL of the generated image.
    """
    client = OpenAI()

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
     )

    return response.data[0].url

# Python Execution
python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be valid python commands. ALWAYS print ANY results out with `print(...)`",
    func=python_repl.run,
)

# Define list of tools
tools=[ duckduckgo_search, generate_dalle_image, repl_tool]

# Instantiate LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

system_prompt="Ensure your generation of the image URL is exact, add an extra space after it to ensure no new lines mess it up. Always print executed python statements for logging."

# Main Graph
call_gpt3 = create_react_agent(llm, tools, messages_modifier=system_prompt)