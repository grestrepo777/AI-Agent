from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool


# Loads env (environment variables) so we have all the credentials we need
load_dotenv() 

# Create a class
class ResearchReponse(BaseModel): 
    # Generate a topic of type string 
    topic: str
    # Generate a summary of type string
    summary: str
    # Have sources be a list of strings
    sources: list[str]
    # tools used be a list of strings
    tools: list[str]

# Create a parser
parser = PydanticOutputParser(pydantic_object=ResearchReponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            You are a research asssistant that will help generate a researh paper. 
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ), 
        ("placeholder", "{chat_history}"), 
        ("human", "{query}"), 
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Set up an LLM
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# define tools
tools = [search_tool, wiki_tool, save_tool]

# Create and set up agent
agent = create_tool_calling_agent(
    llm = llm, 
    prompt = prompt, 
    tools = tools
)

agent_executor = AgentExecutor(agent= agent, tools = tools, verbose = True)

#get the query from the user
query = input("Enter your research topic: ")

raw_response = agent_executor.invoke({"query": query})

# Output parsing
structured_response = parser.parse(raw_response.get("output")[0]["text"])
print(structured_response)