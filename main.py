from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, save_tool, pubmed_tool, arxiv_tool

#provide path to .env with API or 
load_dotenv(dotenv_path='/home/mpavia/.env')

#load in your own api
#OPENAI_API_KEY="your_key_here"

class Paper(BaseModel):
    title: str = Field(description="The title of the research paper.")
    authors: list[str] = Field(description="The list of authors of the paper.")
    url: str = Field(description="The URL or DOI link to the paper.")
    summary: str = Field(description="A concise summary of the paper's abstract or key findings.")

class ScientificResearchResponse(BaseModel):
    topic: str = Field(description="The primary research topic.")
    summary: str = Field(description="A detailed synthesis of the findings from all sources, including methodologies and conclusions.")
    key_papers: list[Paper] = Field(description="A list of the most relevant papers found.")
    unanswered_questions: list[str] = Field(description="A list of open questions or areas for future research identified from the literature.")
    tools_used: list[str] = Field(description="The list of tools that were used to generate this response.")


llm = ChatOpenAI(model="gpt-4.1") #works gpt-4o

parser = PydanticOutputParser(pydantic_object=ScientificResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert scientific research assistant. Your goal is to conduct thorough literature reviews.

            1. Always prefer academic sources. Use the `arxiv_tool` and `pubmed_search` tools first. Use the general `search` tool only as a last resort for very recent or non-academic topics.
            2. Do not simply return a list of summaries. Synthesize the information from multiple papers to form a coherent, well-structured summary of the current state of research on the topic.
            3. Identify the key findings, methodologies, and conclusions. If sources conflict, point this out.
            4. A crucial part of your role is to identify what is *not* known. Based on the literature, list the open questions or areas that require further research.
            5. You must wrap your final response in the provided JSON format. Do not provide any other text or explanation. This should be saved in a text file with a relvant name.

            
            \n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, save_tool, pubmed_tool, arxiv_tool] 

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

query = input("Hello, I'm Athena, your AI research assistant. What topic can I help you with? ")
raw_response = agent_executor.invoke({"query": query})

try:
    output_text = raw_response.get("output")
    structured_response = parser.parse(output_text)
    print("Structured response:", structured_response)
except Exception as e:
    print("Error parsing response:", e)
    print("Raw Response:", raw_response)
    print("Output text:", raw_response.get("output"))