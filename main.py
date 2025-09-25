import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, pubmed_tool, arxiv_tool, save_to_txt

# Provide path to .env file
load_dotenv(dotenv_path='/home/mpavia/.env')

#load in your own api
#os.environ["OPENAI_API_KEY"] = "your_key_here"

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

def format_research_for_file(response: ScientificResearchResponse) -> str:
    """
    Formats the structured research response into a clean, human-readable string.
    This replaces the old display_research_summary function.
    """
    report_parts = []
    
    report_parts.append("="*80)
    report_parts.append(f"Research Report: {response.topic}")
    report_parts.append("="*80 + "\n")

    report_parts.append("## Summary ##")
    report_parts.append(response.summary + "\n")

    report_parts.append("## Open Questions & Future Directions ##")
    if not response.unanswered_questions:
        report_parts.append("No specific unanswered questions were identified from the literature.\n")
    else:
        for question in response.unanswered_questions:
            report_parts.append(f"- {question}")
        report_parts.append("\n")

    report_parts.append("## References ##")
    if not response.key_papers:
        report_parts.append("No key papers were identified.\n")
    else:
        for i, paper in enumerate(response.key_papers, 1):
            report_parts.append(f"{i}. {paper.title}")
            report_parts.append(f"   - Authors: {', '.join(paper.authors)}")
            report_parts.append(f"   - Link: {paper.url}")
            report_parts.append(f"   - Summary: {paper.summary}\n")
            
    return "\n".join(report_parts)

llm = ChatOpenAI(model="gpt-4o") # Using gpt-4o as it's a powerful and cost-effective model

parser = PydanticOutputParser(pydantic_object=ScientificResearchResponse)

# The agent is no longer instructed to save the file. Its job is just to return the JSON.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert scientific research assistant. Your goal is to conduct thorough literature reviews.

            1. Always prefer academic sources. Use the `arxiv_tool` and `pubmed_search` tools first. Use the general `search` tool only as a last resort.
            2. Synthesize the information from multiple papers to form a coherent, well-structured summary of the current state of research on the topic.
            3. Identify the key findings, methodologies, and conclusions. If sources conflict, point this out.
            4. Identify what is *not* known. Based on the literature, list the open questions or areas that require further research.
            5. You must wrap your final response in the provided JSON format. Do not provide any other text or explanation.
            
            \n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# The save_tool has been removed from the agent's tools.
tools = [search_tool, pubmed_tool, arxiv_tool] 

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
    if "```json" in output_text:
        output_text = output_text.strip().split("```json\n")[1].split("\n```")[0]
    
    structured_response = parser.parse(output_text)
    human_readable_report = format_research_for_file(structured_response)
    save_to_txt(data=human_readable_report, filename="research_output.txt")
    print("Analysis complete.")
    print(f"   Tools used: {', '.join(structured_response.tools_used)}")
    print("   Full report saved to 'research_output.txt'")

except Exception as e:
    print("\n---")
    print("Error processing the response from the LLM.")
    print(f"   Error: {e}")
    print("   This may happen if the model doesn't follow the JSON format perfectly.")
    print("\nRaw LLM Output:")
    print(raw_response.get("output"))
    print("---")