import os
from dotenv import load_dotenv
from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langgraph.graph import StateGraph, END

from tools import all_tools, save_to_txt

# Load environment variables for API keys
load_dotenv(dotenv_path='/home/mpavia/.env')
# os.environ["OPENAI_API_KEY"] = "your_key_here" # Uncomment and replace if not using .env

class Paper(BaseModel):
    title: str = Field(description="The title of the research paper.")
    authors: list[str] = Field(description="The list of authors of the paper.")
    url: str = Field(description="The URL or DOI link to the paper.")
    summary: str = Field(description="A concise summary of the paper's abstract or key findings.")

class ResearchState(TypedDict):
    query: str 
    papers: List[Paper] 
    analysis: str  
    critique: str 
    report: str

llm = ChatOpenAI(model="gpt-4o")

def create_agent(prompt: ChatPromptTemplate, tools: list = None):
    if tools is None:
        return prompt | llm
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

search_agent = create_agent(
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You are an expert research assistant. Your goal is to find relevant scientific papers for a given query."),
        ("system", "Use the provided tools to search for papers. Return a list of the most relevant papers with their titles, authors, URLs, and summaries."),
        ("system", "You must return a list of `Paper` objects. You can return an empty list if no relevant papers are found."),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]),
    tools=all_tools
)

def search_node(state: ResearchState):
    print("--- SEARCHING ---")
    query = state["query"]
    result = search_agent.invoke({"query": query})
    papers_list = result.get('output', [])
    print(f"--- FOUND {len(papers_list)} PAPERS ---")
    return {"papers": papers_list}

analysis_agent = create_agent(
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You are a scientific analysis expert. Your role is to synthesize findings from a list of research papers."),
        ("system", "Read through the summaries of the provided papers and create a cohesive analysis."),
        ("system", "Your analysis should highlight the key findings, methodologies, and main conclusions. Do not add new information or critique the work yet."),
        ("human", "Here are the papers: {papers}"),
    ])
)

def analysis_node(state: ResearchState):
    """The 'Analysis' node. Summarizes and synthesizes the papers."""
    print("--- ANALYZING ---")
    if not state["papers"]:
        return {"analysis": "No papers were found to analyze."}
    
    result = analysis_agent.invoke({"papers": state["papers"]})
    return {"analysis": result.content}

critique_agent = create_agent(
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You are a research critic. Your job is to identify gaps and unanswered questions from a body of research."),
        ("system", "Based on the provided analysis of research papers, identify the following:"),
        ("system", "- Gaps in the current research."),
        ("system", "- Contradictions or conflicting findings between papers."),
        ("system", "- Open questions or areas that require future investigation."),
        ("human", "Here is the analysis of the research: {analysis}"),
    ])
)

def critique_node(state: ResearchState):
    """The 'Critique' node. Identifies gaps and future directions."""
    print("--- CRITIQUING ---")
    result = critique_agent.invoke({"analysis": state["analysis"]})
    return {"critique": result.content}

writer_agent = create_agent(
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You are a scientific writer. Your task is to compose a final, comprehensive research report in a clear, human-readable format."),
        ("system", "Use the provided analysis, critique, and list of papers to structure the report."),
        ("system", "The report should include the following sections: a detailed summary, a section on open questions and future directions, and a list of references."),
        ("human", "Please generate a full research report based on the following information:\n\n"
         "## Research Topic ##\n{query}\n\n"
         "## Synthesized Analysis ##\n{analysis}\n\n"
         "## Identified Gaps & Future Directions ##\n{critique}\n\n"
         "## References ##\n{papers}"),
    ])
)

def writer_node(state: ResearchState):
    """The 'Writer' node. Compiles all information into the final report."""
    print("--- WRITING REPORT ---")
    report = writer_agent.invoke({
        "query": state["query"],
        "analysis": state["analysis"],
        "critique": state["critique"],
        "papers": state["papers"]
    })
    return {"report": report.content}

graph = StateGraph(ResearchState)

graph.add_node("search", search_node)
graph.add_node("analyze", analysis_node)
graph.add_node("critique", critique_node)
graph.add_node("write", writer_node)

graph.set_entry_point("search")
graph.add_edge("search", "analyze")
graph.add_edge("analyze", "critique")
graph.add_edge("critique", "write")
graph.add_edge("write", END)

research_graph = graph.compile()

if __name__ == "__main__":
    query = input("Hello, I'm Athena, your AI research assistant. What topic can I help you with today? ")
    
    final_state = None
    for state_update in research_graph.stream({"query": query}):
        last_completed_node = list(state_update.keys())[-1]
        print(f"Finished '{last_completed_node}' step.")
        final_state = state_update[last_completed_node]

    final_report = final_state.get("report", "No report was generated.")
    
    save_to_txt(data=final_report, filename_prefix="research_report")