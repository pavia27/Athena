import os
from datetime import datetime
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.utilities import PubMedAPIWrapper

def save_to_txt(data: str, filename_prefix: str = "research_report"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.txt"
    report_content = f"--- Research Report ---\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{data}"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_content)

search_tool = DuckDuckGoSearchRun()
arxiv_tool = ArxivQueryRun()

pubmed_api_wrapper = PubMedAPIWrapper(ncbi_api_key=os.getenv("NCBI_API_KEY"))
pubmed_tool = Tool(
    name="pubmed_search",
    func=pubmed_api_wrapper.run,
    description="A wrapper around PubMed. Use for questions on medicine, biology, health, and biomedical research. Input should be a search query."
)

all_tools = [search_tool, pubmed_tool, arxiv_tool]