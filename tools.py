import os
from datetime import datetime
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.utilities import PubMedAPIWrapper

def save_to_txt(data: str, filename: str ):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

save_tool = Tool.from_function(
    name="save_text_to_file",
    func=save_to_txt,
    description="Save structured research data to a text file. The input should be a dictionary with 'data' and 'filename' keys."
)

save_tool = Tool.from_function(
    name="save_text_to_file",
    func=save_to_txt,
    description="Save structured research data to a text file. The input should be a dictionary with 'data' and 'filename' keys."
)

pubmed_api_wrapper = PubMedAPIWrapper(ncbi_api_key=os.getenv("NCBI_API_KEY"))

pubmed_tool = Tool(
    name="pubmed_search",
    func=pubmed_api_wrapper.run,
    description="A wrapper around PubMed. Use this for questions about medicine, biology, health, and biomedical research papers. Input should be a search query."
)

search_tool = DuckDuckGoSearchRun()

arxiv_tool = ArxivQueryRun()