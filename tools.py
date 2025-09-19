from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain.tools import Tool
from datetime import datetime

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

save_tool = Tool(
    name = "save_text_to_file",
    func=save_to_txt,
    description="Save structured research data to a text file"
)

search_tool = DuckDuckGoSearchRun()

pubmed_tool = PubmedQueryRun()

arxiv_tool = ArxivQueryRun()