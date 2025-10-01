# A multi-tool, Pydantic-structured research agent built with LangChain


### Create the environment
``` 
python3 -m venv research_agent
```
### Activate it
```
source /home/mpavia/01_enviorments/research_agent/bin/activate
```
### Install dependencies
```
python -m pip install -r requirements.txt
```
### Athena can be run with the following command :
```
python main.py
```


### test question
```
How do you detect outbreaks of Mycobacterium tuberculosis? 
```

### ideas to implement
* extract information from papers not just abstract (RAG pipeline)
* human-in-the-loop for checking papers
* add relfection step (ReflectionAgent) or CritqueAgent?
* Convert output into a JSON format 
* Add in text citations

