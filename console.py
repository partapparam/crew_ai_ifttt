import os
from dotenv import load_dotenv
from langchain_community.tools.ifttt import IFTTTWebhook
from crewai import Agent, Task, Crew, Process
from crewai.task import TaskOutput
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL_NAME")
IFTTT_KEY = os.getenv('IFTTT_KEY')
os.environ["OPENAI_MODEL_NAME"]= openai_model
os.environ["OPENAI_API_KEY"]= openai_key


# URL for If this than that
url = f"https://maker.ifttt.com/trigger/test_crew_ai/json/with/key/{IFTTT_KEY}"


ifttt_tool = IFTTTWebhook(name="test_crew_ai", description="""Useful for sending emails with data.
                When invoking the test_crew_ai.run() method, the parameter tool_input is required and should be of type string.

                For example: tool_input="[string]"
            """, url=url)
search_tool = DuckDuckGoSearchRun()

researcher = Agent(
  role='Senior software sales',
  goal="""craft a story about sales.""",
  backstory="""You are a Senior software sales account executive and want to sell your sales tool to HVAC contractors. """,
  verbose=True,
  allow_delegation=False,
)

writer = Agent(
  role='Data Analyst',
  goal="""Spell check the story and send it to the IFttt webhook tool to email to Param Singh at param@customer.ai'
    """,
  backstory="""You are a renowned data analyst and are in charge of formatting our pipeline for our sales team.""",
  verbose=True,
  allow_delegation=False,
)

def callback_function(output: TaskOutput):
    # Do something after the task is completed
    # Example: Send an email to the manager
    print(f"""
        Task completed!
        Task: {output.description}
        Output: {output.raw_output}
    """)

task1 = Task(
  description="""write a story'
""",
  expected_output='a story with 10 sentences',
  callback=callback_function,
  agent=researcher,
  tools=[search_tool]
)

task2 = Task(
  description="""Take the story and email it using the test_crew_ai tool. 
    You must call this tool the following parameter:
                tool_input='[story]'
    if there is an error, run the command again. 
    """,
  agent=writer,
  tools=[ifttt_tool],
  callback=callback_function,
  context=[task1]
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2
)
result = crew.kickoff()

print("######################")
print(result)

