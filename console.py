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


# URL for IFTTT
url = f"https://maker.ifttt.com/trigger/test_crew_ai/json/with/key/{IFTTT_KEY}"

# Define IFTTT tool
ifttt_tool = IFTTTWebhook(name="test_crew_ai", description="""Useful for sending emails with data.
                When invoking the test_crew_ai.run() method, the parameter tool_input is required and should be of type string.\n
                          ----------\n
                        tool_input="{chunk}"
            """, url=url)
search_tool = DuckDuckGoSearchRun()


account_exec = Agent(
  role='Senior software account executive',
  goal="""Search the web and find prospects with contact information that can benefit from your software, ServiceTitan.""",
  backstory="""You are a Senior software sales account executive and want to sell your software to HVAC companies in Los Angeles.""",
  verbose=True,
  allow_delegation=False,
)


data_analyst = Agent(
  role='Analyst',
  goal="""Parse the returned prospects list and organize it in JSON format with name, phone number, and email. Send this list to the IFTTT webhook.
    """,
  backstory="""You are a renowned data analyst and are in charge of formatting our pipeline for our sales team. You ensure there an no duplicates and double check that the accounts being added are verified.""",
  verbose=True,
  allow_delegation=False,
)

def callback_function(output: TaskOutput):
    # Do something after a task is completed
    print(f"""
        Task completed!
        Task: {output.description}
        Output: {output.raw_output}
    """)


task1 = Task(
  description="""Find a list of 10 prospects, each with a person to contact, phone, and email.
""",
  expected_output="A list of prospects",
  callback=callback_function,
  agent=account_exec,
  tools=[search_tool]
)

task2 = Task(
  description="""Take the list of prospects, parse it into JSON format and email it using the test_crew_ai tool. 
    You must call this tool the following parameter:\n
                tool_input='{list_of_prospects}'
              \n
    if there is an error, run the command again. 
    """,
  agent=data_analyst,
  tools=[ifttt_tool],
  callback=callback_function,
  context=[task1]
)


# Instantiate your crew with a sequential process
crew = Crew(
  agents=[account_exec, data_analyst],
  tasks=[task1, task2],
  process=Process.sequential,
  verbose=2
)
result = crew.kickoff()

print("######################")
print(result)

