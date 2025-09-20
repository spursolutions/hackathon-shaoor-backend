# agents/jira_agent.py

from phi.agent import Agent
from phi.tools.jira_tools import JiraTools
from dotenv import load_dotenv
import os

load_dotenv()

JIRA_SERVER_URL = os.getenv("JIRA_URL")
JIRA_USERNAME = os.getenv("JIRA_USERNAME")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

def make_jira_agent():
    agent = Agent(
        name="Jira Project Management Specialist", 
        role="Expert project management and task tracking specialist focused on Jira issues, sprints, and project workflows",
        description="You are a specialized project management assistant that excels at retrieving and analyzing information from Jira. You understand project lifecycles, sprint planning, issue tracking, and can provide insights about task priorities, project status, and team workloads.",
        tools=[JiraTools(JIRA_SERVER_URL, JIRA_USERNAME, JIRA_API_TOKEN)],
        instructions=[
            "Always include project restrictions in your queries to avoid unbounded JQL searches",
            "When searching for issues, provide context about project, status, assignee, or timeline",
            "Analyze and summarize issue information in a clear, actionable format",
            "When discussing priorities, explain the reasoning behind prioritization",
            "Include relevant issue keys, summaries, and statuses in your responses",
            "For sprint-related queries, provide comprehensive sprint information including progress",
            "When discussing workloads, consider both current assignments and upcoming tasks",
            "Always provide source links to specific Jira issues when referencing them",
            "If queries are too broad, suggest more specific search criteria",
            "Focus on actionable insights for project management decisions"
        ],
        markdown=True,
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        prevent_hallucinations=True
    )
    return agent

def safe_jira_query(user_query: str, default_project: str = "AWS Migration") -> str:
    if "project" not in user_query.lower():
        return f"{user_query} (in project {default_project})"
    return user_query

def run_jira_query(query: str):
    agent = make_jira_agent()
    safe_query = safe_jira_query(query)
    response = agent.run(safe_query, stream=False, markdown=True)
    return response
