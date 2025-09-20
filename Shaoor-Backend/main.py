import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware

from notion_agent import run_notion_query, make_notion_agent
from jira_agent import run_jira_query, make_jira_agent
from phi.agent import Agent
from phi.model.openai import OpenAIChat


load_dotenv()

# Ensure OpenAI API key is available globally
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment or .env file")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # make sure LangChain/LangGraph can see it

app = FastAPI()

# global variables
client = None
tools = []
agent = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # Run the data.py script to refresh notion_pages.csv
    import runpy
    try:
        runpy.run_path("data.py")
    except Exception as ex:
        print(f"Error running data.py: {ex}")

    global client, tools, agent
    try:
        client = MultiServerMCPClient(
            {
                "mcp-atlassian": {
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",
                        "--rm",
                        "-e", "CONFLUENCE_URL",
                        "-e", "CONFLUENCE_USERNAME",
                        "-e", "CONFLUENCE_API_TOKEN",
                        "-e", "JIRA_URL",
                        "-e", "JIRA_USERNAME",
                        "-e", "JIRA_API_TOKEN",
                        "ghcr.io/sooperset/mcp-atlassian:latest"
                    ],
                    "env": {
                        "CONFLUENCE_URL": os.getenv("CONFLUENCE_URL"),
                        "CONFLUENCE_USERNAME": os.getenv("CONFLUENCE_USERNAME"),
                        "CONFLUENCE_API_TOKEN": os.getenv("CONFLUENCE_API_TOKEN"),
                        "JIRA_URL": os.getenv("JIRA_URL"),
                        "JIRA_USERNAME": os.getenv("JIRA_USERNAME"),
                        "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN")
                    },
                    "transport": "stdio",
                },
                "mcp-notion": {
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",
                        "--rm",
                        "-e", "OPENAPI_MCP_HEADERS",
                        "mcp/notion"
                    ],
                    "env": {
                        "OPENAPI_MCP_HEADERS": os.getenv("OPENAPI_MCP_HEADERS")
                        # e.g., '{"Authorization":"Bearer ntn_XXXX...","Notion-Version":"2022-06-28"}'
                    },
                    "transport": "stdio",
                }
            }
        )

        tools = await client.get_tools()
        agent = create_react_agent('gpt-4o', tools=tools)
        print(f"Successfully initialized with {len(tools)} tools")
    except Exception as ex:
        print(ex)


class ChatInput(BaseModel):
    message: str


class AgentChatInput(BaseModel):
    message: str
    tone: str
    prompt: str


from typing import List, Dict, Any

class SourceInfo(BaseModel):
    agent: str
    sources: List[str]

class TeamChatOutput(BaseModel):
    responses: Dict[str, str]
    sources: List[SourceInfo]

class ChatOutput(BaseModel):
    response: str


@app.post("/chat", response_model=ChatOutput)
async def chat_endpoint(chat_input: ChatInput):
    try:
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=chat_input.message)]}
        )
        ai_response_message = response["messages"][-1].content
        print(f"AI message {ai_response_message}")
        return ChatOutput(response=ai_response_message)
    except Exception as ex:
        print(ex)

@app.post("/chat_agent", response_model=ChatOutput)
async def chat_endpoint(chat_input: AgentChatInput):
    try:
        message = chat_input.message + ", Use Tone: "+chat_input.tone+", Your Role and Responsibilities: "+chat_input.prompt
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]}
        )
        ai_response_message = response["messages"][-1].content
        print(f"AI message {ai_response_message}")
        return ChatOutput(response=ai_response_message)
    except Exception as ex:
        print(ex)

# --- New Team Chat Endpoint ---

# --- Multi-Agent Team Implementation ---
from phi.agent import Agent as PhiAgent

import importlib


@app.post("/team_chat", response_model=TeamChatOutput)
async def team_chat_endpoint(chat_input: ChatInput):
    try:
        # Instantiate individual agents
        notion_agent = make_notion_agent()
        jira_agent = make_jira_agent()

        # Build a team agent
        team_instructions = [
            "You are leading a team of specialized agents to provide comprehensive answers combining project management and knowledge base information",
            "Coordinate between the Notion Knowledge Specialist and Jira Project Management Specialist to provide complete responses",
            "When the user asks about projects, tasks, or work items, consult the Jira specialist for current status and the Notion specialist for relevant documentation",
            "For strategic questions, roadmaps, or procedures, prioritize the Notion specialist while getting context from Jira about current implementation status",
            "Always provide a synthesis of information from both sources when relevant",
            "Include sources and references from both systems in your final response",
            "Use clear formatting with headers, bullet points, and tables when presenting information",
            "If information from one system contradicts the other, highlight the discrepancy and explain the context",
            "Prioritize actionable insights and next steps in your responses",
            "When information is incomplete from one source, explicitly leverage the other source to fill gaps"
        ]
        team_agent = Agent(
            name="Integrated Workspace Assistant",
            role="Team leader coordinating specialized agents to provide comprehensive workspace insights",
            description="You are an expert workspace assistant that coordinates between project management (Jira) and knowledge management (Notion) specialists to provide comprehensive, actionable insights about projects, tasks, documentation, and organizational processes.",
            model=OpenAIChat(id="gpt-4o"),
            team=[notion_agent, jira_agent],
            instructions=team_instructions,
            markdown=True,
            show_tool_calls=True,
            add_history_to_messages=True,
            num_history_responses=3,
            add_datetime_to_instructions=True,
            prevent_hallucinations=True,
            add_transfer_instructions=True
        )

        # Run the team agent and capture the response
        response = team_agent.run(chat_input.message, stream=False)

        # Extract content
        content = response.content

        print(f"Team agent response: {content}")

        # Optionally extract other fields like tools used, context, etc.
        # e.g., response.messages, response.context, etc.
        sources = []
        if hasattr(response, 'context') and response.context is not None:
            # If context contains source info, extract
            # This is illustrative; you’ll need to adapt depending how your agents/tooling store source info
            for ctx in response.context:
                # If ctx has a “source” or similar field
                src = getattr(ctx, "source", None)
                if src:
                    sources.append(src)

        # Return structured output
        return TeamChatOutput(responses={"team": str(content)}, sources=sources)

    except Exception as ex:
        # For debugging/logging
        print("Error in /team_chat:", ex)
        raise HTTPException(status_code=500, detail=str(ex))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)