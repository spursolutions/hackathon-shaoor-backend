import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv()

async def test_jira_server():
    print("Testing JIRA Server...")
    client = MultiServerMCPClient({
        "mcp-atlassian": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "-e", "JIRA_URL",
                "-e", "JIRA_USERNAME", 
                "-e", "JIRA_API_TOKEN",
                "ghcr.io/sooperset/mcp-atlassian:latest"
            ],
            "env": {
                "JIRA_URL": os.getenv("JIRA_URL"),
                "JIRA_USERNAME": os.getenv("JIRA_USERNAME"),
                "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN")
            },
            "transport": "stdio",
        }
    })
    
    try:
        #await client.initialize()
        tools = await client.get_tools()
        print(f"JIRA server tools: {[tool.name for tool in tools]}")

        agent = create_react_agent("gpt-4o", tools=tools)

        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "What are the tasks for the current sprint?"}]}
        )

        print(response["messages"][-1].content)

        return True
    except Exception as e:
        print(f"JIRA server error: {e}")
        return False
    finally:
        try:
            await client.close()
        except:
            pass

async def main():
    jira_ok = await test_jira_server()
    
    print(f"\nResults:")
    print(f"JIRA server: {'✓' if jira_ok else '✗'}")

if __name__ == "__main__":
    asyncio.run(main())