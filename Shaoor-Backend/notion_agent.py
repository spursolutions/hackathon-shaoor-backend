# agents/notion_agent.py

from phi.knowledge.csv import CSVKnowledgeBase
from phi.vectordb.pgvector import PgVector
from phi.agent import Agent
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment or .env file")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def make_notion_agent():
    knowledge_base = CSVKnowledgeBase(
        path="notion_pages.csv",
        vector_db=PgVector(
            table_name="csv_documents",
            db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        ),
    )
    agent = Agent(
        name="Notion Knowledge Specialist",
        role="Expert knowledge retrieval specialist focused on organizational documentation, procedures, and information stored in Notion databases",
        description="You are a specialized knowledge assistant that excels at searching through Notion databases to find relevant documentation, procedures, guidelines, feature roadmaps, and organizational information. You have deep expertise in understanding context and providing comprehensive answers from knowledge bases.",
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=[
            "Search the Notion knowledge base thoroughly for information relevant to the user's query",
            "Provide detailed, contextual answers based on the documentation found",
            "When referencing information, mention the specific page or database title where the information was found",
            "If multiple relevant documents exist, synthesize information from all sources",
            "Always include source references in your responses",
            "Focus on providing actionable insights and clear explanations",
            "If the query relates to features, roadmaps, or procedures, provide comprehensive details",
            "When information is incomplete, clearly state what additional context might be needed"
        ],
        markdown=True,
        show_tool_calls=True,
        add_datetime_to_instructions=True
    )
    agent.knowledge.load(recreate=True)
    return agent

def run_notion_query(query: str):
    agent = make_notion_agent()
    # returns a RunResponse object
    response = agent.run(query, stream=False)
    return response
