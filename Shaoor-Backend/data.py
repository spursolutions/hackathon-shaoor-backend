import os
import csv
from dotenv import load_dotenv
from notion_client import Client

# Load token
load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_API_KEY")
if not NOTION_TOKEN:
    raise RuntimeError("NOTION_API_KEY is not set in the environment or .env file")

notion = Client(auth=NOTION_TOKEN)

# CSV setup
output_file = "notion_pages.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Database ID", "Database Title", "Page ID", "Page Title", "Summary"])

    # Step 1: List all databases the integration has access to
    search_results = notion.search(filter={"property": "object", "value": "database"}).get("results", [])

    for db in search_results:
        db_id = db["id"]
        db_title = db["title"][0]["plain_text"] if db.get("title") else "Untitled"

        print(f"Fetching pages for database: {db_title} ({db_id})")

        # Step 2: Query all pages from database
        has_more = True
        next_cursor = None

        while has_more:
            response = notion.databases.query(database_id=db_id, start_cursor=next_cursor)
            pages = response.get("results", [])
            has_more = response.get("has_more", False)
            next_cursor = response.get("next_cursor")

            for page in pages:
                page_id = page["id"]

                # Extract page title
                title_prop = page["properties"].get("Name", {}).get("title", [])
                page_title = title_prop[0]["plain_text"] if title_prop else "Untitled"

                # For summary, pull text from a text property (adjust as needed)
                summary = ""
                for prop_name, prop_value in page["properties"].items():
                    if prop_value["type"] == "rich_text" and prop_value["rich_text"]:
                        summary = prop_value["rich_text"][0]["plain_text"]
                        break

                writer.writerow([db_id, db_title, page_id, page_title, summary])
