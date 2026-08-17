"""Fast, no-LLM sanity checks for the tool + data layer.

Run:  python test_tools.py
This does NOT need an API key or any model — it just verifies the mock data
loads and each tool returns sensible output.
"""
from app import tools


def main() -> None:
    print(tools.list_pipelines.invoke({}))
    print("---")
    print(tools.list_pipelines.invoke({"status": "failed"}))
    print("---")
    print(tools.get_pipeline_status.invoke({"pipeline_id": "pipe-1001"}))
    print("---")
    print(tools.get_build_logs.invoke({"pipeline_id": "pipe-1004"}))
    print("---")
    print(tools.get_resource_usage.invoke({"pipeline_id": "pipe-1005"}))
    print("---")
    print(tools.list_tickets.invoke({}))
    print("---")
    # create_ticket with no confirmation channel must NOT write:
    print(tools.create_ticket.invoke({
        "title": "Fix build", "description": "llm-router 2.4.1 missing", "pipeline_id": "pipe-1004",
    }))
    print("\nAll tool/data checks ran.")


if __name__ == "__main__":
    main()
