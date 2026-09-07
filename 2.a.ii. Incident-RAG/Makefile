.PHONY: install gen-data ingest test eval eval-retrieval chat api dashboard compose-up compose-down

install:
	pip install -r requirements.txt

# (Re)generate the synthetic corpus + golden Q&A
gen-data:
	python scripts/generate_data.py --incidents 220

# Build the Chroma index (needs Ollama + nomic-embed-text)
ingest:
	python -m rag.ingest

# Deterministic metric tests — no model/key needed
test:
	python -m eval.test_eval

# Full evaluation (retrieval + generation) -> eval/report.json
eval:
	python -m eval.run_eval --threshold 0.8

# Faster: retrieval-only (embeddings, no LLM)
eval-retrieval:
	python -m eval.run_eval --retrieval-only --threshold 0.8

chat:
	streamlit run chat_ui.py

api:
	uvicorn rag.api:app --reload

dashboard:
	streamlit run dashboard.py

compose-up:
	docker compose up --build

compose-down:
	docker compose down
