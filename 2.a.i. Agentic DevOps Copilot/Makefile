.PHONY: install test eval run api ui docker-build docker-run

install:
	pip install -r requirements.txt

# Deterministic checks — no API key needed
test:
	python test_tools.py
	python -m eval.test_eval

# Full agent evaluation — needs OPENAI_API_KEY (or Ollama configured in .env)
eval:
	python -m eval.run_eval --threshold 0.8

run:
	python -m app.cli

api:
	uvicorn app.api:app --reload

ui:
	streamlit run streamlit_app.py

docker-build:
	docker build -t devops-copilot .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env devops-copilot

# Full stack (app + Ollama + auto model pull) in one command
compose-up:
	docker compose up --build

compose-down:
	docker compose down
