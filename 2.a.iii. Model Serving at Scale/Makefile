.PHONY: install test run loadtest loadtest-compare report docker-up docker-down

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	python -m tests.test_serving

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

loadtest:
	python -m loadtest.run_loadtest

# Runs the same workload against three server configs (naive / batching-only /
# batching+cache) and writes the before/after comparison to loadtest/report.md.
# Ports 8011-8013 must be free; the app on port 8000 (if any) is untouched.
loadtest-compare:
	python -m loadtest.compare

report:
	python -m loadtest.render_report

docker-up:
	docker compose up --build

docker-down:
	docker compose down
