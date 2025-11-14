.PHONY: build up shell run-example1 run-example2 down clean

build:
	docker compose build

up:
	docker compose up -d

shell:
	docker compose exec checkpoint-demo bash

run-example1:
	docker compose exec checkpoint-demo python examples/example1.py

run-example2:
	docker compose exec checkpoint-demo python examples/example2.py

down:
	docker compose down

clean:
	docker compose down -v
	rm -rf checkpoints data

logs:
	docker compose logs -f