# Simple helpers for running the demo and connectivity checks

# DEMO_MODEL lets you switch between 20b/120b in one place.
# Usage: make demo DEMO_MODEL=120b
DEMO_MODEL ?=
MODEL ?= gpt-oss:20b
BASE_URL ?= http://localhost:11434
FORMAT ?= text
TOKENS ?= 128
REPORT ?=
PROMPT ?= Say hi in five words
FILE ?= example_pipeline.yaml
OUT ?= dag.dot
CONFIG ?=
NO_CACHE ?=

# If DEMO_MODEL is set, override MODEL accordingly (e.g., 20b -> gpt-oss:20b)
ifneq ($(strip $(DEMO_MODEL)),)
  MODEL := gpt-oss:$(DEMO_MODEL)
endif

REPORT_ARG := $(if $(REPORT),-o $(REPORT),)

.PHONY: help demo demo-120b check check-model check-model-120b quick quick-120b validate visualize run

help:
	@echo "Targets:"
	@echo "  make demo            # Run demo (use DEMO_MODEL=20b|120b or explicit MODEL)"
	@echo "  make check           # Check Ollama server (BASE_URL)"
	@echo "  make check-model     # Check server + model (use DEMO_MODEL or MODEL)"
	@echo "  make quick           # Quick prompt (use DEMO_MODEL or MODEL; PROMPT text)"
	@echo "  make validate        # Validate a pipeline spec (FILE)"
	@echo "  make visualize       # Output Graphviz DOT (FILE -> OUT)"
	@echo "  make run             # Run a pipeline spec (FILE, CONFIG, NO_CACHE)"
	@echo "  make judge           # LLM-as-Judge demo (JTEXT, JTEMPLATE, EXPLAIN, STREAM)"
	@echo "  make synthetic       # Synthetic security evaluation (SYNTHETIC_CASES, SYNTHETIC_ROUNDS, SYNTHETIC_CRITICS)"
	@echo "  make synthetic-pipeline # Run complete synthetic evaluation pipeline"
	@echo "Aliases (prefer DEMO_MODEL form above):"
	@echo "  make demo-120b | check-model-120b | quick-120b | judge-120b | synthetic-120b"
	@echo "Variables: DEMO_MODEL=$(DEMO_MODEL) MODEL=$(MODEL) BASE_URL=$(BASE_URL) FORMAT=$(FORMAT) TOKENS=$(TOKENS) REPORT=$(REPORT) PROMPT='$(PROMPT)'"
	@echo "           FILE=$(FILE) OUT=$(OUT) CONFIG=$(CONFIG) NO_CACHE=$(NO_CACHE)"
	@echo "Judge demo variables: JTEXT='$(JTEXT)' JTEMPLATE='$(JTEMPLATE)' EXPLAIN=$(EXPLAIN) STREAM=$(STREAM) STREAM_PRINT=$(STREAM_PRINT)"
	@echo "Synthetic variables: SYNTHETIC_CASES=$(SYNTHETIC_CASES) SYNTHETIC_ROUNDS=$(SYNTHETIC_ROUNDS) SYNTHETIC_CRITICS=$(SYNTHETIC_CRITICS) SYNTHETIC_TYPE=$(SYNTHETIC_TYPE) SYNTHETIC_OUTPUT=$(SYNTHETIC_OUTPUT)"

demo:
	source venv/bin/activate && python -m cli demo-ollama \
	  --model $(MODEL) \
	  --base-url $(BASE_URL) \
	  --format $(FORMAT) \
	  --tokens-per-prediction $(TOKENS) \
	  $(REPORT_ARG)

demo-120b:
	$(MAKE) demo MODEL=gpt-oss:120b

check:
	source venv/bin/activate && python -m cli check-ollama --base-url $(BASE_URL)

check-model:
	source venv/bin/activate && python -m cli check-ollama --base-url $(BASE_URL) --model $(MODEL)

check-model-120b:
	$(MAKE) check-model MODEL=gpt-oss:120b

quick:
	source venv/bin/activate && python -m cli check-ollama --base-url $(BASE_URL) --model $(MODEL) --quick-prompt "$(PROMPT)"

quick-120b:
	$(MAKE) quick MODEL=gpt-oss:120b

validate:
	source venv/bin/activate && python -m cli validate -f $(FILE)

visualize:
	source venv/bin/activate && python -m cli visualize -f $(FILE) -o $(OUT)

# Run any pipeline spec via CLI
CONFIG_ARG := $(if $(CONFIG),--config $(CONFIG),)
NO_CACHE_ARG := $(if $(filter 1 true yes,$(NO_CACHE)),--no-cache,)

run:
	source venv/bin/activate && python -m cli run -f $(FILE) $(CONFIG_ARG) $(NO_CACHE_ARG)

# Judge demo (LLM-as-Judge via EvaluatorNode)
JTEXT ?= Please say hello politely.
JTEMPLATE ?= Rate the politeness of this text (1-5): {text}
EXPLAIN ?=
STREAM ?=
STREAM_PRINT ?=

EXPLAIN_ARG := $(if $(filter 1 true yes,$(EXPLAIN)),--explanation,)
STREAM_ARG := $(if $(filter 1 true yes,$(STREAM)),--stream,)
STREAM_PRINT_ARG := $(if $(filter 1 true yes,$(STREAM_PRINT)),--stream-print,)

.PHONY: judge judge-120b synthetic synthetic-120b synthetic-pipeline
judge:
	source venv/bin/activate && python -m cli demo-judge \
	  --text "$(JTEXT)" \
	  --template "$(JTEMPLATE)" \
	  --model $(MODEL) \
	  --base-url $(BASE_URL) \
	  --temperature 0.0 \
	  $(EXPLAIN_ARG) $(STREAM_ARG) $(STREAM_PRINT_ARG)

judge-120b:
	$(MAKE) judge DEMO_MODEL=120b

# Synthetic security evaluation
SYNTHETIC_CASES ?= 5
SYNTHETIC_ROUNDS ?= 2
SYNTHETIC_CRITICS ?= 2
SYNTHETIC_TYPE ?= security_research
SYNTHETIC_OUTPUT ?=

SYNTHETIC_OUTPUT_ARG := $(if $(SYNTHETIC_OUTPUT),--output $(SYNTHETIC_OUTPUT),)

synthetic:
	source venv/bin/activate && python -m cli synthetic-demo \
	  --model $(MODEL) \
	  --base-url $(BASE_URL) \
	  --num-cases $(SYNTHETIC_CASES) \
	  --content-type $(SYNTHETIC_TYPE) \
	  --rounds $(SYNTHETIC_ROUNDS) \
	  --critics $(SYNTHETIC_CRITICS) \
	  $(SYNTHETIC_OUTPUT_ARG)

synthetic-120b:
	$(MAKE) synthetic DEMO_MODEL=120b

synthetic-pipeline:
	source venv/bin/activate && python -m cli run -f synthetic_evaluation_pipeline.yaml
