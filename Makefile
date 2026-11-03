# Variables
PYTHON ?= python3
VENV   = .venv
VPY    = $(VENV)/bin/python
SCRIPT = src/Granite_Inference_Pipeline.py
# Arguments supplémentaires : make run ARGS="--dtype bfloat16 --repeats 10"
ARGS  ?=

.PHONY: all venv install run profile clean distclean

all: install run

# Environnement virtuel (évite de polluer le Python système)
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

venv: $(VENV)/bin/activate

install: venv
	$(VPY) -m pip install --upgrade pip
	$(VPY) -m pip install -r requirements.txt

run:
	$(VPY) $(SCRIPT) $(ARGS)

profile:
	$(VPY) $(SCRIPT) --profile $(ARGS)

# Supprime tous les __pycache__ (récursif)
clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

# clean + environnement virtuel + résultats
distclean: clean
	rm -rf $(VENV) results
