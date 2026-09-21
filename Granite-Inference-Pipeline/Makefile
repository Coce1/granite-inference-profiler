# Variables
PYTHON = python
PIP = pip
SCRIPT = Granite_Inference_Pipeline.py

# Déclare les cibles qui ne sont pas des fichiers physiques
.PHONY: all install run clean

# Cible par défaut exécutée si on tape juste "make"
all: install run

# Installation des dépendances depuis le fichier requirements.txt
install:
	$(PIP) install -r requirements.txt

# Exécution du benchmark
run:
	$(PYTHON) $(SCRIPT)

# Nettoyage des dossiers de cache générés par Python
clean:
	$(PYTHON) -c "import os, shutil; [shutil.rmtree(d) for d in os.listdir('.') if d == '__pycache__']"