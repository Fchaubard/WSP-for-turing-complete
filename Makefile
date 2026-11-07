PYTHON ?= python3

.PHONY: paper verify figures
paper:
	$(PYTHON) paper/build.py
	cp output/pdf/weight-space-programming.pdf output/pdf/weight-space-programming-public.pdf
	cp output/pdf/weight-space-programming.pdf paper.pdf

verify:
	$(PYTHON) demo.py verify
	$(PYTHON) verify.py
	$(PYTHON) focus/learning_review/audit.py
	$(PYTHON) focus/learning_review/check_plot_data.py
	$(PYTHON) focus/learning_review/check_curves.py

figures:
	$(PYTHON) focus/holistic_plots/build.py
	$(PYTHON) focus/learning_plots/build.py
	$(PYTHON) focus/main_visuals/build.py
	$(PYTHON) revision/architecture_release.py
	$(PYTHON) revision/learning_release.py
