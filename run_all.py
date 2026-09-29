# run_all.py
#
# Runs the analysis pipeline in order: data -> tests of pairs -> hedge ratios ->
# stationarity -> signals -> backtest -> metrics -> robustness -> independent
# verification -> figures -> paper macros/tables -> unit tests.
#   python run_all.py
# It does NOT compile the PDF (see the message printed at the end).
# Note: the unit tests run last, after the outputs they inspect exist.

import subprocess
import sys
import os

STEPS = [
    "src/data_collection.py",
    "src/cointegration.py",
    "src/static_ols.py",
    "src/rolling_ols.py",
    "src/kalman_filter.py",
    "src/stationarity.py",
    "src/signal_generation.py",
    "src/backtesting.py",
    "src/performance_metrics.py",
    "src/robustness.py",
    "src/verify_independent.py",
    "src/visualizations.py",
    "src/make_paper_numbers.py",
]

os.chdir(os.path.dirname(os.path.abspath(__file__)))

for step in STEPS:
    print(f"\n===== {step} =====")
    subprocess.run([sys.executable, step], check=True)

print("\nRunning tests...")
subprocess.run([sys.executable, "tests/test_basic.py"], check=True)

print("\nDone. Analysis outputs: data/*.csv, results/tables/*.tex, results/figures/*.png, paper/numbers.tex.")
print("run_all.py does NOT compile the PDF. To build the paper separately:")
print("  cd paper && latexmk -pdf main.tex")
print("  (or: pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex; pdflatex main.tex)")
