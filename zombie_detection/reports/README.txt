Experiment report (LaTeX)
=========================

1) Generate figures and summary_tables.tex from a completed evaluate_models run:

   python zombie_detection/reports/build_report_assets.py \
       --results-root /path/to/results

   Default --results-root is /media/tristan-toye/ESD-USB/results (override if needed).

   Outputs under zombie_detection/reports/figures/:
   - Learning curve PNGs (copied or plotted)
   - Summary / ablation matplotlib figures
   - summary_tables.tex (LaTeX fragment for numeric tables)

2) Compile the PDF (requires a LaTeX install, e.g. TeX Live):

   cd zombie_detection/reports
   pdflatex experiment_report.tex
   pdflatex experiment_report.tex

   Second run clears cross-reference warnings if any.

Packages used by experiment_report.tex: geometry, graphicx, booktabs,
hyperref, microtype, enumitem, amsmath, xcolor.

If pdflatex is not installed, install texlive-latex-base (or full texlive)
on your system; the report source is still valid for Overleaf by uploading
this directory including figures/.
