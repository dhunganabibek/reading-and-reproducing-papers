all: report.pdf
report.pdf: report.tex preamble.tex
	latexmk -pdf -interaction=nonstopmode -quiet report.tex
clean:
	latexmk -C
