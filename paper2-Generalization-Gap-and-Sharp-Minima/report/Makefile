.PHONY: all clean

all: report.pdf

report.pdf: report.tex preamble.tex $(wildcard figures/*.png)
	latexmk -pdf -interaction=nonstopmode -halt-on-error -quiet report.tex

clean:
	latexmk -c
