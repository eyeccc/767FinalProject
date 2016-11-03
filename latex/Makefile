all: report

report:
	pdflatex report
	bibtex report
	pdflatex report
	pdflatex report


clean:
	rm *.log *.dvi *.aux *.brf *.out *.bbl $(INCLUDED)
