# Which Black–Litterman trades survive unknown dependence between views?

[Read the manuscript](paper/manuscript.pdf) · [LaTeX source](paper/manuscript.tex) · [Certified results](results.json)

A portfolio manager may know the error variance of each view without knowing which errors move together. The paper tests whether a proposed allocation can arise under any joint covariance with those variances. A polygon inequality gives an exact test in any view dimension, and three factors suffice to realize every attainable allocation. For three views, a univariate quartic determines whether a single trade can disappear.

In the published eight-asset example all three directions survive arbitrary dependence at the benchmark variances. With twice the marginal error variances, developed equity and the growth spread can both reverse. A single-pair stress sweep misses the former, while a common-correlation sweep misses the latter. Repeating the common-correlation sweep across every sign convention finds both cancellations at shares 0.97391 and 0.86975, compared with 0.9170–0.9171 and 0.8355–0.8356 for the full set. The bond spread retains its direction. These are conditional allocation results for a fixed book.

## Reproduce the study

Run from the project root. The reference run used Windows, Python 3.12.10, SymPy 1.14.0 and mpmath 1.3.0.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe study.py
.\.venv\Scripts\python.exe signed_stress.py
~~~

The script rewrites results.json, recording the environment and SHA-256 of the input and scripts. Threshold brackets are integer multiples of 0.0001. The lower endpoint excludes cancellation and the upper endpoint admits it. A null radius certifies that zero is unattainable even at unrestricted dependence. Reported correlations and gain vectors for strict reversals are decimal renderings. The polynomial decisions use exact arithmetic. The test suite includes the closed-form 16/17 threshold and a positive-definite witness that both restricted stress families miss.

The input file transcribes rounded tables from [Idzorek's July 2004 guide](https://people.duke.edu/~charvey/Teaching/BA453_2005/Idzorek_onBL.pdf). The main reconstruction uses market weights, risk aversion and the covariance table. The sensitivity calculation uses the implied returns printed separately in Table 1. The manuscript compares all three view trades under four variance specifications. Its objective is identification of allocations under incomplete covariance information. It contains no realized-return backtest.

Build the PDF with `tectonic paper/manuscript.tex`.
