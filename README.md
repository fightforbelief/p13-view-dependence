# Sharp position bounds under unknown Black–Litterman view correlations

[Paper](paper/manuscript.pdf) · [Three-view sign results](results.json) · [Historical position bounds](historical_results.json) · [Four-view rank certificates](four_view_results.json)

Given the separate error variances of a set of forecasts, which positions can a portfolio manager justify? The paper gives a joint feasibility test and sharp limits on linear portfolio exposures. In any view dimension the extremum either attains the convex outer bound or can be realized with a rank-two dependent covariance component. With three views, searching planar configurations covers every position endpoint. With four views a finite check determines exactly when a third factor is essential.

Across 46 January US industry configurations, unrestricted correlation structures and all signed common-correlation paths agree on which manufacturing trades can reverse. The median of the within-year width differences lies between 2.06 and 2.10 percentage points. A manufacturing weight limit of 0–40% reduces that median difference to zero, with an increase above one point in eight years. Adding a fourth spread reveals eight endpoints requiring a three-factor dependent component. The inputs are historical reconstructions and the positions are hypothetical. The paper also audits the eight-asset Idzorek example.

## Reproduce the study

Run from the project root. The reference run used Windows, Python 3.12.10, NumPy 2.2.6, SymPy 1.14.0 and mpmath 1.3.0.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe study.py
.\.venv\Scripts\python.exe signed_stress.py
.\.venv\Scripts\python.exe historical.py
.\.venv\Scripts\python.exe mandate.py
.\.venv\Scripts\python.exe four_views.py
~~~

The original sign study (study.py) rewrites results.json, recording the environment and SHA-256 of the input and scripts. Threshold brackets are integer multiples of 0.0001. The lower endpoint excludes cancellation and the upper endpoint admits it. A null radius certifies that zero is unattainable even at unrestricted dependence. Reported correlations and gain vectors for strict reversals are decimal renderings. The polynomial decisions use exact arithmetic. The test suite includes the closed-form 16/17 threshold and a positive-definite witness that both restricted stress families miss.

The input file transcribes rounded tables from [Idzorek's July 2004 guide](https://people.duke.edu/~charvey/Teaching/BA453_2005/Idzorek_onBL.pdf). The main reconstruction uses market weights, risk aversion and the covariance table. The sensitivity calculation uses the implied returns printed separately in Table 1. The manuscript compares all three view trades under four variance specifications. Its objective is identification of allocations under incomplete covariance information. It contains no realized-return backtest.

The historical study writes historical_results.json, including every origin, both endpoint enclosures and input/code hashes. four_views.py writes four_view_results.json with a rank classification for each endpoint and a value enclosure for rank-three cases. Other endpoints in that file are classified by rank only. mandate.py computes the single-asset mandate results from the saved three-view intervals. The historical data vintage and the pre-computation design are in [the design record](EXTENSION_DESIGN.md). The PDF figure is included. To regenerate it from the results use Python with Matplotlib and run `python paper/width_figure.py`.

Build the PDF with `tectonic paper/manuscript.tex`.
