# System-Wide Stress Test Model

This is the reference implementation of the system-wide stress test model, as
described in [this
paper](https://www.bankofengland.co.uk/working-paper/2020/foundations-of-system-wide-financial-stress-testing-with-heterogeneous-institutions).
This repo uses public data from EBA, while the result in the paper is generated
using confidential data. And so this repo is not sufficient to exactly
reproduce the paper result. However, the result is qualitatively the same.

This repo uses the [Resilience
library](https://github.com/ox-inet-resilience/resilience).

For a self-contained introduction to the model in fewer than 1k lines of code,
see https://github.com/ox-inet-resilience/firesale_stresstest.  The
`firesale_stresstest` repository implements all of its building block from
scratch instead of using the Resilience library.  It reproduces the result of
Cont-Schaanning 2017.

## Usage

Requires Python 3. It is recommended to run everything within a virtualenv.
- `git clone` the repo and `cd` to it.
- Install the dependencies with `pip install .`
- Run the simulation with `python sw_stresstest/simulation.py`

You can tweak the number of repetitions of the simulations by changing `NSIM`
in the simulation.py file. To speed up the simulation, you may decrease the
value of `NPOINTS`. This represents the number of points in the x-axis in the
simulation plots. Note: for now, changing `NPOINTS` will cause problem with
running simulation FF11 because it is hardcoded to assume that the length in
11. This will be fixed in the future version.
