# uniform-competition-continuum

Code for the figures of

> A. C. Fassoni. *Fitness is asymptotically irrelevant under uniform competition in phenotype-structured populations.* Preprint, 2026. arXiv: *to be added*.

This paper proves the continuum results stated in the review *Stochastic gradient descent on the epigenetic landscape* (code in [sgd-epigenetic-landscape-review](https://github.com/arturfassoni/sgd-epigenetic-landscape-review)).

## Figures

A single script, `make_figures.py`, produces the three figures:

| Output | Figure | What it shows |
|---|---|---|
| `fig1_mechanism.pdf` | 1 | Mechanism of the proof: monotone total population, one-signed reaction term, and the switching dynamics erasing zero-mass perturbations |
| `fig2_numerics.pdf` | 2 | Tilted double well: uniform competition erases the imprint of fitness, additive competition does not; memory time in the Kramers regime |
| `fig3_examples.pdf` | 3 | Whole real line (Ornstein–Uhlenbeck, heavy tails), nonlocal switching by jumps, and the absence of a uniform rate for the OU dynamics |

All PDEs are discretized with a finite-volume scheme that conserves mass, preserves positivity and has the stationary density as exact steady state (Section 6.2 of the paper).

## Running

```bash
pip install -r requirements.txt
python3 make_figures.py
```

The script writes the PDFs to the current folder and prints the numbers quoted in the text (spectral gaps, distances, memory times). It takes about 20 seconds. Tested with Python 3.11, NumPy 2.4, SciPy 1.17 and Matplotlib 3.11.

## License

MIT (see `LICENSE`).
