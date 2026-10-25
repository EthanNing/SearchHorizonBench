# SearchHorizonBench project website

A buildless GitHub Pages site for **SearchHorizonBench: A Benchmark for Search Value in AI Agents**.

Serve the folder with a local HTTP server so the recorded decision data loads:

```bash
python3 -m http.server 8080 --directory docs
```

The website uses local Helvetica fonts, vector illustrations and recorded measurements. It makes no external requests. `assets/results.json` is a reduced export of `runs/decision_information_factorial_v2/analysis.json`; all five token prices and their component-bootstrap intervals are preserved.

The release preparation step supplies:

- `assets/SearchHorizonBench.pdf`, the authored arXiv manuscript.
- `downloads/SearchHorizonBench.zip`, the standalone benchmark release and its README.

The page displays a real, explicitly selected paired-execution example. The gain waveform is labeled schematic. The price explorer shows empirical measurements, including zero token price, and states that selection observations are free. It separates that information-value diagnostic from acquisition-costed deployment.

Publish the contents of this directory as the GitHub Pages source. No framework build or package installation is required.
