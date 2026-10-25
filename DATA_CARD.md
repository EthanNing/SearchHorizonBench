# Data and evaluation card

The benchmark joins public agent histories to repeated outcomes from alternative
continuations. A forecaster estimates action success and expected reward gains;
the offline scorers measure those predictions and a cost-threshold policy.

## Populations

- `cost_confirmation` supplies the Test-1632 main cohort, the saved calibration
  baseline, and binary forecast families. Five HotpotQA/MuSiQue strata receive
  equal evaluation mass; the four-hop stratum contains 32 questions.
- `strengthening` includes Test-400, multiple checkpoints, budget conditions,
  and cross-model forecast records. Dataset metadata identifies each frame.
- `component_maintenance` and `component_static_sources` contain pending-query
  action comparisons across actor/retriever or indexed-source settings.
- `natural_queries` contains the separate Natural Questions transfer study.
- `collection_value` contains 32 development and 96 evaluation collection tasks,
  with STOP, same-source search and source-switch alternatives.
- Other exported directories preserve calibration and repeat-control records.

Metadata and split records retain the original experimental provenance.
Repeated endpoint rows, shared STOP outcomes, and revisited questions create
overlap across files. File-row totals are not counts of independent experiments.

## Metrics

Action-success Brier averages the two binary squared-probability errors. Gain
MSE uses independent repeated draws to subtract within-state generation noise.
Its finite-sample estimate can be negative and stays unmodified. Support-page
components define bootstrap resampling units. Policy utility subtracts charged
API tokens from the selected reward. Collection outcomes use bounded set F1.

The main answer target is reference-equivalent correctness under the saved
Flash semantic scoring rule. Exact match, alternate semantic judges, and
shared-uncertainty projections retain their own endpoint definitions. Human
annotation is not supplied in this artifact; recorded review labels are AI
assessments. Original labels, references and shared case identities are retained.

## Reuse

Saved public inputs support offline prediction. Training uses designated
development records. Report the chosen endpoint, actor, weighting, predictor
training population, prediction cost and evaluation frame with each result.
Compare new methods against matched historical means as well as raw forecasts.
For questions and corrections contact jening@cs.cmu.edu.
