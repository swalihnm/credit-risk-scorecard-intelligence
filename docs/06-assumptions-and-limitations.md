# Assumptions and limitations

- **Two different portfolios, not merged.** Home Credit (unsecured consumer lending) and
  Fannie Mae (US agency mortgages) are different products, markets, and regulatory regimes.
  They are used side by side to demonstrate complementary techniques — application
  scorecarding vs. portfolio lifecycle/vintage analytics — never combined into one book.
  See `docs/01-data-sources.md`.
- **Home Credit is now real data** (downloaded from Kaggle, 307,511 applications, verified
  against the known ~8.07% base default rate); **Fannie Mae and FRED remain synthetic
  stand-ins** — disclosed explicitly, not hidden — because real Fannie Mae data requires
  manual portal registration and FRED's swap-in reader (`ingest/fred_ingest.py`) just hasn't
  been run yet. Every generator produces the exact real-world schema, and every
  corresponding real-data reader (`ingest/kaggle_download.py`, `ingest/fannie_mae_ingest.py`,
  `ingest/fred_ingest.py`) is written — swapping the remaining two in means running those and
  rebuilding the warehouse, with nothing downstream changing. The Fannie Mae generator
  (`ingest/synthetic_fannie_mae.py`) is a macro-modulated Markov delinquency model tuned to be
  directionally realistic (crisis vintages default more than benign vintages).
- **The Home Credit reject pool is still synthetic** even though the accepted population is
  now real — see `docs/01-data-sources.md` and `docs/03-reject-inference.md`. This isn't a
  regression from using real data; no version of this dataset (real or synthetic) includes a
  declined-applicant population, so a synthetic augmentation is the only way to demonstrate
  reject inference at all. Its specific numbers (reject-pool feature distributions) are
  generated independently of the real accepted population's actual distributions, so treat
  the reject-inference results as illustrating the *mechanism*, not a production-calibrated
  correction.
- Default definition, performance window, and SICR trigger are choices made for this project;
  in a real bank they are set by policy and regulation.
- LGD is estimated from a (synthetic) US mortgage-shaped portfolio and would not transfer to
  unsecured lending — it is used only against the Fannie-Mae-side ECL engine, never against
  Home Credit exposures.
- Reject inference is an assumption-driven correction and cannot be fully validated without a
  randomized approval experiment.
- Stress scenarios (baseline/adverse/severely adverse) are illustrative, not
  regulator-prescribed CCAR/DFAST scenarios.
- **No fair lending or disparate impact testing is performed.** In a real deployment this
  would be mandatory. This omission is named explicitly rather than left implicit.
- The synthetic Fannie Mae panel samples ~22,000 loans across 2005-2010 and 2015-2019
  vintages — a deliberately small, illustrative panel, not a claim about real portfolio scale.
