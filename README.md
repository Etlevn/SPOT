# SPOT — Sparse Portfolio Optimization for ETF Tracking

SPOT tracks an artificial-intelligence ETF with a small subset of its disclosed constituents. The decision, the data, the mixed-integer quadratic program, and the quarterly out-of-sample backtest have to line up, and the whole pipeline has to be reproducible.

A first master table is built from the raw Excel extracts by `src/data` and inspected with `test_data.py`. The optimizer, the backtest, and the experiment results do not exist yet. Decisions marked TBD have to be frozen before those stages. Quantitative claims require results from the actual pipeline.

## Question

The decision-maker is an investment manager. The candidate stocks are the holdings of the **Global X Artificial Intelligence & Technology ETF (AIQ)** that were already public at the decision time, and AIQ's own returns are the benchmark. AIQ represents that fund's portfolio, not the full set of AI companies.

At the start of each quarter the model chooses which stocks to hold and what weight each receives, minimizing the mean squared difference between the portfolio's daily returns and AIQ's daily returns over a historical estimation window. The baseline rule allows at most 10% of the eligible constituents that quarter, rounded down. The question is whether a basket that sparse can match AIQ's out-of-sample daily returns, and how tracking quality, concentration, and holding stability change when the cap is relaxed to 20%, 30%, 40%, and 50%.

This is index replication. The objective is to shrink the gap between portfolio returns and ETF returns. It is not a search for the stocks with the highest expected returns.

## Scope

- **Benchmark:** AIQ, under one USD return definition for the whole study.
- **Universe:** the latest holdings publicly available before each decision cutoff, after eligibility checks.
- **Portfolio:** long only; nonnegative weights that sum to one at formation; rebalanced quarterly; buy-and-hold between rebalances, with no daily reset to target weights.
- **Solver:** Gurobi, as a mixed-integer quadratic program (MIQP).
- **Left out of the base model:** shorting, leverage, derivatives, taxes, explicit transaction costs, and turnover penalties.
- **Tentative window:** prices from about October 2022 through September 2026; twelve out-of-sample quarters from October 2023 through September 2026. Exact bounds depend on verified historical holdings and data availability.

## Model

For quarter \(q\), an estimation window, and a selection ratio, solve over the eligible set \(\mathcal I_q\):

\[
\begin{aligned}
\min_{w,z,e}\quad & \frac{1}{|\mathcal T|}\sum_{t} e_{t}^{2}\\
\text{s.t.}\quad
& e_{t}=\sum_{i} r_{it} w_{i}-b_{t},\\
& \sum_{i} w_{i}=1,\\
& \sum_{i} z_{i}\le K,\quad K=\lfloor\alpha N\rfloor,\\
& \ell z_{i}\le w_{i}\le u z_{i},\\
& w_{i}\ge 0,\quad z_{i}\in\{0,1\}.
\end{aligned}
\]

| Symbol | Meaning | Baseline |
|---|---|---|
| \(z_i\in\{0,1\}\) | Whether stock \(i\) is activated and may receive a positive weight | Decision variable |
| \(w_i\ge 0\) | Formation weight | Decision variable |
| \(e_t\) | Portfolio return minus AIQ return on a training day | Decision variable |
| \(N=\|\mathcal I_q\|\) | Number of eligible stocks that quarter | From the data |
| \(\alpha\) | Maximum selection ratio | 0.10 |
| \(K=\lfloor\alpha N\rfloor\) | Maximum number of activated stocks | From that quarter's \(N\) |
| \(\ell,u\) | Lower and upper weight on an activated stock | 0 and 1 |

\(K\) is a maximum, so the portfolio may hold fewer names. The 10% cap counts eligible stocks, not 10% of ETF market value. Because \(\ell=0\), activation does not force a strictly positive weight; actual holdings are counted from positive weights with a documented numerical tolerance. Selection and weights are chosen in one solve: the objective is the tracking error of the portfolio as a whole.

For a fixed eligible set, estimation window, and position bounds, a larger \(\alpha\) only expands the feasible set, so the globally optimal in-sample mean squared error cannot get worse. Whether out-of-sample tracking improves is an empirical question.

If \(K=0\), full investment is infeasible. That run is flagged, excluded, and explained. Position-cap experiments also require \(Ku\ge 1\).

## Timing and the backtest

A holdings as-of date is not the date the snapshot became public. A quarter-end file published after quarter-end cannot be used for a rebalance at the start of that quarter. Six timestamps stay distinct: holdings date, publication time, information cutoff, decision time, assumed execution time, and the out-of-sample holding interval. A quarter with no verifiable pre-cutoff snapshot is flagged. A later snapshot is not substituted.

At each decision date the 3-, 6-, and 12-month estimation windows share one eligible universe. Eligibility requires at least twelve months of acceptable prior returns, so differences across windows come from sample length. The optimizer approximates training-period returns with constant weights. The backtest buys and holds from formation, and weights drift with prices. Position bounds apply at formation only. If a holding is delisted or acquired during the quarter, its final realizable return is included and the proceeds stay in cash until the next rebalance.

Validate one quarter end to end before scaling to all twelve. That pass tests the pipeline. If earlier out-of-sample quarters are used to choose parameters, later quarters are reserved as the final holdout.

## Evaluation and comparisons

Out-of-sample active return \(a_t\) is the portfolio return minus the AIQ return. Report both of the following, because they answer different questions:

- **RMSE:** the square root of mean squared active return, which includes the average gap.
- **Tracking error:** the standard deviation of active return, which measures dispersion only.

Also report cumulative-return differences, the number of positive-weight holdings, the largest weight, HHI \(\sum_i w_i^2\), effective holdings \(1/\mathrm{HHI}\), quarterly turnover, overlap across quarters, and the disclosed AIQ weight of excluded securities. Turnover compares pre-rebalance drifted weights with the new target weights.

Three portfolios share the point-in-time universe, the execution rule, and the corporate-event treatment:

1. **MIQP:** joint selection and weights.
2. **Top-\(K\) equal weight:** the \(K\) eligible names with the largest weights in the latest public AIQ holdings, held equally.
3. **Top-\(K\) disclosure weight** (proposed extension): the same names, with weights proportional to disclosed AIQ weights and scaled to sum to one.

The optimized portfolio may hold fewer than \(K\) names. The two baselines hold exactly \(K\) when that is feasible.

Experiment grid: baseline \(\alpha=0.10\); selection ratios \(\{0.10,0.20,0.30,0.40,0.50\}\); estimation windows of 3, 6, and 12 months; optional position caps \(u\in\{1.00,0.50,0.25\}\). Whether the headline result uses the 12-month window is still open.

## Data

AIQ, THNQ, and ARTY were compared, with ROBT and BOTZ as backups. AIQ has the most complete holdings history and price coverage from October 2022 through September 2026, so it is the first case. The points below describe material downloaded in October 2026. They are not a quarter-by-quarter verification.

- **Holdings:** Global X issuer records cover calendar quarter-ends from about September 2024 through September 2026. SEC-derived records cover about May 2022 through May 2026 and include filing dates. Earlier reporting dates often miss calendar quarter-ends. Each decision uses the latest holdings verifiably public before the cutoff, reconciled to original disclosures and matched to prices by security identifier.
- **Constituent prices:** 114 price series and 108,860 daily observations are in hand. Fourteen historical ticker symbols are still missing, and nine adjusted closes are missing in the retrieved series. U.S. listings are planned from WRDS/CRSP; overseas listings from Yahoo Finance with the exchange suffix. Securities that stay unavailable are dropped from the eligible set, and their share of disclosed ETF weight is reported.
- **AIQ prices:** Yahoo Finance, 2022-09-30 through 2026-09-30, with no missing adjusted prices.

Returns are USD total returns. For a foreign-currency asset, \((1+r^{\mathrm{USD}})=(1+r^{\mathrm{local}})(1+r^{\mathrm{FX}})\), with the FX rate as USD per local currency unit. Tickers change, so the permanent key is a stable security identifier.

The master table does not forward-fill prices. Exchange holidays, halts, and vendor gaps stay distinct, and failures stay explicit.

## How the raw extracts become the master table

The raw files are not in the shape the study needs. Holdings are periodic snapshots, prices are daily, and the same security is identified differently in each workbook. `preprocess_master_frame` applies the following rules and returns one table: one row is one USD security on one price date. The current build has 87,081 rows, 26 columns, and 95 securities, from 2022-09-30 through 2026-09-30.

**Holdings use the SEC filing date.** Global X quarter-end files are not read. Those files say which stocks the fund held on 31 March, 30 June, 30 September, or 31 December, but they do not say when the public could see them. `Last-Modified` only shows that the server later changed the file, so it is not a publication date. The SEC extract has both dates: `asOf` becomes `holdings_as_of`, and `filedDate` becomes `publication_at`. Each price date carries the latest snapshot whose filing date is on or before that price date. If the security is absent from that snapshot, weight, CUSIP, and holding name stay empty and `in_published_holdings` is false. Reported weights stay in percent of net assets. They are not rescaled to sum to one.

The two holdings sources are different calendars, not two copies of the same file. SEC dates fall on February, May, August, and November month-ends. Global X dates fall on March, June, September, and December month-ends. From September 2024 both sources exist and the lists can look similar, but the fund may have traded during the month between them.

All twelve quarter-start rebalances from 2023-10-02 through 2026-07-01 have a filing that was already public. None of those filings is the calendar quarter-end portfolio. Each as-of date is about four months before the rebalance. Rebalancing on the quarter-end day itself uses the same filing, because the next filing is published after the quarter-end. October 2023 through September 2024 has no Global X quarter-end file, so skipping those files does not remove that part of the sample.

Adding Global X later requires one new fact per file: `publication_at`, the first time that file was publicly visible. On the holdings sheet every row from the same quarter-end file carries the same value. A file may be used only after that timestamp. The download date and `Last-Modified` do not fill the field. An empty value means that quarter-end file stays unused.

**One security id comes from the Yahoo coverage sheet.** SEC rows have an ISIN and no trading symbol. Price rows use a Yahoo or WRDS symbol. The coverage sheet is the crosswalk, and `security_id` is that symbol. Several ISINs separated by semicolons belong to one security. If one snapshot lists more than one of those ISINs, their disclosed weights are summed. Values of `N/A` become missing. AIQ itself stays in the benchmark columns and is not a constituent. A missing issuer ticker or exchange is left missing.

**Local-currency listings stay out.** No FX table is used. A security quoted in a currency other than USD is dropped. A name Yahoo no longer prices is kept only when the Compustat North America extract has a USD series, which is how delisted names such as TWTR and SPLK enter. Compustat Global is not read. Toshiba, Aveva, and Mandiant have no USD price here, so they are absent.

**The two price extracts share one set of columns.** Yahoo rows keep close, adjusted close, and daily return, with `return_basis` equal to `adjusted_close`. WRDS rows use `prccd` as close and `prccd / ajexdi` as adjusted close. When the total-return factor `trfd` is present, the daily return uses it. Abiomed and CyberArk have no factor, so their return is the price change and `quality_flag` says so. The first day of each series has no previous price, so its return stays missing. A series stops on its last available price. No forward fill is applied, and no cash proceeds are invented.

**The benchmark is attached by date.** AIQ close and return are merged onto each row. A date on which the stock did not trade is not inserted. If the AIQ return is missing after the first benchmark day, that row is flagged.

The build stops on a duplicate security-date row, an ISIN that maps to two security ids, a repeated ISIN inside one snapshot, a non-USD row in the master table, or an infinite numeric value.

## Implementation principles

Raw inputs stay immutable, with provenance, transformations, and quality flags kept alongside them. Acquisition, cleaning, universe construction, optimization, backtesting, evaluation, and experiments stay separate. The optimizer receives validated return matrices and ordered security identifiers. The backtester handles execution, weight drift, and corporate events; it does not rewrite eligibility or realized returns. Every run stores its configuration, input snapshot, software and solver versions, run ID, solver status, MIP gap, and runtime. A feasible solution that is not proven optimal is labeled as such. A methodological change updates the specification and the tests before results are regenerated.

`src/data/` and `test_data.py` are in place. The rest of the target layout is not:

```text
SPOT/
├── configs/          # base, data, experiments
├── data/             # raw, processed, model_inputs
├── src/              # data, universe, optimization, backtest, evaluation, experiments
├── scripts/
├── tests/
├── results/
└── reports/
```

Reusable logic lives in `src/`, orchestration in `scripts/`, and configuration in `configs/`. Notebooks are for exploration and figures.

## Decisions still open

Fixed already: the project name, the AIQ benchmark, long-only fully invested tracking mean squared error, \(\alpha=0.10\), \(\ell=0\), \(u=1\), quarterly rebalancing, the 3/6/12-month windows on a shared eligible universe, and the exclusion of transaction costs from the base model.

Still able to change whether the conclusions are valid:

1. Quarter boundaries, market timezone, information cutoff, execution prices, and non-trading days.
2. A quarter-by-quarter audit of when each holdings file was actually published.
3. Non-U.S. calendars and FX, aligned to one USD return calendar shared with the benchmark.
4. Missing prices, delistings, dividends, and cash.
5. Solver tolerances and time limits, and which quarters are held out if earlier tests are used for tuning.

`DATA_SPEC.md` (schemas, sources, cleaning, return and calendar rules) and `ARCHITECTURE.md` (interfaces, dependencies, commands, and error handling) are still to be written.
