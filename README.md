# SPOT — Sparse Portfolio Optimization for ETF Tracking

SPOT tracks an artificial-intelligence ETF with a small subset of its disclosed constituents. The decision, the data, the mixed-integer quadratic program, and the quarterly out-of-sample backtest have to line up, and the whole pipeline has to be reproducible.

`src` builds the master table from the raw Excel extracts. `check.py` audits that table. `main.py` solves the baseline MIQP for the quarters and windows selected in `config.py`. The out-of-sample backtest and the selection-ratio experiments are not built. A `time_limit` status is a feasible incumbent, not a proven optimum.

## Question

The decision-maker is an investment manager. The candidate stocks are the holdings of the **Global X Artificial Intelligence & Technology ETF (AIQ)** that were already public at the decision time, and AIQ's own returns are the benchmark. AIQ represents that fund's portfolio, not the full set of AI companies.

At the start of each quarter the model chooses which stocks to hold and what weight each receives, minimizing the mean squared difference between the portfolio's daily returns and AIQ's daily returns over a historical estimation window. The baseline rule allows at most 10% of the eligible constituents that quarter, rounded down. The question is whether a basket that sparse can match AIQ's out-of-sample daily returns, and how tracking quality, concentration, and holding stability change when the cap is relaxed to 20%, 30%, 40%, and 50%.

This is index replication. The objective is to shrink the gap between portfolio returns and ETF returns. It is not a search for the stocks with the highest expected returns.

## Scope

- **Benchmark:** AIQ, quoted in USD. Constituent returns stay in the vendor's USD series; local-currency listings are not converted.
- **Universe:** the latest holdings whose SEC filing date is on or before the decision date, after the eligibility checks below.
- **Portfolio:** long only; nonnegative weights that sum to one at formation; rebalanced quarterly; buy-and-hold between rebalances, with no daily reset to target weights.
- **Solver:** Gurobi, as a mixed-integer quadratic program (MIQP).
- **Left out of the base model:** shorting, leverage, derivatives, taxes, explicit transaction costs, and turnover penalties.
- **Window in the current check:** prices from 2022-09-30 through 2026-09-30, and twelve decision dates, one on the first AIQ trading day of each quarter from 2023-10-01 through 2026-07-01.

## Model

For quarter $q$, an estimation window, and a selection ratio, solve over the eligible set $\mathcal{I}_q$:

$$
\begin{aligned}
\min_{w,z,e}\quad & \frac{1}{|\mathcal{T}|}\sum_{t} e_{t}^{2}\\
\text{s.t.}\quad & e_{t}=\sum_{i} r_{it} w_{i}-b_{t},\\
& \sum_{i} w_{i}=1,\\
& \sum_{i} z_{i}\le K,\quad K=\lfloor\alpha N\rfloor,\\
& \ell z_{i}\le w_{i}\le u z_{i},\\
& w_{i}\ge 0,\quad z_{i}\in\{0,1\}.
\end{aligned}
$$

| Symbol | Meaning | Baseline |
|---|---|---|
| $z_i\in\{0,1\}$ | Whether stock $i$ is activated and may receive a positive weight | Decision variable |
| $w_i\ge 0$ | Formation weight | Decision variable |
| $e_t$ | Portfolio return minus AIQ return on a training day | Decision variable |
| $N=\|\mathcal{I}_q\|$ | Number of eligible stocks that quarter | From the data |
| $\alpha$ | Maximum selection ratio | 0.10 |
| $K=\lfloor\alpha N\rfloor$ | Maximum number of activated stocks | From that quarter's $N$ |
| $\ell,u$ | Lower and upper weight on an activated stock | 0 and 1 |

$K$ is a maximum, so the portfolio may hold fewer names. The 10% cap counts eligible stocks, not 10% of ETF market value. Because $\ell=0$, activation does not force a strictly positive weight; actual holdings are counted from positive weights with a documented numerical tolerance. Selection and weights are chosen in one solve: the objective is the tracking error of the portfolio as a whole.

For a fixed eligible set, estimation window, and position bounds, a larger $\alpha$ only expands the feasible set, so the globally optimal in-sample mean squared error cannot get worse. Whether out-of-sample tracking improves is an empirical question.

If $K=0$, full investment is infeasible. That run is flagged, excluded, and explained. Position-cap experiments also require $Ku\ge 1$.

## Timing and the backtest

A holdings as-of date is not the date the snapshot became public. A quarter-end file published after quarter-end cannot be used for a rebalance at the start of that quarter. Six timestamps stay distinct: holdings date, publication time, information cutoff, decision time, assumed execution time, and the out-of-sample holding interval. A quarter with no verifiable pre-cutoff snapshot is flagged. A later snapshot is not substituted.

At each decision date the 3-, 6-, and 12-month estimation windows share one eligible universe. Eligibility requires at least twelve months of acceptable prior returns, so differences across windows come from sample length. The optimizer approximates training-period returns with constant weights. The planned backtest buys and holds from formation, and weights drift with prices. Position bounds apply at formation only. If a holding is delisted or acquired during the quarter, its final realizable return is included and the proceeds stay in cash until the next rebalance. That cash rule is not in the master table or the solver.

`main.py` solves every selected quarter and window in calendar order. The backtest that would hold those weights until the next decision date is not built. If earlier out-of-sample quarters are used to choose parameters, later quarters are reserved as the final holdout.

## Evaluation and comparisons

A finished solve stores the training mean squared error, formation HHI, and effective holdings $1/\mathrm{HHI}$. Out-of-sample active return $a_t$ is the portfolio return minus the AIQ return. That series is not computed yet. Report both of the following, because they answer different questions:

- **RMSE:** the square root of mean squared active return, which includes the average gap.
- **Tracking error:** the standard deviation of active return, which measures dispersion only.

Also report cumulative-return differences, the number of positive-weight holdings, the largest weight, HHI $\sum_i w_i^2$, effective holdings $1/\mathrm{HHI}$, quarterly turnover, overlap across quarters, and the disclosed AIQ weight of excluded securities. Turnover compares pre-rebalance drifted weights with the new target weights.

Three portfolios share the point-in-time universe, the execution rule, and the corporate-event treatment:

1. **MIQP:** joint selection and weights.
2. **Top-$K$ equal weight:** the $K$ eligible names with the largest weights in the latest public AIQ holdings, held equally.
3. **Top-$K$ disclosure weight** (proposed extension): the same names, with weights proportional to disclosed AIQ weights and scaled to sum to one.

The optimized portfolio may hold fewer than $K$ names. The two baselines hold exactly $K$ when that is feasible.

Experiment grid: baseline $\alpha=0.10$; selection ratios $\{0.10,0.20,0.30,0.40,0.50\}$; estimation windows of 3, 6, and 12 months; optional position caps $u\in\{1.00,0.50,0.25\}$. Whether the headline result uses the 12-month window is still open.

## Data

The raw Excel extracts live under `data/` on the local machine. That directory is gitignored and is not redistributed with this repository. WRDS / Compustat downloads are covered by the WRDS academic terms of use and must not be posted publicly. Yahoo Finance extracts are also kept local. Place the three workbooks below in `data/` before running `check.py` or `main.py`.

| Local file | Source | Sheets the loader reads |
|---|---|---|
| `raw_data_AIQ holdings and ETF prices.xlsx` | SEC N-PORT-derived AIQ holdings, Yahoo coverage crosswalk, AIQ prices | `SEC derived holdings`, `Yahoo coverage`, `AIQ prices` |
| `raw_data_Yahoo_constituent_prices.xlsx` | Yahoo Finance constituent prices | `Yahoo stock prices` |
| `raw_data_WRDS_supplemental_prices.xlsx` | WRDS Compustat North America daily prices for names Yahoo does not cover | `Compustat NA` |

How to obtain each extract:

1. **Holdings and coverage.** Build the SEC-derived holdings sheet from AIQ N-PORT filings (`asOf`, `filedDate`, ISIN, weight). Keep a Yahoo coverage sheet that maps each ISIN to the trading `security_id` used in the price files. Download AIQ's own Yahoo price series into the same workbook. Global X issuer quarter-end files may sit in the workbook for reference; the loader does not read them.
2. **Yahoo constituent prices.** For every coverage ticker marked available, download daily close, adjusted close, and return from 2022-09-30 through 2026-09-30 into `Yahoo stock prices`.
3. **WRDS supplement.** Under an authorized WRDS account, pull Compustat North America daily prices (`tic`, `datadate`, `prccd`, `ajexdi`, `trfd`, and related identifiers) for coverage names Yahoo does not price. Do not commit or publish that extract. Compustat Global is not read.

AIQ, THNQ, and ARTY were compared, with ROBT and BOTZ as backups. AIQ has the most complete holdings history and price coverage from October 2022 through September 2026, so it is the first case. The study extracts were downloaded in October 2026. The counts below are what the current loader reads from a complete local `data/` folder.

- **Holdings:** the SEC-derived sheet has 1,441 rows and 17 snapshots from 2022-05-31 through 2026-05-31. Every filing date is on or after its as-of date, and every row matches an ISIN on the coverage sheet. Global X issuer quarter-end files are in the workbook and are not read.
- **Coverage sheet:** 129 symbols, 115 marked available and 14 unavailable. AIQ is the benchmark, not a constituent.
- **Constituent prices:** the Yahoo sheet has 108,860 rows and 114 tickers from 2022-09-30 through 2026-09-30, quoted in USD and in other currencies. Compustat North America supplies USD prices where the coverage sheet marks a name unavailable. Compustat Global is not read.
- **AIQ prices:** Yahoo Finance, 1,003 rows from 2022-09-30 through 2026-09-30, currency USD, with no missing adjusted close. The first daily return is missing.

The master table has no FX series. A name whose coverage currency is not USD is left out, rather than converted from a local return. Missing prices stay missing; nothing is forward-filled. Tickers change, so `security_id` is the Yahoo or WRDS symbol on the coverage sheet.

## How the raw extracts become the master table

The raw files are not in the shape the study needs. Holdings are periodic snapshots, prices are daily, and the same security is identified differently in each workbook. `preprocess_master_frame` applies the following rules and returns one table: one row is one USD security on one price date. The current build has 87,081 rows, 26 columns, and 95 securities, from 2022-09-30 through 2026-09-30.

**Holdings use the SEC filing date.** Global X quarter-end files are not read. Those files say which stocks the fund held on 31 March, 30 June, 30 September, or 31 December, but they do not say when the public could see them. `Last-Modified` only shows that the server later changed the file, so it is not a publication date. The SEC extract has both dates: `asOf` becomes `holdings_as_of`, and `filedDate` becomes `publication_at`. Each price date carries the latest snapshot whose filing date is on or before that price date. If the security is absent from that snapshot, weight, CUSIP, and holding name stay empty and `in_published_holdings` is false. Reported weights stay in percent of net assets. They are not rescaled to sum to one.

The two holdings sources are different calendars, not two copies of the same file. SEC dates fall on February, May, August, and November month-ends. Global X dates fall on March, June, September, and December month-ends. From September 2024 both sources exist and the lists can look similar, but the fund may have traded during the month between them.

All twelve quarter-start rebalances from 2023-10-02 through 2026-07-01 have a filing that was already public. None of those filings is the calendar quarter-end portfolio. Each as-of date is about four months before the rebalance. Rebalancing on the quarter-end day itself uses the same filing, because the next filing is published after the quarter-end. October 2023 through September 2024 has no Global X quarter-end file, so skipping those files does not remove that part of the sample.

Adding Global X later requires one new fact per file: `publication_at`, the first time that file was publicly visible. On the holdings sheet every row from the same quarter-end file carries the same value. A file may be used only after that timestamp. The download date and `Last-Modified` do not fill the field. An empty value means that quarter-end file stays unused.

**One security id comes from the Yahoo coverage sheet.** SEC rows have an ISIN and no trading symbol. Price rows use a Yahoo or WRDS symbol. The coverage sheet is the crosswalk, and `security_id` is that symbol. Several ISINs separated by semicolons belong to one security. If one snapshot lists more than one of those ISINs, their disclosed weights are summed. Values of `N/A` become missing. AIQ itself stays in the benchmark columns and is not a constituent. A missing issuer ticker or exchange is left missing.

**Local-currency listings stay out.** No FX table is used. A coverage currency other than USD is dropped. A blank coverage currency can still enter when the Compustat North America row is USD. A name Yahoo no longer prices is kept only when that Compustat extract has a USD series, which is how delisted names such as TWTR and SPLK enter. Of the coverage names other than AIQ, 33 are absent from the master table: 29 because the quotation is not USD, three because no USD price was found (`6502.T`, `AVV.L`, `MNDT`), and `SPCX` because it is not in the SEC USD holdings. Compustat Global is not read.

**The two price extracts share one set of columns.** Yahoo rows keep close, adjusted close, and daily return, with `return_basis` equal to `adjusted_close`. WRDS rows use `prccd` as close and `prccd / ajexdi` as adjusted close. When the total-return factor `trfd` is present, the daily return uses it. Abiomed and CyberArk have no factor, so their return is the price change and `quality_flag` says so. The first day of each series has no previous price, so its return stays missing. A series stops on its last available price. No forward fill is applied, and no cash proceeds are invented.

**The benchmark is attached by date.** AIQ close and return are merged onto each row. A date on which the stock did not trade is not inserted. If the AIQ return is missing after the first benchmark day, that row is flagged.

The build stops on a duplicate security-date row, an ISIN that maps to two security ids, a repeated ISIN inside one snapshot, a non-USD row in the master table, or an infinite numeric value.

## Run

Use Python 3.14. Gurobi 13 publishes wheels through 3.14 and not for 3.15.

```text
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python check.py
.venv/bin/python main.py
```

`config.py` chooses the solver, the quarters, and the windows. An empty `quarters` tuple runs all twelve decision dates, from 2023Q4 through 2026Q3. `window_months` accepts only the prepared lengths 3, 6, and 12, and a single length is written `(3,)`. The two filters combine. No match raises `ValueError`.

`check.py` writes the audit tables under `output/check/`. `main.py` does not read that directory. It reloads the Excel extracts, builds the return matrices in memory, and solves them.

Each run creates `output/solve/<run_id>/`:

| File | Contents |
|---|---|
| `run.json` | Python version, solver and version, windows, $\alpha$, $\ell$, $u$, time limit, and MIP-gap setting |
| `summary.csv` | Status, training mean squared error, HHI, effective holdings, MIP gap, best bound, and runtime |
| `checks.csv` | Budget, bounds, cardinality, and objective checks, each cell as `value (passed)` |
| `weights.csv` | Every eligible name, in solve order, with weight and activation |

Rows are kept in calendar order, older quarters first. Within a quarter the shorter window comes first. `status` of `time_limit` or `iteration_limit` means the search stopped with a feasible incumbent.

## Solvers

`config.solver` is `"gurobi"`, `"scip"`, or `"bonmin"`.

The pip-installed Gurobi license is size-limited to 200 variables. This formulation uses two variables per eligible name plus one error variable per training day, so a 3-month window fits and a 6- or 12-month window is rejected at `optimize`. SCIP and Bonmin can run those longer windows. SCIP reports the MIP gap and best bound on a stopped search. Bonmin reports them when the solve is proven optimal; otherwise those two summary cells stay empty.

Bonmin is the executable at `.tools/bonmin/bin/bonmin`, or the path in `BONMIN_EXECUTABLE`. Pyomo calls it.

```text
SPOT/
├── config.py
├── main.py
├── check.py
├── test_data.py
├── model/miqp.py
├── src/
├── data/            # local only; gitignored
├── requirements.txt
└── output/
```

## Decisions still open

Fixed already: the project name, the AIQ benchmark, long-only fully invested tracking mean squared error, $\alpha=0.10$, $\ell=0$, $u=1$, quarterly rebalancing, the 3/6/12-month windows on one eligible universe, and the exclusion of transaction costs from the base model. The current master table also fixes USD-only membership with no FX conversion, SEC filing dates as the publication time, no price forward fill, and eligibility that requires a return on every AIQ trading day in the prior twelve months.

Still able to change whether the conclusions are valid:

1. An information cutoff separate from the SEC filing date, the market timezone, and the execution price.
2. A check of each SEC-derived filing date against the original EDGAR filing.
3. Delisting proceeds in the backtest. The master table stops a series at its last price and does not create cash.
4. The time limit used for reported solves, and which quarters are held out if earlier tests are used for tuning. The code records the limit it was given.

The backtest, the equal-weight and disclosure-weight comparisons, and the selection-ratio grid are not implemented.
