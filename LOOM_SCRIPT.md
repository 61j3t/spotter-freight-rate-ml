# Loom Script — Freight Rate Prediction (2–3 min)

Speak at a normal pace. Each block is ~1 line of talking. **Cue** = what to show
on screen. Total ≈ 2:45.

---

### 0:00 — Intro  ·  **Cue: REPORT.pdf title page**
"Hi, this is my walkthrough of the freight-rate prediction assessment. I'll cover
what I found in the data, the quality issues, my model, and how I validated it."

### 0:15 — Data findings  ·  **Cue: REPORT.pdf 'Data' section / a quick look at train_test.csv**
"The training data is about 48,000 loads from January to October 2025. The
validation set is November and December — so this is really a *forecasting*
problem: predict later months from earlier ones."
"The rate is driven mostly by distance and a market-index signal. In fact, a
plain linear model was as strong as gradient boosting — so the pricing is close
to linear, not full of weird nonlinear jumps."

### 0:45 — Data-quality issues  ·  **Cue: the geo-audit lines in REPORT.pdf**
"Two quality findings. First, the coordinates are *synthetic* — every city has
one fixed lat/long, but it's geographically wrong. I confirmed they match the
given distance at r = 0.9995, so they're consistent but redundant. I didn't rely
on them."
"Second, no missing or negative rates — the data is clean. But every model family
converged to the same error floor, which tells me there's irreducible noise in
the rate that no feature can explain."

### 1:20 — Model choice  ·  **Cue: results/final_selection.csv (the leaderboard)**
"I ran a full bake-off: Ridge, random forest, XGBoost, LightGBM, CatBoost, an MLP,
and TabPFN on Kaggle. The biggest single win was predicting *log* of the rate —
that cut my typical error, MAPE, from about 8.4% down to 5.6%. TabPFN wasn't
competitive here — it favors small datasets, and this one is large and linear."
"My final submission is an equal-weight blend of the top three — CatBoost, Ridge,
and XGBoost — which just edged out any single model."

### 1:50 — Validation & split  ·  **Cue: src/data.py `time_folds`**
"For validation I used a *time-based* split, never a random one — that would leak
the future. Each fold trains on earlier months and tests on a later month:
holding out August, then September, then October. Target encoding for lanes and
cities is fitted *inside* each fold, so there's no leakage."

### 2:15 — Two models + December  ·  **Cue: the December chart**
"I trained two models: a full-feature one for the graded 12k loads, and a
reduced one for December, which only has six columns. For December I used
calendar features — weekday, seasonality — so the model doesn't flatline on a
future month. You can see the weekly rhythm here: mid-week peaks, weekend dips."

### 2:35 — Code walkthrough  ·  **Cue: scroll src/finalize.py briefly**
"The code is modular: `features_v2.py` builds features and the leak-safe target
encoder, and `finalize.py` picks best-single-versus-blend, refits on all the
data, writes the predictions, and runs the provided scorer. Thanks for watching."

---

**Recording tips**
- Have these tabs open in order: REPORT.pdf, `results/final_selection.csv`,
  `src/data.py`, the December chart, `src/finalize.py`.
- If you run long, cut the code-walkthrough block to one sentence.
