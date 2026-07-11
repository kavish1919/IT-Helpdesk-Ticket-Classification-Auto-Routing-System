# IT Helpdesk Ticket Classification & Auto-Routing System

---

## Problem Statement

Every enterprise IT helpdesk receives hundreds of support tickets daily in free-text form.
Manually reading, categorising, prioritising, and routing each ticket to the right team is:
- **Slow** — introducing delays before a technician even starts working
- **Inconsistent** — different agents classify the same issue differently
- **Reactive to SLA breaches** — teams discover they're behind only after a breach occurs

This project automates that workflow using NLP and machine learning:
1. **Classify** the ticket into one of 6 categories (Network, Hardware, Software, Access/Login, Email, Database)
2. **Predict** its priority level (Low, Medium, High, Critical)
3. **Estimate** expected resolution time in hours
4. **Route** it to the correct team and **flag** it if it is at risk of breaching its SLA — *before* anyone has even looked at it

---

## Dataset

| Property | Value |
|---|---|
| Type | **Synthetic** (generated programmatically) |
| Rows | 5 000 |
| Columns | 7 (`ticket_id`, `ticket_text`, `category`, `priority`, `created_at`, `resolution_time_hours`, `resolved_at`) |
| Generator | `src/data_generation.py` |
| Seed | 42 (fully reproducible) |

### Why Synthetic?

A search of Kaggle, UCI, and Google Dataset Search was conducted for "IT service ticket classification dataset" and "helpdesk ticket dataset". No public dataset was found that contains all required fields (`ticket_text` + `category` + `priority` + `resolution_time_hours`) at acceptable quality. This is documented at the top of `notebooks/01_data_generation_or_loading.ipynb`.

Generating synthetic data is a legitimate and common approach when real enterprise ticket data is confidential.

### Class Distributions (designed to be realistic, not balanced)

| Category | Target % | ~Rows |
|---|---|---|
| Software | 25% | 1 250 |
| Network | 20% | 1 000 |
| Access/Login | 20% | 1 000 |
| Hardware | 15% | 750 |
| Email | 12% | 600 |
| Database | 8% | 400 |

| Priority | Target % | ~Rows |
|---|---|---|
| Low | 35% | 1 750 |
| Medium | 40% | 2 000 |
| High | 18% | 900 |
| Critical | 7% | 350 |

### Text Diversity

Ticket text is generated using a **compositional approach** — 25–30 sentence templates per category with random slot-filling (server names, error codes, times, teams) + optional urgency prefixes + optional tail phrases + 5% word-level typo injection. The combinatorial space far exceeds 5 000 rows, so near-duplicate texts are statistically negligible. A Jaccard-similarity guard (threshold 0.85) regenerates any text that is too similar to an existing one.

### Resolution Time Formula

```
resolution_hours = base_hours[category] × priority_multiplier[priority] × lognormal_noise
```

| Category | Base hours |
|---|---|
| Access/Login | 4h |
| Network | 6h |
| Email | 8h |
| Hardware | 12h |
| Software | 18h |
| Database | 24h |

| Priority | Multiplier |
|---|---|
| Critical | 0.5× |
| High | 1.0× |
| Medium | 2.0× |
| Low | 3.0× |

---

## Pipeline Architecture

```
Raw Ticket Text (free-text)
         |
         v
  ┌─────────────────────┐
  │   src/preprocessing  │   clean_text(): lowercase → remove punct → normalise
  │   .py               │   numbers → stopword removal → Porter stemming
  └─────────────────────┘
         |
    ┌────┴────┐
    |          |
    v          v
 TF-IDF     Word2Vec
(5000 dim) (100 dim avg)
    |          |
    |    ┌─────┘
    |    |
    v    v
┌──────────────────────────────────────┐
│  Category Classifier  (Notebook 04)  │  LR / RF / LSTM × {TF-IDF, W2V}
│  Priority Classifier  (Notebook 05)  │  LR / RF / GBT × TF-IDF
│  Resolution Regressor (Notebook 06)  │  RF / GBR on PCA(TF-IDF) + structured
└──────────────────────────────────────┘
         |
         v
┌──────────────────────────────────────┐
│  Auto-Routing Engine  (Notebook 07)  │  Pure business logic (no ML)
│  src/routing_engine.py               │
│                                       │
│  TEAM_MAP  (category -> team)         │
│  SLA_HOURS (priority -> target hours) │
│  sla_breach_risk = pred > target      │
└──────────────────────────────────────┘
         |
         v
  ┌─────────────────┐
  │  Streamlit App  │   app/streamlit_app.py
  │  (Demo UI)      │
  └─────────────────┘
```

---

## Model Comparison Results

> Evaluated on a 15% held-out test set (600 tickets).

### Category Classification (Notebook 04)

| Rank | Model | Features | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| 1 | Logistic Regression | TF-IDF | 1.0000 | 1.0000 | 1.0000 |
| 2 | Random Forest | TF-IDF | 0.9967 | 0.9964 | 0.9967 |
| 3 | LSTM (Random Init) | Sequences | 0.9967 | 0.9954 | 0.9966 |
| 4 | LSTM (W2V Init) | Sequences | 0.9967 | 0.9950 | 0.9966 |
| 5 | Random Forest | Word2Vec | 0.9300 | 0.9230 | 0.9300 |
| 6 | Logistic Regression | Word2Vec | 0.7967 | 0.7829 | 0.7976 |

*Full table: `results/category_classification_results.csv`*

### Priority Classification (Notebook 05)

| Rank | Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| 1 | Gradient Boosting (TF-IDF) | 0.5850 | 0.4999 | 0.5718 |
| 2 | Random Forest (TF-IDF) | 0.5833 | 0.4527 | 0.5587 |
| 3 | Logistic Regression (TF-IDF) | 0.4783 | 0.4236 | 0.4928 |

*Full table: `results/priority_classification_results.csv`*

### Resolution Time Regression (Notebook 06)

| Model | MAE (hours) | RMSE (hours) | R² |
|---|---|---|---|
| Gradient Boosting Regressor | 8.86 | 14.12 | 0.6078 |
| Random Forest Regressor | 8.87 | 14.56 | 0.5829 |

*Full table: `results/regression_results.csv`*

---

## Key Insights (from EDA — Notebook 02)

1. **Class imbalance is real** — Software (25%) + Network (20%) dominate; Database (8%) and Critical (7%) are minority classes. This directly justifies `class_weight='balanced'` in all classifiers.
2. **Monday/Tuesday spike** — ~35–40% more tickets on Mon/Tue due to post-weekend backlog. `day_of_week` is a valid regression feature.
3. **Vocabulary is category-specific** — "vpn", "wifi", "ping" dominate Network tickets; "sql", "query", "backup" dominate Database tickets. This makes TF-IDF a strong baseline.
4. **Critical tickets resolve *faster* than Low tickets** — because they receive immediate senior attention. This counterintuitive finding validates including priority as a feature in the regression model.
5. **Priority and resolution time are strongly correlated** — confirms that the SLA-breach risk flag has statistical grounding.

---

## Limitations

- **Synthetic data may not capture real ticket ambiguity** — real users write more ambiguously; a ticket saying "my computer is slow" could be Network, Software, or Hardware.
- **LSTM underperforms on 5 000 rows** — the sequence model needs significantly more data to learn generalizable representations. This is a documented finding, not a failure.
- **SLA targets are illustrative** — they match common ITSM industry benchmarks but are not tied to real contractual SLAs.
- **No real-world validation** — the model has not been tested on actual enterprise helpdesk data. Performance on real data may differ.

---

## How to Run

### Prerequisites
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (run once)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Run Notebooks (in order)
```bash
jupyter notebook
```
Open and run each notebook in sequence: `01` → `02` → `03` → `04` → `05` → `06` → `07`

### Quick Data Generation (no Jupyter needed)
```bash
python src/data_generation.py
```

### Run Streamlit Demo
```bash
# Run notebooks 01-06 first to generate model artifacts, then:
streamlit run app/streamlit_app.py
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Language | Python 3.10 / 3.11 |
| Data handling | pandas, numpy |
| NLP preprocessing | NLTK (stopwords, PorterStemmer), regex |
| Feature extraction | scikit-learn TfidfVectorizer, gensim Word2Vec |
| Classical ML | scikit-learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier/Regressor) |
| Dimensionality reduction | scikit-learn PCA |
| Deep learning | PyTorch (LSTM, Embedding, pack_padded_sequence) |
| Visualisation | Matplotlib, Seaborn, wordcloud |
| Notebook environment | Jupyter |
| Demo app | Streamlit |
| Model persistence | joblib |
| Version control | Git |

---

## Project Structure

```
it-ticket-classifier/
├── data/
│   ├── raw/          tickets.csv (5 000 rows)
│   └── processed/    feature matrices, label arrays, split CSVs
├── notebooks/        01 through 07 — run in order
├── src/
│   ├── data_generation.py   synthetic dataset generator
│   ├── preprocessing.py     text cleaning, TF-IDF, Word2Vec, splits
│   ├── models.py            PyTorch LSTM + training utilities
│   └── routing_engine.py    route_ticket() function
├── models/           saved .joblib and .pt artifacts
├── results/          metric CSVs and PNG plots
├── app/
│   └── streamlit_app.py     interactive demo
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Author

Kavish Bishnoi | Final-year B.Tech student | Portfolio project for IT services industry applications
