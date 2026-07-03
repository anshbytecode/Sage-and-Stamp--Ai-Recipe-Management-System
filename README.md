# Sage & Stamp — AI-Powered Recipe Management System

A full Python + Flask recipe manager with real, locally-running machine learning
features — no external AI API calls, no API keys needed.

## Features

**Core recipe management**
- Add / view / edit / delete recipes (title, cuisine, ingredients, instructions,
  prep/cook time, servings, calories, tags)
- SQLite storage (`data/recipes.db`), no external database required

**AI / ML features** (see `ml_engine.py`)
1. **Content-based recommender** — TF-IDF vectorization of each recipe's
   ingredients + method, ranked with cosine similarity, powers the
   "You might also like" panel on every recipe page.
2. **Cuisine classifier** — a Multinomial Naive Bayes model trained on your
   labeled recipes. Leave "Cuisine" blank when adding a recipe and it will be
   predicted automatically from the ingredient list, with a confidence score.
3. **"What Can I Cook?" ingredient matcher** — give it a pantry list and it
   ranks every recipe by ingredient overlap, showing what you have and what's
   missing.
4. **Recipe clustering ("Discover")** — KMeans over the TF-IDF space
   automatically groups your whole collection into flavor families, with
   auto-generated labels from each cluster's top terms — no manual tagging.
5. **Smart search** — the home page search box ranks recipes by TF-IDF/cosine
   similarity to your query instead of plain substring matching, so
   "spicy creamy chicken" surfaces the right dishes even without exact
   keyword hits.

All models retrain in memory (`engine.refresh()`) every time a recipe is
added, edited, or deleted, so the system stays current as your collection
grows — the more recipes you add, the smarter the recommendations and
predictions get.

## Project structure
```
recipe_ai/
├── app.py            # Flask routes / web app
├── database.py       # SQLite schema + CRUD helpers
├── ml_engine.py       # All ML models (recommender, classifier, matcher, clustering, search)
├── seed_data.py       # Seeds 25 diverse sample recipes (needed for the classifier/clustering to have enough data)
├── requirements.txt
├── templates/         # Jinja2 HTML templates
├── static/style.css   # Styling
└── data/recipes.db    # Created on first run
```

## Setup

```bash
cd recipe_ai
pip install -r requirements.txt
python seed_data.py      # populates ~25 starter recipes (skips if DB already has data)
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Notes
- The cuisine classifier and clustering need a handful of recipes with varied
  cuisines to be useful — the seed data provides a good starting mix across
  8 cuisines. Delete `data/recipes.db` and re-run `seed_data.py` any time you
  want to reset to the starter set.
- To use your own recipe set instead, just skip `seed_data.py` and add
  recipes through the "Add Recipe" page — the ML features activate
  automatically as soon as there's enough data (the cuisine classifier needs
  at least 4 labeled recipes across 2+ cuisines; clustering needs at least 4
  recipes total).
