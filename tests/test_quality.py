from quality import category_balance, distinct_2, quality_report, text_similarity
from schema import Column

COLUMNS = [Column("review", "text"), Column("sentiment", "category", options=["positive", "negative"]),
           Column("rating", "integer", minimum=1, maximum=5)]


def rows(texts, sentiments, ratings, edge=False):
    return [{"review": t, "sentiment": s, "rating": r, "_edge_case": edge} for t, s, r in zip(texts, sentiments, ratings)]


REPETITIVE = rows(["Great product, works well."] * 4 + ["Great product, works fine."] * 4, ["positive"] * 8, [5] * 8)
VARIED = rows([
    "The blender jammed on day two and support never answered my emails.",
    "Makes a decent espresso, though the milk frother is loud.",
    "My kids love the pancakes from this griddle, cleaning is a pain.",
    "Kettle boils fast. Lid hinge snapped after a month, sadly.",
    "Air fryer cut our oil use in half, fries come out crisp.",
    "Toaster browns one side only, returned it.",
    "Honestly the best coffee grinder under 100 euros.",
    "Cord is far too short for my kitchen layout.",
], ["negative", "positive", "positive", "negative", "positive", "negative", "positive", "negative"], [1, 4, 4, 2, 5, 1, 5, 2])


def test_varied_data_scores_higher_than_repetitive_data():
    repetitive = quality_report(REPETITIVE, COLUMNS)
    varied = quality_report(VARIED, COLUMNS)
    assert varied["diversity_score"] > repetitive["diversity_score"] + 30
    assert varied["mean_similarity"] < repetitive["mean_similarity"]
    assert repetitive["near_duplicate_pairs"] > 0.4


def test_distinct_2():
    assert distinct_2(["a b c", "a b c"]) == 0.5  # 4 word pairs, 2 unique
    assert distinct_2(["one", ""]) is None       # no word pairs at all


def test_identical_texts_are_near_duplicates():
    mean, near = text_similarity(["the same sentence"] * 3)
    assert mean > 0.99 and near == 1.0


def test_category_balance():
    assert category_balance(["a", "b", "a", "b"], ["a", "b"])[0] == 1.0
    assert category_balance(["a", "a", "a"], ["a", "b"])[0] == 0.0


def test_validity_duplicates_and_ranges_are_counted():
    data = VARIED[:4] + rows(["Rated it zero because it exploded."], ["negative"], [0], edge=True)
    report = quality_report(data, COLUMNS, invalid=5, duplicates=2)
    assert report["valid_rate"] == 7 / 12          # 5 kept + 2 duplicates were valid, out of 12 returned
    assert report["duplicate_rate"] == 2 / 7
    assert report["edge_case_rows"] == 1
    assert report["numbers"]["rating"]["out_of_range"] == 1
    assert report["numbers"]["rating"]["out_of_range_in_edge_cases"] == 1
    assert report["categories"]["sentiment"]["coverage"] == 1.0
