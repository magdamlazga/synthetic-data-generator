"""The generation loop, with a fake model instead of a real API."""

import itertools

import pytest

import generator
from schema import Column, DatasetSpec

SPEC = DatasetSpec("Test messages", [Column("message", "text"), Column("mood", "category", options=["calm", "angry"])])


class FakeModel:
    """Returns scenarios for the brainstorm, then batches of rows: some unique, some broken, some repeated."""

    def __init__(self, bad_every=0, repeat=False, fail_batches=0):
        self.counter = itertools.count()
        self.prompts = []
        self.bad_every, self.repeat, self.fail_batches = bad_every, repeat, fail_batches

    def __call__(self, model_id, system, user, json_schema, temperature=0.9):
        self.prompts.append(user)
        if "sub-scenarios" in user:
            return {"scenarios": ["new customers", "returning customers", "business buyers"]}, 10, 5
        if self.fail_batches:
            self.fail_batches -= 1
            raise TimeoutError("model too slow")
        size = int(user.split("Write exactly ")[1].split(" rows")[0])  # matches the prompt wording
        rows = []
        for _ in range(size):
            i = next(self.counter)
            if self.bad_every and i % self.bad_every == 0:
                rows.append({"message": "broken", "mood": "furious"})  # not an allowed value
            else:
                rows.append({"message": "same" if self.repeat else f"message {i}", "mood": "calm"})
        return {"rows": rows}, 100, 50


@pytest.fixture
def fake(monkeypatch):
    def install(**kwargs):
        model = FakeModel(**kwargs)
        monkeypatch.setattr(generator, "call_model", model)
        return model
    return install


def test_generates_the_requested_number_of_rows(fake):
    model = fake()
    result = generator.generate_all(SPEC, "ollama/fake", 25, seed=1)
    assert result.done and len(result.rows) == 25
    assert result.invalid == 0 and result.duplicates == 0
    assert {r["_scenario"] for r in result.rows} <= {"new customers", "returning customers", "business buyers"}
    assert result.input_tokens == 10 + 3 * 100  # brainstorm + three batches (10, 10, 5)
    assert "Write exactly 5 rows" in model.prompts[-1]


def test_invalid_rows_are_replaced_with_extra_batches(fake):
    fake(bad_every=4)
    result = generator.generate_all(SPEC, "ollama/fake", 20, seed=1)
    assert len(result.rows) == 20
    assert result.invalid > 0
    assert all(r["mood"] == "calm" for r in result.rows)


def test_duplicates_are_dropped_and_generation_stops_at_the_limit(fake):
    fake(repeat=True)
    result = generator.generate_all(SPEC, "ollama/fake", 20, seed=1)
    assert len(result.rows) == 1          # only one unique row exists
    assert result.duplicates > 0
    assert result.batches == 2 + 2        # 2 planned batches + the extra allowance, then it gives up


def test_failed_batches_are_counted_not_fatal(fake):
    fake(fail_batches=1)
    result = generator.generate_all(SPEC, "ollama/fake", 20, seed=1)
    assert result.failed_batches == 1
    assert len(result.rows) == 20
    assert "TimeoutError" in result.errors[0]


def test_edge_case_share_is_respected(fake):
    model = fake()
    result = generator.generate_all(SPEC, "ollama/fake", 100, edge_ratio=0.3, seed=1)
    edge = sum(r["_edge_case"] for r in result.rows)
    assert edge == 30
    assert sum("EDGE CASES" in p for p in model.prompts) == 3


def test_later_batches_are_told_what_not_to_repeat(fake):
    model = fake()
    generator.generate_all(SPEC, "ollama/fake", 20, seed=1)
    assert "Do not repeat" not in model.prompts[1]
    assert "Do not repeat" in model.prompts[2] and "message " in model.prompts[2]


def test_cost_is_zero_for_local_models_and_unknown_without_a_price():
    assert generator.GenerationResult("ollama/llama3.2", 1).cost == 0.0
    assert generator.GenerationResult("openai/some-new-model", 1).cost is None
