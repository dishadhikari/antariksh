from src.har.smoother import PredictionSmoother


def main():

    smoother = PredictionSmoother(
        window_size=5,
        confidence_threshold=0.60,
        min_votes=3,
    )

    result = smoother.update(
        "PICK_RED",
        0.90,
    )

    assert result["event"] is None

    result = smoother.update(
        "PICK_RED",
        0.91,
    )

    assert result["event"] is None

    result = smoother.update(
        "PICK_RED",
        0.92,
    )

    assert result["stable"] is True
    assert result["action"] == "PICK_RED"
    assert result["event"] == "PICK_RED"

    # Same action should not generate another event.
    result = smoother.update(
        "PICK_RED",
        0.93,
    )

    assert result["event"] is None

    # Low confidence should be ignored.
    result = smoother.update(
        "PICK_YELLOW",
        0.20,
    )

    assert result["event"] is None

    smoother.reset()

    assert len(smoother.predictions) == 0
    assert smoother.last_emitted_action is None

    print("HAR smoother test passed.")


if __name__ == "__main__":
    main()