from src.har.temporal_voting import TemporalVoter


def main():
    voter = TemporalVoter(
        window_size=5,
        minimum_votes=3,
    )

    predictions = [
        "LEFT_HAND_UP",
        "LEFT_HAND_UP",
        "RIGHT_HAND_UP",
        "LEFT_HAND_UP",
        "LEFT_HAND_UP",

        "RIGHT_HAND_UP",
        "RIGHT_HAND_UP",
        "RIGHT_HAND_UP",
        "LEFT_HAND_UP",
        "RIGHT_HAND_UP",
    ]

    for prediction in predictions:
        result = voter.update(prediction)

        print(
            f"Prediction: {prediction:<20} "
            f"Confirmed: {result}"
        )


if __name__ == "__main__":
    main()