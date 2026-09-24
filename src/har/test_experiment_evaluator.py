from src.har.experiment_evaluator import ExperimentEvaluator


def run_test(name, predictions):
    evaluator = ExperimentEvaluator()

    print(f"\n=== {name} ===")

    for prediction in predictions:
        result = evaluator.update(prediction)

        if result["confirmed_action"]:
            print(
                f"Prediction: {prediction:<22} "
                f"CONFIRMED: {result['confirmed_action']:<22} "
                f"STATUS: {result['status']}"
            )

    print(f"Final status: {evaluator.state_machine.status}")


def main():
    correct = [
        "LEFT_HAND_UP",
        "LEFT_HAND_UP",
        "LEFT_HAND_UP",
        "RIGHT_HAND_UP",
        "RIGHT_HAND_UP",
        "RIGHT_HAND_UP",
        "PICK_BLUE_BOX",
        "PICK_BLUE_BOX",
        "PICK_BLUE_BOX",
        "PUT_DOWN_BLUE_BOX",
        "PUT_DOWN_BLUE_BOX",
        "PUT_DOWN_BLUE_BOX",
        "PICK_BLACK_BOX",
        "PICK_BLACK_BOX",
        "PICK_BLACK_BOX",
        "PUT_DOWN_BLACK_BOX",
        "PUT_DOWN_BLACK_BOX",
        "PUT_DOWN_BLACK_BOX",
    ]

    skip_order = [
        "LEFT_HAND_UP",
        "LEFT_HAND_UP",
        "LEFT_HAND_UP",
        "RIGHT_HAND_UP",
        "RIGHT_HAND_UP",
        "RIGHT_HAND_UP",
        "PUT_DOWN_BLUE_BOX",
        "PUT_DOWN_BLUE_BOX",
        "PUT_DOWN_BLUE_BOX",
    ]

    incomplete = [
        "LEFT_HAND_UP",
        "LEFT_HAND_UP",
        "LEFT_HAND_UP",
        "RIGHT_HAND_UP",
        "RIGHT_HAND_UP",
        "RIGHT_HAND_UP",
        "PICK_BLUE_BOX",
        "PICK_BLUE_BOX",
        "PICK_BLUE_BOX",
    ]

    run_test("CORRECT", correct)
    run_test("SKIP_ORDER", skip_order)
    run_test("INCOMPLETE", incomplete)


if __name__ == "__main__":
    main()