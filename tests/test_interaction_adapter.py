from src.interaction.adapter import PerceptionAdapter
from src.interaction.interaction_engine import (
    InteractionEngine,
)
from src.perception.tracker import TrackedObject


def main():

    adapter = PerceptionAdapter()

    tracked_objects = [
        TrackedObject(
            track_id=1,
            class_id=0,
            class_name="red_box",
            confidence=0.95,
            bbox=(100, 100, 200, 200),
        ),

        TrackedObject(
            track_id=2,
            class_id=1,
            class_name="yellow_box",
            confidence=0.92,
            bbox=(300, 100, 400, 200),
        ),

        TrackedObject(
            track_id=3,
            class_id=2,
            class_name="experiment_box",
            confidence=0.97,
            bbox=(200, 300, 500, 500),
        ),
    ]

    objects = adapter.objects_from_tracking(
        tracked_objects
    )

    assert len(objects) == 3
    assert objects[0].name == "red_box"
    assert objects[0].track_id == 1

    engine = InteractionEngine()

    features = engine.extract(
        hands=[],
        objects=objects,
    )

    print("Interaction features:")

    print(
        "left_hand_red_distance:",
        features.left_hand_red_distance,
    )

    print(
        "right_hand_red_distance:",
        features.right_hand_red_distance,
    )

    print(
        "left_hand_yellow_distance:",
        features.left_hand_yellow_distance,
    )

    print(
        "right_hand_yellow_distance:",
        features.right_hand_yellow_distance,
    )

    print(
        "left_hand_experiment_box_distance:",
        features.left_hand_experiment_box_distance,
    )

    print(
        "right_hand_experiment_box_distance:",
        features.right_hand_experiment_box_distance,
    )

    print(
        "left_hand_speed:",
        features.left_hand_speed,
    )

    print(
        "right_hand_speed:",
        features.right_hand_speed,
    )

    print()
    print("Interaction adapter test passed.")


if __name__ == "__main__":
    main()