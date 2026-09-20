from src.interaction.interaction_engine import (
    InteractionEngine,
    ObjectInfo,
    HandInfo,
)


def main():

    engine = InteractionEngine()

    objects = [

        ObjectInfo(
            name="red_box",
            bbox=(350, 250, 450, 350),
            confidence=0.95,
        ),

        ObjectInfo(
            name="yellow_box",
            bbox=(550, 250, 650, 350),
            confidence=0.94,
        ),

        ObjectInfo(
            name="experiment_box",
            bbox=(250, 150, 750, 500),
            confidence=0.90,
        ),
    ]

    hands = [

        HandInfo(
            name="right",
            position=(420, 300),
            confidence=0.90,
        ),

        HandInfo(
            name="left",
            position=(700, 300),
            confidence=0.88,
        ),
    ]

    features = engine.extract(
        hands=hands,
        objects=objects,
    )

    print("\nInteraction features:")
    print(features)


if __name__ == "__main__":
    main()