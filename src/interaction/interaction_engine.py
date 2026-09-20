from dataclasses import dataclass
from math import sqrt


@dataclass
class ObjectInfo:
    name: str
    bbox: tuple[float, float, float, float]
    confidence: float
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )


@dataclass
class HandInfo:
    name: str
    position: tuple[float, float]
    confidence: float


@dataclass
class InteractionFeatures:
    left_hand_red_distance: float
    right_hand_red_distance: float
    left_hand_yellow_distance: float
    right_hand_yellow_distance: float
    left_hand_experiment_box_distance: float
    right_hand_experiment_box_distance: float
    left_hand_speed: float
    right_hand_speed: float


class InteractionEngine:

    def __init__(self):
        self.previous_left_hand = None
        self.previous_right_hand = None

    @staticmethod
    def distance(point_a, point_b) -> float:
        dx = point_a[0] - point_b[0]
        dy = point_a[1] - point_b[1]

        return sqrt(
            dx * dx + dy * dy
        )

    @staticmethod
    def get_object(objects, name):
        for obj in objects:
            if obj.name == name:
                return obj

        return None

    def calculate_speed(
        self,
        current_position,
        previous_position,
    ) -> float:

        if (
            current_position is None
            or previous_position is None
        ):
            return 0.0

        return self.distance(
            current_position,
            previous_position,
        )

    def extract(
        self,
        hands,
        objects,
    ) -> InteractionFeatures:

        left_hand = None
        right_hand = None

        for hand in hands:

            if hand.name == "left":
                left_hand = hand

            elif hand.name == "right":
                right_hand = hand

        red_box = self.get_object(
            objects,
            "red_box",
        )

        yellow_box = self.get_object(
            objects,
            "yellow_box",
        )

        experiment_box = self.get_object(
            objects,
            "experiment_box",
        )

        left_position = (
            left_hand.position
            if left_hand
            else None
        )

        right_position = (
            right_hand.position
            if right_hand
            else None
        )

        left_speed = self.calculate_speed(
            left_position,
            self.previous_left_hand,
        )

        right_speed = self.calculate_speed(
            right_position,
            self.previous_right_hand,
        )

        def hand_object_distance(
            hand_position,
            obj,
        ):

            if (
                hand_position is None
                or obj is None
            ):
                return 0.0

            return self.distance(
                hand_position,
                obj.center,
            )

        features = InteractionFeatures(
            left_hand_red_distance=
                hand_object_distance(
                    left_position,
                    red_box,
                ),

            right_hand_red_distance=
                hand_object_distance(
                    right_position,
                    red_box,
                ),

            left_hand_yellow_distance=
                hand_object_distance(
                    left_position,
                    yellow_box,
                ),

            right_hand_yellow_distance=
                hand_object_distance(
                    right_position,
                    yellow_box,
                ),

            left_hand_experiment_box_distance=
                hand_object_distance(
                    left_position,
                    experiment_box,
                ),

            right_hand_experiment_box_distance=
                hand_object_distance(
                    right_position,
                    experiment_box,
                ),

            left_hand_speed=left_speed,

            right_hand_speed=right_speed,
        )

        self.previous_left_hand = left_position
        self.previous_right_hand = right_position

        return features