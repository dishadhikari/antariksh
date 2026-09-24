from src.har.temporal_voting import TemporalVoter
from src.har.state_machine import ExperimentStateMachine


class ExperimentEvaluator:
    def __init__(self, confidence_threshold=0.60):
        self.confidence_threshold = confidence_threshold

        self.voter = TemporalVoter(
            window_size=5,
            minimum_votes=3,
        )

        self.state_machine = ExperimentStateMachine()

    def update(self, prediction, confidence):

        # Low-confidence predictions do not advance the state,
        # but they still occupy the temporal window.
        
        

        confirmed_action = self.voter.update(prediction)

        if confirmed_action is None:
            return {
                "confirmed_action": None,
                "status": self.state_machine.status,
                "expected": self.state_machine.expected_action(),
            }

        # Don't repeatedly send the same confirmed action
        # to the state machine.
        if (
            self.state_machine.completed
            and self.state_machine.completed[-1] == confirmed_action
        ):
            return {
                "confirmed_action": confirmed_action,
                "status": self.state_machine.status,
                "expected": self.state_machine.expected_action(),
            }

        status = self.state_machine.update(confirmed_action)

        return {
            "confirmed_action": confirmed_action,
            "status": status,
            "expected": self.state_machine.expected_action(),
        }