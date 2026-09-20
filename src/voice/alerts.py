from src.voice.tts import OfflineTTS


class ExperimentVoiceAlerts:

    def __init__(
        self,
        tts: OfflineTTS,
    ):
        self.tts = tts

    def correct(
        self,
        next_step: str,
    ):

        text = (
            f"Correct. "
            f"Next step: {next_step}."
        )

        return self.tts.speak(
            text,
            filename="correct.wav",
        )

    def wrong_sequence(
        self,
        expected_step: str,
    ):

        text = (
            f"Incorrect sequence. "
            f"Please perform: "
            f"{expected_step}."
        )

        return self.tts.speak(
            text,
            filename="wrong_sequence.wav",
        )

    def completed(self):

        return self.tts.speak(
            "Experiment completed successfully.",
            filename="completed.wav",
        )