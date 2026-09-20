from pathlib import Path

from src.voice.tts import OfflineTTS


def main():

    model_path = Path(
        "models/voice/test.onnx"
    )

    # We only test initialization failure handling
    # here because a real Piper voice model is required
    # for synthesis.
    if not model_path.exists():

        try:

            OfflineTTS(
                model_path=str(model_path)
            )

        except FileNotFoundError:

            print(
                "Correctly detected missing "
                "Piper model."
            )

            print(
                "TTS infrastructure test passed."
            )

            return

        raise AssertionError(
            "Missing Piper model should "
            "raise FileNotFoundError"
        )


if __name__ == "__main__":
    main()