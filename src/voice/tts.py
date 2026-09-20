import subprocess
from pathlib import Path


class OfflineTTS:

    def __init__(
        self,
        model_path: str,
        output_dir: str = "data/audio",
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Piper model not found: "
                f"{self.model_path}"
            )

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def synthesize(
        self,
        text: str,
        output_path: str,
    ):

        if not text.strip():
            raise ValueError(
                "Text cannot be empty"
            )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            "piper",
            "--model",
            str(self.model_path),
            "--output_file",
            str(output_path),
        ]

        process = subprocess.run(
            command,
            input=text,
            text=True,
            capture_output=True,
        )

        if process.returncode != 0:
            raise RuntimeError(
                "Piper TTS failed:\n"
                + process.stderr
            )

        return output_path

    def speak(
        self,
        text: str,
        filename: str = "alert.wav",
    ):

        output_path = (
            self.output_dir
            / filename
        )

        return self.synthesize(
            text,
            output_path,
        )