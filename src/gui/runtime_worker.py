from PySide6.QtCore import QThread, Signal

from src.camera.camera import Camera
from src.perception.pipeline import PerceptionPipeline
from src.har.runtime_pipeline import HARRuntimePipeline
from src.protocol.config import ExperimentConfig
from src.logging.logger import ExperimentLogger
from src.voice.tts import OfflineTTS
from src.voice.alerts import ExperimentVoiceAlerts
from src.protocol.voice_runtime import VoiceRuntimePipeline


class RuntimeWorker(QThread):

    frame_ready = Signal(object)

    status_update = Signal(
        str,
        str,
        float,
        str,
    )

    event_detected = Signal(str)

    error = Signal(str)

    def __init__(
        self,
        camera_source=0,
        detector_model="yolo11n.pt",
        har_model=None,
        voice_model=None,
        device="cpu",
        process_every=3,
    ):
        super().__init__()

        self.camera_source = camera_source
        self.detector_model = detector_model
        self.har_model = har_model
        self.voice_model = voice_model
        self.device = device
        self.process_every = process_every

        self.running = False
        self.frame_count = 0

        self.camera = None
        self.perception = None
        self.har_runtime = None
        self.voice_runtime = None
        self.logger = None

    def setup_runtime(self):

        config = ExperimentConfig(
            "experiments/box_experiment/experiment.yaml"
        )

        self.logger = ExperimentLogger(
            "data/logs/experiment.db"
        )

        self.perception = PerceptionPipeline(
            detector_model=self.detector_model,
            confidence=0.5,
            device=self.device,
        )

        # -----------------------------
        # HAR
        # -----------------------------

        if self.har_model is not None:

            self.har_runtime = HARRuntimePipeline(
                model_path=self.har_model,
                label_map={
                    index: action
                    for index, action
                    in enumerate(config.sequence)
                },
                experiment_id=config.experiment_id,
                sequence=config.sequence,
                logger=self.logger,
                sequence_length=30,
                confidence_threshold=(
                    config.confidence_threshold
                ),
                smoothing_window=(
                    config.temporal_window
                ),
                minimum_votes=(
                    config.minimum_votes
                ),
                device=self.device,
            )

        # -----------------------------
        # Voice
        # -----------------------------

        if self.voice_model is not None:

            tts = OfflineTTS(
                model_path=self.voice_model
            )

            alerts = ExperimentVoiceAlerts(
                tts
            )

            self.voice_runtime = (
                VoiceRuntimePipeline(
                    har_runtime=self.har_runtime,
                    alerts=alerts,
                )
            )

        return config

    def run(self):

        try:

            config = self.setup_runtime()

            self.camera = Camera(
                source=self.camera_source,
                width=640,
                height=480,
                fps=15,
            )

            self.camera.open()

            self.running = True
            self.frame_count = 0

            self.status_update.emit(
                config.sequence[0],
                "--",
                0.0,
                "RUNNING",
            )

            while self.running:

                frame = self.camera.read()

                if frame is None:

                    self.error.emit(
                        "Failed to read camera frame."
                    )

                    break

                self.frame_count += 1

                self.frame_ready.emit(
                    frame.copy()
                )

                if (
                    self.frame_count
                    % self.process_every
                    != 0
                ):
                    continue

                perception_result = (
                    self.perception.process(
                        frame
                    )
                )

                # -----------------------------
                # HAR unavailable
                # -----------------------------

                if self.har_runtime is None:

                    self.status_update.emit(
                        config.sequence[0],
                        "WAITING_FOR_HAR_MODEL",
                        0.0,
                        "PERCEPTION OK",
                    )

                    continue

                # -----------------------------
                # HAR
                # -----------------------------

                if self.voice_runtime is not None:

                    result = (
                        self.voice_runtime.process(
                            perception_result
                            .interaction_features
                        )
                    )

                else:

                    result = (
                        self.har_runtime.process(
                            perception_result
                            .interaction_features
                        )
                    )

                prediction = result.get(
                    "prediction",
                    {}
                )

                validation = result.get(
                    "validation"
                )

                action = prediction.get(
                    "action",
                    "--"
                )

                confidence = prediction.get(
                    "confidence",
                    0.0
                )

                if self.voice_runtime is not None:

                    expected = (
                        self.voice_runtime
                        .har_runtime
                        .runtime
                        .expected_event
                    )

                else:

                    expected = (
                        self.har_runtime
                        .runtime
                        .expected_event
                    )

                status = "PROCESSING"

                if validation is not None:

                    status = (
                        validation.status.value
                    )

                    self.event_detected.emit(
                        validation.message
                    )

                self.status_update.emit(
                    expected,
                    action,
                    confidence,
                    status,
                )

        except Exception as error:

            self.error.emit(
                str(error)
            )

        finally:

            if self.camera is not None:
                self.camera.release()

            if self.logger is not None:
                self.logger.close()

            self.running = False

    def stop(self):

        self.running = False

        if self.isRunning():
            self.wait()