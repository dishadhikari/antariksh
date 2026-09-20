from src.camera.camera import Camera
from src.perception.pipeline import PerceptionPipeline
from src.har.runtime_pipeline import HARRuntimePipeline
from src.protocol.config import ExperimentConfig
from src.logging.logger import ExperimentLogger
from src.voice.tts import OfflineTTS
from src.voice.alerts import ExperimentVoiceAlerts
from src.protocol.voice_runtime import VoiceRuntimePipeline


def main():

    print("Loading experiment configuration...")

    config = ExperimentConfig(
        "experiments/box_experiment/experiment.yaml"
    )

    print(
        f"Experiment: {config.experiment_name}"
    )

    camera = Camera(
        source=0,
        width=640,
        height=480,
        fps=15,
    )

    perception = PerceptionPipeline(
        detector_model="models/detection/box_detector/weights/best.pt",
        confidence=0.5,
        device="cpu",
    )

    logger = ExperimentLogger(
        "data/logs/experiment.db"
    )

    har_runtime = HARRuntimePipeline(
        model_path="models/har/best_tcn.pt",
        label_map={
            index: action
            for index, action in enumerate(
                config.sequence
            )
        },
        experiment_id=config.experiment_id,
        sequence=config.sequence,
        logger=logger,
        sequence_length=30,
        confidence_threshold=config.confidence_threshold,
        smoothing_window=config.temporal_window,
        minimum_votes=config.minimum_votes,
        device="cpu",
    )

    tts = OfflineTTS(
        model_path="models/voice/piper.onnx"
    )

    alerts = ExperimentVoiceAlerts(tts)

    runtime = VoiceRuntimePipeline(
        har_runtime=har_runtime,
        alerts=alerts,
    )

    camera.open()

    print("Runtime started.")
    print("Press Q to stop.")

    frame_count = 0

    try:

        while True:

            frame = camera.read()

            if frame is None:
                print("Camera frame unavailable.")
                break

            frame_count += 1

            # CPU optimization.
            if frame_count % 3 != 0:
                continue

            perception_result = perception.process(
                frame
            )

            result = runtime.process(
                perception_result.interaction_features
            )

            if result.get("event") is not None:

                print(
                    f"Event: {result['event']}"
                )

                validation = result.get(
                    "validation"
                )

                if validation:
                    print(
                        f"Status: "
                        f"{validation.status.value}"
                    )

                    print(
                        f"Expected: "
                        f"{validation.expected}"
                    )

            if result.get("validation") is not None:

                if result["validation"].status.value == "COMPLETED":
                    print("Experiment completed.")
                    break

    except KeyboardInterrupt:

        print("\nRuntime interrupted.")

    finally:

        camera.release()
        logger.close()

        print("Runtime stopped.")


if __name__ == "__main__":
    main()