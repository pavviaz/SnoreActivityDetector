import argparse

from ml_pipeline.model_manager import ModelManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name", 
        type=str, 
        help="Name of the model to be saved", 
        required=True
    )

    args = parser.parse_args()

    manager = ModelManager(model_name=args.model_name)
    manager.export_jit_model()
