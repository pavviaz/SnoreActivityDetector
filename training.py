import argparse

from ml_pipeline.model_manager import ModelManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-path", type=str, help="Path to conf file", required=True
    )

    args = parser.parse_args()

    trainer = ModelManager(args.config_path)
    trainer.train()
    trainer.export_jit_model()