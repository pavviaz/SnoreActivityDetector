import argparse

from data_utils.extract_features_from_dataset import FeatureExtractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-path", type=str, help="Path to conf file", required=True
    )

    args = parser.parse_args()

    extractor = FeatureExtractor(args.config_path)
    extractor.extract_features()