# type: ignore
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig


def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path="s3://emotion-sentiment-analysis/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard"
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role="arn:aws:iam::672919669633:role/mesa-analysis-execution-role",
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.m5.xlarge",
        hyperparameters={
            "batch-size": 32,
            "epochs": 25
        },
        tensorboard_config=tensorboard_config
    )

    # Start training
    estimator.fit({
        "training": "s3://emotion-sentiment-analysis/dataset/train",
        "validation": "s3://emotion-sentiment-analysis/dataset/dev",
        "test": "s3://emotion-sentiment-analysis/dataset/test"
    })


if __name__ == "__main__":
    start_training()
