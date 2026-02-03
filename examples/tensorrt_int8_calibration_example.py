
import glob
import os
from typing import Iterator

import torch
from PIL import Image
from torchvision.transforms import v2

import lightly_train
from lightly_train._export.tensorrt_calibrator import Int8EntropyCalibrator


def create_calibration_iterator(
    image_dir: str, image_size: tuple[int, int], mean: list[float], std: list[float]
) -> Iterator[torch.Tensor]:
    """
    Creates an iterator that yields preprocessed image batches for TensorRT calibration.
    """
    # Find images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    # Define transforms to match inference/validation pipeline
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(image_size, antialias=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            img_tensor = transforms(image)
            # Add batch dimension (1, C, H, W)
            yield img_tensor.unsqueeze(0)
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")


def main():
    # Configuration
    checkpoint_path = "path/to/your/model.pt"
    calibration_images_dir = "path/to/calibration/images"
    output_engine_path = "model_int8.engine"
    cache_file = "calibration.cache"

    # 1. Load the model
    print(f"Loading model from {checkpoint_path}...")
    model = lightly_train.load_model(checkpoint_path)
    
    # 2. Extract normalization and image size directly from the model
    mean = model.image_normalize["mean"]
    std = model.image_normalize["std"]
    
    # model.image_size is a tuple (height, width)
    image_size = model.image_size
    
    print(f"Model config: size={image_size}, mean={mean}, std={std}")

    # 3. Create the calibration iterator
    calibration_iterator = create_calibration_iterator(
        image_dir=calibration_images_dir,
        image_size=image_size,
        mean=mean,
        std=std,
    )

    # 4. Initialize the calibrator
    calibrator = Int8EntropyCalibrator(
        calibration_data=calibration_iterator, 
        cache_file=cache_file, 
        batch_size=8
    )

    # 5. Export to TensorRT with INT8 precision
    print("Starting TensorRT INT8 export (this may take a while)...")
    model.export_tensorrt(
        out=output_engine_path,
        precision="int8",
        int8_calibrator=calibrator,
        verbose=True,
        min_batchsize=1,
        opt_batchsize=8,
        max_batchsize=16
    )
    print(f"Successfully exported INT8 engine to {output_engine_path}")


if __name__ == "__main__":
    main()
