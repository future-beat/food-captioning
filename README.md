# Food-101 Image Classification

A deep learning project that classifies food images into 101 categories using transfer learning with the Xception architecture. Trained on the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) from ETH Zurich, the model achieves **80.9% accuracy** on the test set.

## Overview

The model uses [Xception](https://keras.io/api/applications/xception/) pretrained on ImageNet as a feature extractor, with a custom classification head trained on 75,750 food images spanning 101 categories -- from apple pie to waffles.

### Model Architecture

| Layer | Details |
|---|---|
| Base | Xception (ImageNet weights, top removed) |
| Pooling | GlobalAveragePooling2D |
| Dense | 256 units, sigmoid |
| Dense | 128 units, sigmoid |
| Dropout | 0.2 |
| Output | 101 units, sigmoid |

### Training

- **Optimizer:** Adam
- **Loss:** Categorical crossentropy
- **Stage 1:** 60 epochs, learning rate 0.001
- **Stage 2:** 60 epochs, learning rate 0.0001 (fine-tuning)
- **Data augmentation:** Shear (0.2), zoom (0.2), horizontal flip

### Results

| Metric | Value |
|---|---|
| Test accuracy | 80.91% |
| Test loss | 0.7338 |

## Food Categories

The model recognizes 101 food types including: apple pie, baklava, bibimbap, bruschetta, caesar salad, cheesecake, chicken curry, chocolate cake, creme brulee, dumplings, falafel, filet mignon, fish and chips, french fries, fried rice, guacamole, hamburger, ice cream, lasagna, macarons, nachos, omelette, pad thai, paella, pancakes, pho, pizza, ramen, ravioli, risotto, sashimi, steak, sushi, tacos, tiramisu, waffles, and more.

The full list is available in [`food-101/data/classes.txt`](food-101/data/classes.txt).

## Prerequisites

- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd food-captioning
```

### 2. Install dependencies

```bash
pip install tensorflow numpy pandas matplotlib
```

### 3. Download the dataset

```bash
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xf food-101.tar.gz
```

This extracts ~5 GB of images into the `food-101/images/` directory.

### 4. Run the notebook

```bash
jupyter notebook food_captioning.ipynb
```

The notebook walks through data preparation, model training, evaluation, and inference on custom images.

## Project Structure

```
food-captioning/
├── food_captioning.ipynb   # Main notebook (data prep, training, evaluation)
├── food101.keras           # Trained model weights (Git LFS)
├── Project Report A.pdf    # Project report
└── food-101/
    ├── data/
    │   ├── classes.txt     # 101 class names
    │   ├── labels.txt      # Human-readable labels
    │   ├── train.txt       # 75,750 training samples
    │   ├── train.json      # Training metadata
    │   ├── test.txt        # 25,250 test samples
    │   └── test.json       # Test metadata
    ├── images/             # Dataset images (not tracked in git)
    ├── README.txt
    └── license_agreement.txt
```

## Using the Trained Model

To classify a food image using the pretrained model:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model("food101.keras")

img = load_img("your_food_image.jpg", target_size=(256, 256))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
```

Load the class names from `food-101/data/classes.txt` to map prediction indices to food labels.

## Dataset License

The Food-101 dataset images are from [Foodspotting](http://www.foodspotting.com/) and are not the property of ETH Zurich. Use beyond scientific fair use must be negotiated with the respective image owners. See [`food-101/license_agreement.txt`](food-101/license_agreement.txt) for details.
