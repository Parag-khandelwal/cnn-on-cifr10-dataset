---
# Convolutional Neural Network (CNN) with CIFAR-10 Dataset

This repository contains Python code implementing a Convolutional Neural Network (CNN) using the CIFAR-10 dataset. The code includes data loading, preprocessing, model building, training, evaluation, and the option to load a pre-trained model.

## File Structure

- **CNNWithCIFR-10Dataset.ipynb**: Jupyter notebook containing the code for the CNN implementation with CIFAR-10 dataset.

- **cifar_10epochs.h5**: Pre-trained model saved after training for 10 epochs.

- **larger_CIFAR10_model.h5**: Pre-trained larger model saved after training.

## Requirements
- Python 3
- Numpy
- Matplotlib
- Keras
- Scikit-learn

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/Parag-khandelwal/cnn-on-cifr10-dataset.git
    cd cnn-on-cifr10-dataset
    ```

2. Run the Jupyter notebook:

    ```bash
    jupyter notebook CNNWithCIFR-10Dataset.ipynb
    ```

3. Optionally, load a pre-trained model:

    ```python
    from keras.models import load_model

    # Load the model
    model = load_model('cifar_10epochs.h5')  # or 'larger_CIFAR10_model.h5'
    ```


## Description

- **Data Loading:** CIFAR-10 dataset is loaded using Keras's `cifar10.load_data()` function.

- **Data Visualization:** Matplotlib is used to visualize a sample image from the dataset.

- **Normalization:** Image data is normalized to ensure values are between 0 and 1.

- **One-Hot Encoding:** Labels are one-hot encoded to prepare the data for classification.

- **Model Architecture:** A CNN model is built using Keras with Convolutional, Pooling, Flattening, Dense, and Output layers.

- **Model Compilation:** The model is compiled with categorical crossentropy loss and RMSprop optimizer.

- **Training:** The model is trained on the training data.

- **Evaluation:** Model performance is evaluated on the test data, showing loss and accuracy.

- **Loading a Pre-trained Model:** Optionally, you can load a pre-trained model for further evaluation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
