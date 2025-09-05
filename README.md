# NeuralNetworkFromScratch

This project implements a neural network from scratch, without using advanced frameworks like TensorFlow or PyTorch. The goal is to understand and demonstrate the fundamentals of neural networks, their training algorithms, and how they work internally.

## Features

- Pure Python implementation (no external dependencies for the model core)
- Manual forward and backward propagation
- Configurable activation functions (sigmoid, ReLU, tanh)
- Training via gradient descent
- Example training on simple problems (classification and regression)
- Visualization of training metrics
- **Webcam digit recognition demo**: Test your neural network in real time using your computer's camera

## Project Structure

```
NeuralNetworkFromScratch/
│
├── neural_network.py        # Main implementation of the neural network
├── layers.py                # Layers and activation functions
├── utils.py                 # Helper functions (normalization, metrics, etc.)
├── webcam_mnist_demo.py     # Webcam demo for digit recognition
├── demo.ipynb               # Interactive example in Jupyter Notebook
├── README.md                # This file
└── requirements.txt         # (Optional) Minimal dependencies for notebooks and visualization
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Gorchon/NeuralNetworkFromScratch.git
   cd NeuralNetworkFromScratch
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install minimal dependencies (for notebooks, visualization, and webcam demo):
   ```bash
   pip install -r requirements.txt
   ```
   *For the webcam demo, you will need `opencv-python` (cv2) and `numpy`.*

## Usage

You can run the included example in `demo.ipynb` to see how the neural network is trained and evaluated on a simple dataset.

Or use the main module in your own Python script:

```python
from neural_network import NeuralNetwork

# Define architecture: [n_inputs, n_hidden, n_output]
nn = NeuralNetwork([2, 4, 1], activation="sigmoid")

# Train
nn.train(X_train, y_train, epochs=1000, lr=0.1)

# Predict
preds = nn.predict(X_test)
```

### Webcam Digit Recognition Demo

You can use your computer's camera to draw digits in the air or on paper, and the neural network will try to recognize them in real time.

To run the webcam demo:
```bash
python webcam_mnist_demo.py
```

**Controls:**
- `q`: Quit the demo
- `i`: Toggle manual color inversion
- `a`: Toggle auto-invert (useful if the background is too dark or too bright)
- `c`: Switch between available cameras (if you have more than one)

**Notes:**
- The script will try to automatically detect and open your webcam. If it fails, check your permissions (especially on macOS: Privacy → Camera).
- Make sure you have trained your model and saved weights as `weights_mlp.npz` before running the demo, or use the provided example weights if available.

## Example Application

Training a network to solve the XOR problem:

```python
import numpy as np
from neural_network import NeuralNetwork

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 2, 1], activation="tanh")
nn.train(X, y, epochs=5000, lr=0.1)

print(nn.predict(X))
```

## Contributions

Contributions are welcome! You can open issues or pull requests to suggest improvements, new features, or bug fixes.

## License

This project is licensed under the MIT License.

---

Developed by [Gorchon](https://github.com/Gorchon)

---

**For more details about the webcam functionality, you can check the [`webcam_mnist_demo.py`](https://github.com/Gorchon/NeuralNetworkFromScratch/blob/main/webcam_mnist_demo.py) file.**  
*Note: The code search results may be incomplete. [See more code related to the camera here.](https://github.com/Gorchon/NeuralNetworkFromScratch/search?q=camera+OR+webcam+OR+cv2.VideoCapture+OR+test_camera)*
