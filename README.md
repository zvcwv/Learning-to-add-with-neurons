# 🧠 Neural Net From Scratch

A beginner-friendly neural network built with Keras and TensorFlow that learns to add two numbers together. This project was created as a hands-on introduction to core machine learning concepts — data preparation, scaling, model architecture, training, and visualization.

---

## 📌 What Does It Do?

The model takes two numbers as input (for example, `X1 = 3` and `X2 = 5`) and learns to predict their sum (`y = 8`) by training on 200 randomly generated examples. While the task itself is simple, the project demonstrates the full end-to-end pipeline you would use for any regression problem.

---

## 🏗️ Model Architecture

The network uses a simple two-layer Sequential architecture:

```
Input (2 neurons) → Hidden Layer (3 neurons, linear) → Output (1 neuron, linear)
```

In total, the model has **13 trainable parameters** — 9 in the hidden layer (2 weights + 1 bias per neuron) and 4 in the output layer (3 weights + 1 bias).

---

## 🔍 Key Concepts Covered

**Data scaling** is applied using `MinMaxScaler` from scikit-learn, which transforms all values into the `[0, 1]` range. This is essential because neural networks are sensitive to the scale of input data — without it, training becomes unstable and slow.

**Model compilation** uses `mse` (Mean Squared Error) as the loss function and `adam` as the optimizer, which is a standard choice for regression tasks.

**Training history** is recorded across 300 epochs with an 80/20 train-validation split, making it easy to visualize how the loss decreases over time.

---

## 📊 Visualizations

The project includes three plots to help understand what is happening inside the model.

The first is a **training curve** showing how the loss drops over epochs for both the training and validation sets — the faster it drops and the closer the two lines stay together, the better.

The second is a **network architecture diagram** drawn with `matplotlib`, showing the neurons as colored circles and the connections between them as lines, similar to the diagrams you see in textbooks.

The third is a **predictions vs. actual values** scatter plot. If the model learned well, the points should lie close to the diagonal red line, which represents perfect predictions.

---

## 🚀 Getting Started

First, install the required dependencies:

```bash
pip install tensorflow scikit-learn matplotlib numpy
```

Then open the notebook in VS Code (you will need the Jupyter extension installed) and run the cells one by one from top to bottom:

```bash
# Clone the repository
git clone https://github.com/zvcwv/Learning-to-add-with-neurons.git
cd neural-net-from-scratch

# Open in VS Code
code .
```

---

## 📁 Project Structure

```
neural-net-from-scratch/
│
├── neural_network.ipynb   # Main notebook with all code and visualizations
└── README.md              # You are here
```

---

## 🎓 About

This project was built as part of my journey into Software Engineering and AI. It is my first step toward understanding how neural networks work from the ground up — before moving on to more complex architectures and real-world datasets.

---

## 🛠️ Built With

- [TensorFlow / Keras](https://www.tensorflow.org/) — model building and training
- [scikit-learn](https://scikit-learn.org/) — data preprocessing
- [NumPy](https://numpy.org/) — numerical data generation
- [Matplotlib](https://matplotlib.org/) — all visualizations
