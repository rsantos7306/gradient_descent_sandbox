# Gradient Descent Polynomial Regression Visualizer

An interactive Python application to visualize **polynomial regression** using **batch**, **mini-batch**, and **stochastic gradient descent**. Adjust parameters to see how the slope, intercept, and error evolve.

---

## Features

- Interactive sliders to control:
  - The generating function and noise of the data
  - Learning rate
  - Number of epochs
  - Number of points
  - Initial parameters and degree (highest non-zero parameter)
  - Mini-batch size
- Choose between **stochastic**, **mini-batch**, or **batch gradient descent**
- Visualization of:
  - Original data points
  - Polynomial of best fit
  - Mean squared error over epochs

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/rsantos7306/gradient_descent_sandbox
cd gradient_descent_sandbox

```

2. **Install Dependencies**

```bash
pip install -r requirements.txt

```
- Requires Python 3.8+

---

## Usage

Run the main script while in the __gradient_descent_sandbox__ directory:

```bash
python main.py
```
---
## Structure

```bash
gradient-descent-visualizer/
├── gradient_descent/
│   ├── __init__.py
│   ├── algorithms.py      # Gradient descent implementations
│   └── utils.py           # Error calculation
├── main.py                # Interactive elements (sliders, buttons, plots)
├── requirements.txt       # Project dependencies
├── README.md
├── LICENSE
├── .gitattributes
└── .gitignore

```
