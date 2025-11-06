# Linear Regression from Scratch

This project demonstrates the implementation of **Linear Regression** — one of the most fundamental algorithms in machine learning — **from scratch using NumPy and Pandas**.  
It does not rely on high-level machine learning libraries like Scikit-learn, ensuring a complete understanding of how the algorithm works internally.

---

## 🚀 Features

- Implementation of **Simple and Multiple Linear Regression** using pure mathematical concepts  
- Uses **NumPy** for efficient matrix operations  
- Uses **Pandas** for data handling and preprocessing  
- Includes **data visualization** and **model evaluation metrics**  
- Compares results with Scikit-learn’s implementation (optional)  

---

## 🧠 Concepts Covered

1. Understanding the mathematical formulation of Linear Regression  
   - Hypothesis function  
   - Cost function (Mean Squared Error)  
   - Gradient Descent algorithm  
2. Feature scaling and data normalization  
3. Model training and prediction  
4. Performance evaluation using:  
   - Mean Squared Error (MSE)  
   - R² Score  

---

## 🧩 Technologies Used

| Library | Purpose |
|----------|----------|
| **NumPy** | Matrix and vector computations |
| **Pandas** | Data manipulation and cleaning |
| **Matplotlib / Seaborn** | Visualization (optional) |
| **Jupyter Notebook** | Interactive coding environment |

---

## 📂 Project Structure

```
Linear_Regression_From_Scratch/
│
├── linear_train_2.ipynb        # Main notebook containing the full implementation
├── dataset.csv (optional)      # Input dataset used for training/testing
├── README.md                   # Project documentation
└── requirements.txt (optional) # Dependencies
```

---

## ⚙️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Sumitpalekar/Linear-Regression-From-Scratch.git
   cd Linear-Regression-From-Scratch
   ```

2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib
   ```

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook linear_train_2.ipynb
   ```

4. Run all cells to see the full workflow — from data preprocessing to model evaluation.

---

## 📊 Example Outputs

- **Loss function convergence graph** (Cost vs Iterations)
- **Predicted vs Actual plot**
- **Performance metrics** (MSE, R²)

---

## 🧮 Mathematical Background

The hypothesis for linear regression is:

\[
h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n
\]

The cost function (Mean Squared Error) is:

\[
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2
\]

The parameters are optimized using **Gradient Descent**:

\[
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
\]

---

## 📈 Evaluation Metrics

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

---

## 🔮 Future Improvements

- Implement **Polynomial Regression**
- Add **Regularization (Ridge/Lasso)**
- Build a small **CLI or Web Interface** for predictions

---

## 👨‍💻 Author

**Sumit Palekar**  
Student at IIT (ISM) Dhanbad  
Passionate about Data Science and Machine Learning  
