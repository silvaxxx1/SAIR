# 📈 Regression Module – SAIR ML/DL Course

Welcome to the **Regression Module** of the **SAIR – Sudanese Artificial Intelligence Road** program.

This is your **first hands-on machine learning course** — where you’ll build real models, deploy interactive UIs, and work with your **own datasets**.

---

## 🧭 Module Overview

This folder contains everything you need to **learn, build, and showcase** your first end-to-end machine learning project:

```
Regression/
├── Lecture_1.ipynb
├── Lecture_2.ipynb
├── Lecture_3.ipynb
│
├── Resources/
│   ├── optional_read_1.pdf
│   └── optional_read_2.pdf
│
├── app.py                    # Streamlit UI (main design)
├── app_2.py                  # Alternative UI design
├── utils.py                  # Helper functions
├── utils2.py
│
├── assets/                   # Images & additional assets
├── experiments/              # MLflow experiment tracking
├── models/                   # Saved trained models
├── california_housing_model.pkl
├── california_housing_model_metadata.json
├── poly_model.pkl
│
├── 'Regression Capstone Projects'/   # 👈 Student projects live here!
│
└── README.md                 # You are here
```

---

## 🧠 What You’ll Learn

| Lecture       | Content                        | Key Skills                                                    |
| ------------- | ------------------------------ | ------------------------------------------------------------- |
| **Lecture 1** | Linear Regression Fundamentals | Linear models, gradient descent, basic metrics                |
| **Lecture 2** | Advanced Regression            | Feature scaling, polynomial regression, regularization        |
| **Lecture 3** | Model Evaluation & Deployment  | MLflow, cross-validation, hyperparameter tuning, Streamlit UI |

* **`Resources`** → Extra readings to deepen your understanding
* **`app.py`** & **`app_2.py`** → Reference UIs for deployment
* **`utils.py`** & **`utils2.py`** → Reusable preprocessing & feature engineering code

---

## 🧪 Experiments & Models

* All experiments are tracked with **MLflow** for reproducibility.
* Trained models are stored in the `models/` directory.
* Experiment logs, hyperparameters, and metrics are saved under `experiments/`.

---

## 🏁 Student Capstone Projects

All students must complete a **Regression Capstone Project** as part of this module.

Your capstone must include:

* ✅ A **dataset of your choice** (public or self-collected)
* ✅ A **Jupyter notebook** with all preprocessing, model training, and evaluation
* ✅ A **Streamlit app (`app.py`)** for an interactive UI
* ✅ Any additional `utils.py` or assets you used
* ✅ A short `README.md` explaining your project

### 📂 Required Folder Structure

```
Regression/Regression Capstone Projects/
└── YourProjectName/
    ├── notebook.ipynb
    ├── app.py
    ├── utils.py           (if needed)
    ├── data/              (your dataset)
    ├── models/            (your trained model)
    ├── experiments/       (MLflow logs if used)
    └── README.md          (short project description)
```

👉 **Name your folder clearly** using your project or dataset name.
👉 Do **not** upload massive datasets directly; link to them or keep them small.

---

## 📤 How to Upload Your Project to GitHub

### 1️⃣ Fork the SAIR Repository

👉 [https://github.com/silvaxxx1/SAIR](https://github.com/silvaxxx1/SAIR)

Click on **“Fork”** (top right corner).

### 2️⃣ Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/SAIR.git
cd SAIR/Regression/Regression\ Capstone\ Projects
```

### 3️⃣ Add Your Project Folder

Place your `YourProjectName/` inside `Regression Capstone Projects`.

### 4️⃣ Commit and Push

```bash
git add .
git commit -m "Add My Regression Capstone Project"
git push origin main
```

### 5️⃣ Submit a Pull Request (PR)

* Go to your fork on GitHub
* Click **“Compare & Pull Request”**
* Add a clear title & description
* Submit ✅

Your project will be reviewed and merged into the main SAIR repo.

---

## 🖥️ Running Your Streamlit App

To test your UI locally before submitting:

```bash
uv run streamlit run app.py 
# or
streamlit run app.py
```

If inside your project folder:

```bash
cd Regression/Regression\ Capstone\ Projects/YourProjectName
uv run streamlit run app.py
# or
streamlit run app.py
```

Your app will launch at 👉 [http://localhost:8501](http://localhost:8501).

---

## 🏆 Tips for a Great Capstone

* Use **clean and well-documented code** 🧼
* Explain your **data source** and **problem statement** clearly
* Add **visualizations** (matplotlib, seaborn, plotly) 📊
* Log experiments with MLflow for credibility
* Make your Streamlit UI **simple and interactive** 🖱️
* Include a `README.md` with:

  * 🔸 Project title
  * 📊 Dataset description
  * 🧠 Model used
  * 🚀 How to run your app
  * 📌 Results and observations

---

## 🧑‍🏫 Instructor’s Note

This capstone project is your **first real ML milestone**.

By completing it, you prove that you can:

* Build, train & evaluate a regression model
* Work with real datasets
* Deploy a simple web UI
* Share your work like a real ML engineer 🚀

✨ You MUST send video of your project in the SAIR Telegram group for demo and feedback.
---

## 📜 License

This module is part of **SAIR – Sudanese Artificial Intelligence Research**
Licensed under **MIT License**. You are free to use, modify, and share your work.

---

## 🤝 Join the SAIR Community

* ⭐ **Star** the repo to support the initiative
* 📢 Share your project with others
* 🧑‍💻 Mentor new contributors
* 🧠 Keep learning, keep building

📲 **Telegram Community:** [Join Here](https://t.me/+jPPlO6ZFDbtlYzU0)

> *“The best way to learn AI is to build with it.”*
> — SAIR Community

---

✅ **Now it’s your turn:**
👉 Open `Regression Capstone Projects`
👉 Create your project folder
👉 Start coding. Build something great.

