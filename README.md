# Student Performance Prediction with Decision Trees

This project predicts student performance using decision tree classification.

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/student_performance_prediction.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd student_performance_prediction
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

- **Run the `decision_trees.py` script:**

    ```bash
    python decision_trees.py
    ```

    This script performs the following tasks:

    - Reads and preprocesses the dataset.
    - Trains a decision tree classifier.
    - Visualizes the decision tree using Graphviz.
    - Evaluates the classifier's performance on a test set.
    - Performs cross-validation to assess model accuracy.
    - Tries different values of `max_depth` and visualizes the results.
