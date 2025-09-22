Breast Cancer Classification with K-Nearest Neighbors

This project implements a K-Nearest Neighbors (KNN) machine learning model to classify breast cancer as malignant or benign using the Breast Cancer Wisconsin dataset from Scikit-learn. The model is trained on features such as mean radius, texture, and perimeter to predict tumor classification, and it includes an interactive feature allowing users to input custom feature values for real-time predictions.
Features

Dataset: Utilizes the Breast Cancer Wisconsin dataset, containing 30 features and binary target labels (malignant or benign).
Model: Employs the KNN algorithm with n_neighbors=3 for classification.
Data Splitting: Splits data into 80% training and 20% testing sets for robust evaluation.
Accuracy Reporting: Outputs the model's accuracy on the test dataset.
Interactive Prediction: Allows users to input custom feature values to predict tumor classification.
Error Handling: Validates user input to ensure correct format and number of features.

Prerequisites

Python 3.8 or higher
Required libraries: scikit-learn, numpy, pandas

Installation

Clone the repository:git clone https://github.com/WIlfredThePELE/Breast-Cancer-Classification.git
cd Breast-Cancer-Classification


Create a virtual environment (optional but recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt



Usage

Run the Script:
python breast_cancer_knn.py

The script will:

Load the Breast Cancer Wisconsin dataset.
Train a KNN model with 3 neighbors.
Display the model's accuracy on the test set.
Prompt for user input to predict tumor classification based on custom feature values.


Interactive Prediction:

When prompted, enter exactly 30 feature values (corresponding to the dataset's features) separated by commas.
Example input for features like mean radius, mean texture, etc.:Enter values: 14.0, 20.0, 90.0, ..., 0.1


The script validates the input and outputs the predicted class (malignant or benign).


Example Output:
A Machine Learning Model using K-Nearest_Neighbours For Breast Cancer Classification
Feature names:
['mean radius' 'mean texture' ... 'worst fractal dimension']
Target names:
['malignant' 'benign']
Model accuracy on test data: 0.9474

Enter the following feature values separated by commas.
You must enter exactly 30 values in the following order:
mean radius, mean texture, ...
Enter values: 14.0, 20.0, 90.0, ..., 0.1
The model predicts: benign



Dataset
The project uses the Breast Cancer Wisconsin dataset from Scikit-learn, which includes:

Features (30): Numerical measurements like mean radius, mean texture, worst area, etc.
Target: Binary labels (malignant = 0, benign = 1).
Size: 569 samples.

The dataset is automatically loaded via sklearn.datasets.load_breast_cancer().
Code Example
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load and split data
data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Train KNN model
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)
print("Model accuracy on test data:", clf.score(x_test, y_test))

# Predict with user input
feature_values = [float(val) for val in input("Enter values: ").split(",")]
prediction = clf.predict([feature_values])
print("The model predicts:", data.target_names[prediction[0]])

Evaluation
The model reports accuracy on the test set, typically around 90-95% depending on the random split. The KNN algorithm classifies tumors based on the proximity of feature values to training data points, using 3 nearest neighbors.
Repository Structure
your-repo/
│
├── breast_cancer_knn.py       # Main script for breast cancer classification
├── requirements.txt           # Project dependencies
├── README.md                 # Project documentation
└── LICENSE                   # License file

Requirements
The required Python libraries are listed in requirements.txt:
scikit-learn==1.5.2
numpy==1.26.4
pandas==2.2.3

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.

See CONTRIBUTING.md for more details.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact
For questions or suggestions, open an issue or contact [your email or GitHub handle].
