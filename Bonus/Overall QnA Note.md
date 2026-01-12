🔹 Linear Regression

What is Linear Regression?
It is a supervised learning algorithm used to predict continuous values.
It fits the best straight line that minimizes prediction error.

What does the line represent?
The line shows the relationship between input and output.
It passes through data in a way that overall error is smallest.

🔹 Gradient Descent

What is Gradient Descent?
It is an optimization algorithm that reduces error step by step.
It updates model parameters in the direction of minimum loss.

Why do we use Gradient Descent?
To find the best values of parameters that minimize error.
It helps the model learn gradually instead of guessing randomly.

🔹 Logistic Regression

What is Logistic Regression?
It is a classification algorithm that predicts probabilities.
The output is between 0 and 1, representing class likelihood.

Why do we use Sigmoid?
Sigmoid converts any number into a probability between 0 and 1.
This helps in making yes/no decisions.

🔹 Decision Tree

What is a Decision Tree?
It is a tree-like model that makes decisions using if-else rules.
Each node splits data based on a condition.

What is Entropy?
Entropy measures the disorder or uncertainty in data.
Lower entropy means purer data.

What is Information Gain?
It measures how much entropy is reduced after a split.
Higher information gain means better splitting.

🔹 Random Forest

What is Random Forest?
It is an ensemble model made of many decision trees.
Final prediction is based on majority voting or averaging.

Why is Random Forest better?
It reduces overfitting and improves accuracy.
Multiple trees make predictions more stable.

🔹 Support Vector Machine (SVM)

What is SVM?
SVM finds the best boundary that separates classes.
It maximizes the margin between data points.

What is the Kernel Trick?
It transforms data into higher dimensions.
This helps separate data that is not linearly separable.

🔹 Naive Bayes

What is Naive Bayes?
It is a probabilistic classifier based on Bayes’ theorem.
It predicts class using probability calculations.

Why is it called Naive?
It assumes features are independent.
This assumption is simple but often effective.

🔹 K-Nearest Neighbors (KNN)

What is KNN?
It predicts based on the nearest data points.
The majority class among neighbors is selected.

Why is KNN called lazy?
It does not learn during training.
Computation happens only during prediction.

🔹 K-Means

What is K-Means?
It is an unsupervised clustering algorithm.
It groups data into K clusters based on similarity.

What is a Centroid?
It is the center of a cluster.
Centroids are updated iteratively.

🔹 Neural Networks

What is a Neural Network?
It is a model inspired by the human brain.
It consists of layers of connected neurons.

Why use Activation Functions?
They introduce non-linearity into the model.
Without them, neural networks behave like linear models.

🔹 Evaluation Metrics

What is R² Score?
It shows how well the model explains variance in data.
Higher value means better fit.

What is RMSE?
It measures average prediction error magnitude.
Lower RMSE means better performance.

What is Accuracy?
It is the ratio of correct predictions to total predictions.
Useful for balanced datasets.

🔹 Classification Metrics

What is Precision?
It measures how many predicted positives are correct.
Important when false positives are costly.

What is Recall?
It measures how many actual positives are detected.
Important when missing positives is dangerous.

What is F1 Score?
It balances precision and recall.
Useful for imbalanced datasets.

🔹 Confusion Matrix

What is a Confusion Matrix?
It shows TP, FP, TN, and FN values.
Helps analyze model mistakes clearly.

🔹 ROC & AUC

What is ROC Curve?
It plots true positive rate vs false positive rate.
It shows model performance at different thresholds.

What is AUC?
Area under ROC curve.
Higher AUC means better classifier.

🔹 Cross-Validation

What is Cross-Validation?
It splits data into multiple parts for evaluation.
Ensures reliable model performance.

Why is Cross-Validation important?
It reduces overfitting risk.
Gives a better estimate of real-world performance.