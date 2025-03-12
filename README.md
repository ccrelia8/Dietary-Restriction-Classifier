# Dietary-Restriction-Classifier
Experiments with binary classifiers for common dietary restrictions in recipes

This project focuses on classifying recipes based on common dietary restrictions using machine learning techniques. The motivation for this project comes from the need for food safety and the hope that individuals with specific dietary needs can determine whether a recipe meets those needs.

The classification task is performed using five binary classifiers for the following dietary restrictions:
Nut-free
Meat-free
Dairy-free
Gluten-free
Vegan

The dataset used for this project is the Recipes Dataset by The Devastator from Kaggle, which consists of recipes from allrecipes.com. The dataset contains 1,090 instances and 14 columns, but only the recipe_name and ingredients columns are used for feature extraction in this project.
The dataset is split as follows:
80% training set
10% development set
10% test set

The classification task uses three different models:
Logistic Regression
Multinomial Naive Bayes
XGBoost

Ingredients are tokenized into unigrams. A binary presence feature representation is used (i.e., whether a word is present or not). Hyperparameter tuning was conducted for each classifier.

The classifiers were evaluated using F1-score across different dietary restrictions. The best performing model varied depending on the class.
Logistic Regression performed best overall with a macro F1-score of 97.03. XGBoost had the highest F1-score for individual classes, reaching up to 100% for gluten-free classification. Multinomial Naive Bayes performed the lowest overall, with a macro F1-score of 84.96.

For specific dietary classes:
Nut-free classification achieved up to 98% F1-score.
Meat-free classification reached 94%.
Dairy-free classification peaked at 99%.
Gluten-free classification achieved 100%.
Vegan classification reached 97%.

To run the project:
install the dependency: pip install pandas scikit-learn nltk numpy xgboost
Download the dataset and place it in the same directory as main.py.
Run the script using: python main.py

The script will output the classification results and F1-scores for each dietary restriction.

Potential improvements for this project include:
Expanding the dataset with more recipes.
Using additional text preprocessing techniques.
Exploring deep learning approaches for classification.
Removing explicit dietary labels and predicting restrictions based on ingredient patterns.


Author

Chloe Crelia

Acknowledgments

Dataset: Recipes Dataset from Kaggle

Machine Learning Libraries: Scikit-learn, XGBoost

Natural Language Processing: NLTK

This project was developed as part of COSI 114a coursework.
