import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import nltk
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('wordnet')


def load_data(file):
    return pd.read_csv(file, usecols=['recipe_name', 'ingredients'], encoding='utf-8')


data = load_data('recipes.csv')


def extract_features(ingredients):
    tokens = word_tokenize(ingredients.lower())
    features = {token: 1.0 for token in tokens}
    return features


data['features'] = data['ingredients'].apply(extract_features)


def is_nut_free(ingredients):
    nuts = ['nut', 'peanut', 'almond', 'walnut', 'cashew', 'pecan', 'hazelnut', 'macadamia', 'brazil nut', 'pistachio']
    return not any(nut in ingredients.lower() for nut in nuts)


data['nut_free'] = data['ingredients'].apply(is_nut_free)

nut_free_counts = data['nut_free'].value_counts()
print(nut_free_counts)


def is_meat_free(ingredients):
    meats = ['pork', 'beef', 'chicken', 'duck', 'sirloin', 'fish', 'salmon', 'cod', 'bison', 'cow', 'chicken', 'shrimp',
             'lobster', 'boar', 'tilapia', 'meat', 'turkey', 'ham', 'bacon']

    return not any(meat in ingredients.lower() for meat in meats)


data['meat_free'] = data['ingredients'].apply(is_meat_free)

meat_free_counts = data['meat_free'].value_counts()
print(meat_free_counts)


def is_dairy_free(ingredients):
    dairy_products = ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'whey', 'ghee', 'sour cream', 'ice cream']
    return not any(dairy in ingredients.lower() for dairy in dairy_products)


data['dairy_free'] = data['ingredients'].apply(is_dairy_free)

dairy_free_counts = data['dairy_free'].value_counts()
print(dairy_free_counts)


def is_gluten_free(ingredients):
    gluten_products = ['wheat', 'barley', 'rye', 'malt', 'brewer’s yeast', 'seitan', 'bread', 'pasta',
                       'cereal', 'cake', 'beer', 'breadcrumbs', 'flour']
    return not any(gluten in ingredients.lower() for gluten in gluten_products)


data['gluten_free'] = data['ingredients'].apply(is_gluten_free)

gluten_free_counts = data['gluten_free'].value_counts()
print(gluten_free_counts)


def is_vegan(ingredients):
    # check if the recipe is free from meat and dairy
    if not is_meat_free(ingredients) or not is_dairy_free(ingredients):
        return False

    additional_non_vegan = ['egg', 'honey', 'gelatin', 'albumin', 'casein', 'lactose', 'shellac', 'isinglass']

    return not any(ingredient in ingredients.lower() for ingredient in additional_non_vegan)


data['vegan'] = data['ingredients'].apply(is_vegan)

vegan_counts = data['vegan'].value_counts()
print(vegan_counts)


def split_dataset(data, train_size=0.8, test_size=0.1, random_state=None):
    dev_size = test_size / (1 - train_size)

    training_data, temp_test_data = train_test_split(data, train_size=train_size, random_state=random_state)

    devset_data, testing_data = train_test_split(temp_test_data, test_size=dev_size, random_state=random_state)

    return training_data, devset_data, testing_data


training_data, devset_data, testing_data = split_dataset(data, train_size=0.8, test_size=0.1, random_state=42)

# sizes of each dataset
total_size = len(data)
train_size = len(training_data)
dev_size = len(devset_data)
test_size = len(testing_data)

# vectorize
vectorizer = DictVectorizer()

X_train = vectorizer.fit_transform(training_data['features'])
X_dev = vectorizer.transform(devset_data['features'])
X_test = vectorizer.transform(testing_data['features'])

# Logistic Regression models

# NUTS

# extract
y_train = training_data['nut_free']
y_dev = devset_data['nut_free']
y_test = testing_data['nut_free']

#new base case
nut_lr_base = LogisticRegression(C=1.0)
nut_lr_base.fit(X_train, y_train)

dev_predictions = nut_lr_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)
dev_f1 = f1_score(y_dev, dev_predictions)
print("NUT ZONE -- LR")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]
# Initialize the best score and corresponding C value
best_f1_score = 0
alpha = None
for value in params:
    nut_lr_tuning = LogisticRegression(C=value)
    nut_lr_tuning.fit(X_train, y_train)

    dev_predictions = nut_lr_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev, dev_predictions)

    if current_f1_score >= best_f1_score: #changed to >=
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

nut_lr_model = LogisticRegression(C = alpha)
nut_lr_model.fit(X_train, y_train)

# dev set
dev_predictions = nut_lr_model.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)
dev_f1 = f1_score(y_dev, dev_predictions)
#print("NUT ZONE -- LR")
# print("Development Set Accuracy:", dev_accuracy)
# print("Development Set F1 Score:", dev_f1)

# test set
nut_test_predictions = nut_lr_model.predict(X_test)
test_accuracy = accuracy_score(y_test, nut_test_predictions)
test_f1_nut = f1_score(y_test, nut_test_predictions)

# print("Test Set Accuracy:", test_accuracy)
#print("Test Set F1 Score:", test_f1_nut)

# Output results
#print("NUT ZONE -- LR")
print(classification_report(y_test, nut_test_predictions))


# MEAT

# extract
y_train = training_data['meat_free']
y_dev = devset_data['meat_free']
y_test = testing_data['meat_free']

#new base case
meat_lr_base = LogisticRegression(C=1.0)
meat_lr_base.fit(X_train, y_train)

dev_predictions = meat_lr_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)
dev_f1 = f1_score(y_dev, dev_predictions)
print("MEAT FACTORY -- LR")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]
best_f1_score = 0
alpha = None
for value in params:
    meat_lr_tuning = LogisticRegression(C=value)
    meat_lr_tuning.fit(X_train, y_train)

    dev_predictions = meat_lr_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev, dev_predictions)

    if current_f1_score >= best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

meat_lr_model = LogisticRegression()
meat_lr_model.fit(X_train, y_train)

# dev set
dev_predictions = meat_lr_model.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)
dev_f1 = f1_score(y_dev, dev_predictions)
# print("MEAT FACTORY -- LR")
# print("Development Set Accuracy:", dev_accuracy)
# print("Development Set F1 Score:", dev_f1)

# test set
meat_test_predictions = meat_lr_model.predict(X_test)
test_accuracy = accuracy_score(y_test, meat_test_predictions)
test_f1_meat = f1_score(y_test, meat_test_predictions)
# print("Test Set Accuracy:", test_accuracy)
# print("Test Set F1 Score:", test_f1_meat)

# Output results
#print("MEAT FACTORY -- LR")
print(classification_report(y_test, meat_test_predictions))

# DAIRY

# extract
y_train = training_data['dairy_free']
y_dev = devset_data['dairy_free']
y_test = testing_data['dairy_free']

#new base case
dairy_lr_base = LogisticRegression(C=1.0)
dairy_lr_base.fit(X_train, y_train)

dev_predictions = dairy_lr_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)
dev_f1 = f1_score(y_dev, dev_predictions)
print("DAIRY FARM -- LR")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]
best_f1_score = 0
alpha = None
for value in params:
    dairy_lr_tuning = LogisticRegression(C=value)
    dairy_lr_tuning.fit(X_train, y_train)

    dev_predictions = dairy_lr_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev, dev_predictions)

    if current_f1_score >= best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

dairy_lr_model = LogisticRegression()
dairy_lr_model.fit(X_train, y_train)

# dev set
dev_predictions = dairy_lr_model.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)
dev_f1 = f1_score(y_dev, dev_predictions)
#print("DAIRY FARM -- LR")
#print("Development Set Accuracy:", dev_accuracy)
#print("Development Set F1 Score:", dev_f1)

# test set
dairy_test_predictions = dairy_lr_model.predict(X_test)
test_accuracy = accuracy_score(y_test, dairy_test_predictions)
test_f1_dairy = f1_score(y_test, dairy_test_predictions)
# print("Test Set Accuracy:", test_accuracy)
# print("Test Set F1 Score:", test_f1_dairy)

# Output results
#print("DAIRY FARM -- LR")
print(classification_report(y_test, dairy_test_predictions))

# GLUTEN

# extract
y_train = training_data['gluten_free']
y_dev = devset_data['gluten_free']
y_test = testing_data['gluten_free']

#new base case
gluten_lr_base = LogisticRegression(C=1.0)
gluten_lr_base.fit(X_train, y_train)

dev_predictions = gluten_lr_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)
dev_f1 = f1_score(y_dev, dev_predictions)
print("BAKERY -- LR")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]
best_f1_score = 0
alpha = None
for value in params:
    gluten_lr_tuning = LogisticRegression(C=value)
    gluten_lr_tuning.fit(X_train, y_train)

    dev_predictions = gluten_lr_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev, dev_predictions)

    if current_f1_score >= best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

gluten_lr_model = LogisticRegression()
gluten_lr_model.fit(X_train, y_train)

# dev set
dev_predictions = gluten_lr_model.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)
dev_f1 = f1_score(y_dev, dev_predictions)
# print("BAKERY -- LR")
# print("Development Set Accuracy:", dev_accuracy)
# print("Development Set F1 Score:", dev_f1)

# test set
gluten_test_predictions = gluten_lr_model.predict(X_test)
test_accuracy = accuracy_score(y_test, gluten_test_predictions)
test_f1_gluten = f1_score(y_test, gluten_test_predictions)
# print("Test Set Accuracy:", test_accuracy)
# print("Test Set F1 Score:", test_f1_gluten)

# Output results
#print("BAKERY -- LR")
print(classification_report(y_test, gluten_test_predictions))

# VEGAN

# extract
y_train = training_data['vegan']
y_dev = devset_data['vegan']
y_test = testing_data['vegan']

#new base case
vegan_lr_base = LogisticRegression(C=1.0)
vegan_lr_base.fit(X_train, y_train)

dev_predictions = vegan_lr_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)
dev_f1 = f1_score(y_dev, dev_predictions)
print("CRUELTY FREE -- LR")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]
best_f1_score = 0
alpha = None
for value in params:
    vegan_lr_tuning = LogisticRegression(C=value)
    vegan_lr_tuning.fit(X_train, y_train)

    dev_predictions = vegan_lr_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev, dev_predictions)

    if current_f1_score >= best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

vegan_lr_model = LogisticRegression()
vegan_lr_model.fit(X_train, y_train)

# dev set
dev_predictions = vegan_lr_model.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)
dev_f1 = f1_score(y_dev, dev_predictions)
# print("CRUELTY FREE -- LR")
# print("Development Set Accuracy:", dev_accuracy)
# print("Development Set F1 Score:", dev_f1)

# test set
vegan_test_predictions = vegan_lr_model.predict(X_test)
test_accuracy = accuracy_score(y_test, vegan_test_predictions)
test_f1_vegan = f1_score(y_test, vegan_test_predictions)
# print("Test Set Accuracy:", test_accuracy)
# print("Test Set F1 Score:", test_f1_vegan)

# Output results
#print("CRUELTY FREE -- LR")
print(classification_report(y_test, vegan_test_predictions))

# Combine the true labels for all classifiers
y_true_combined = np.vstack(
    (testing_data['nut_free'], testing_data['meat_free'], testing_data['dairy_free'], testing_data['gluten_free'], testing_data['vegan'])).T

# Combine the predictions for all classifiers
y_pred_combined = np.vstack(
    (nut_test_predictions, meat_test_predictions, dairy_test_predictions, gluten_test_predictions, vegan_test_predictions)).T

# F1 scores for each MNB classifier
lr_f1_scores = [
    test_f1_nut,  # Nut-free
    test_f1_meat,  # Meat-free
    test_f1_dairy,  # Dairy-free
    test_f1_gluten,  # Gluten-free
    test_f1_vegan  # Vegan
]

# Calculate the average Macro F1 score for MNB classifiers
combined_macro_f1_lr = sum(lr_f1_scores) / len(lr_f1_scores)
print("Combined Macro F1 Score for LR classifiers:", combined_macro_f1_lr)

# Multinomial Naive Bayes models

# NUTS

y_train_nut = training_data['nut_free']
y_dev_nut = devset_data['nut_free']
y_test_nut = testing_data['nut_free']

#new base case
nut_mnb_base = MultinomialNB(alpha=1.0)
nut_mnb_base.fit(X_train, y_train_nut)

dev_predictions = nut_mnb_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev_nut, dev_predictions)
dev_f1 = f1_score(y_dev_nut, dev_predictions)
print("NUT ZONE -- MNB")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]

best_f1_score = 0
alpha = None
for value in params:
    nut_mnb_tuning = MultinomialNB(alpha=value)
    nut_mnb_tuning.fit(X_train, y_train_nut)

    dev_predictions = nut_mnb_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev_nut, dev_predictions)

    if current_f1_score >= best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

nut_mnb_model = MultinomialNB(alpha=alpha)
nut_mnb_model.fit(X_train, y_train_nut)

# dev set
dev_predictions_nut = nut_mnb_model.predict(X_dev)
dev_accuracy_nut = accuracy_score(y_dev_nut, dev_predictions_nut)
dev_f1_nut = f1_score(y_dev_nut, dev_predictions_nut)
# print("NUT ZONE -- MNB")
# print("Development Set Accuracy:", dev_accuracy_nut)
# print("Development Set F1 Score:", dev_f1_nut)

# test set
test_predictions_nut = nut_mnb_model.predict(X_test)
test_accuracy_nut = accuracy_score(y_test_nut, test_predictions_nut)
test_f1_nut = f1_score(y_test_nut, test_predictions_nut)
# print("Test Set Accuracy:", test_accuracy_nut)
# print("Test Set F1 Score:", test_f1_nut)

# Output results
#print("NUT ZONE -- MNB")
print(classification_report(y_test_nut, test_predictions_nut))


# MEAT

y_train_meat = training_data['meat_free']
y_dev_meat = devset_data['meat_free']
y_test_meat = testing_data['meat_free']

#new base case
meat_mnb_base = MultinomialNB(alpha=1.0)
meat_mnb_base.fit(X_train, y_train_meat)

dev_predictions = meat_mnb_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev_meat, dev_predictions)
dev_f1 = f1_score(y_dev_meat, dev_predictions)
print("MEAT FACTORY -- MNB")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]

best_f1_score = 0
alpha = None
for value in params:
    meat_mnb_tuning = MultinomialNB(alpha=value)
    meat_mnb_tuning.fit(X_train, y_train_meat)

    dev_predictions = meat_mnb_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev_meat, dev_predictions)

    if current_f1_score >= best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

meat_mnb_model = MultinomialNB()
meat_mnb_model.fit(X_train, y_train_meat)

# dev set
dev_predictions_meat = meat_mnb_model.predict(X_dev)
dev_accuracy_meat = accuracy_score(y_dev_meat, dev_predictions_meat)
dev_f1_meat = f1_score(y_dev_meat, dev_predictions_meat)
# print("MEAT FACTORY -- MNB")
# print("Development Set Accuracy:", dev_accuracy_meat)
# print("Development Set F1 Score:", dev_f1_meat)

# test set
test_predictions_meat = meat_mnb_model.predict(X_test)
test_accuracy_meat = accuracy_score(y_test_meat, test_predictions_meat)
test_f1_meat = f1_score(y_test_meat, test_predictions_meat)
# print("Test Set Accuracy:", test_accuracy_meat)
# print("Test Set F1 Score:", test_f1_meat)

# Output results
#print("MEAT FACTORY -- MNB")
print(classification_report(y_test_meat, test_predictions_meat))


# DAIRY

y_train_dairy = training_data['dairy_free']
y_dev_dairy = devset_data['dairy_free']
y_test_dairy = testing_data['dairy_free']

#new base case
dairy_mnb_base = MultinomialNB(alpha=1.0)
dairy_mnb_base.fit(X_train, y_train_dairy)

dev_predictions = dairy_mnb_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev_dairy, dev_predictions)
dev_f1 = f1_score(y_dev_dairy, dev_predictions)
print("DAIRY FARM -- MNB")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]

best_f1_score = 0
alpha = None
for value in params:
    dairy_mnb_tuning = MultinomialNB(alpha=value)
    dairy_mnb_tuning.fit(X_train, y_train_dairy)

    dev_predictions = dairy_mnb_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev_dairy, dev_predictions)

    if current_f1_score >= best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

dairy_mnb_model = MultinomialNB()
dairy_mnb_model.fit(X_train, y_train_dairy)

# devset
dev_predictions_dairy = dairy_mnb_model.predict(X_dev)
dev_accuracy_dairy = accuracy_score(y_dev_dairy, dev_predictions_dairy)
dev_f1_dairy = f1_score(y_dev_dairy, dev_predictions_dairy)
# print("DAIRY FARM -- MNB")
# print("Development Set Accuracy:", dev_accuracy_dairy)
# print("Development Set F1 Score:", dev_f1_dairy)

# test set
test_predictions_dairy = dairy_mnb_model.predict(X_test)
test_accuracy_dairy = accuracy_score(y_test_dairy, test_predictions_dairy)
test_f1_dairy = f1_score(y_test_dairy, test_predictions_dairy)
# print("Test Set Accuracy:", test_accuracy_dairy)
# print("Test Set F1 Score:", test_f1_dairy)

# Output results
#print("DAIRY FARM -- MNB")
print(classification_report(y_test_dairy, test_predictions_dairy))


# GLUTEN

y_train_gluten = training_data['gluten_free']
y_dev_gluten = devset_data['gluten_free']
y_test_gluten = testing_data['gluten_free']

#new base case
gluten_mnb_base = MultinomialNB(alpha=1.0)
gluten_mnb_base.fit(X_train, y_train_gluten)

dev_predictions = gluten_mnb_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev_gluten, dev_predictions)
dev_f1 = f1_score(y_dev_gluten, dev_predictions)
print("BAKERY -- MNB")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]

best_f1_score = 0
alpha = None
for value in params:
    gluten_mnb_tuning = MultinomialNB(alpha=value)
    gluten_mnb_tuning.fit(X_train, y_train_gluten)

    dev_predictions = gluten_mnb_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev_gluten, dev_predictions)

    if current_f1_score >= best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

gluten_mnb_model = MultinomialNB()
gluten_mnb_model.fit(X_train, y_train_gluten)

# dev set
dev_predictions_gluten = gluten_mnb_model.predict(X_dev)
dev_accuracy_gluten = accuracy_score(y_dev_gluten, dev_predictions_gluten)
dev_f1_gluten = f1_score(y_dev_gluten, dev_predictions_gluten)
# print("BAKERY -- MNB")
# print("Development Set Accuracy:", dev_accuracy_gluten)
# print("Development Set F1 Score:", dev_f1_gluten)

# test set
test_predictions_gluten = gluten_mnb_model.predict(X_test)
test_accuracy_gluten = accuracy_score(y_test_gluten, test_predictions_gluten)
test_f1_gluten = f1_score(y_test_gluten, test_predictions_gluten)
# print("Test Set Accuracy:", test_accuracy_gluten)
# print("Test Set F1 Score:", test_f1_gluten)

# Output results
#print("BAKERY -- MNB")
print(classification_report(y_test_gluten, test_predictions_gluten))


# VEGAN

y_train_vegan = training_data['vegan']
y_dev_vegan = devset_data['vegan']
y_test_vegan = testing_data['vegan']

#new base case
vegan_mnb_base = MultinomialNB(alpha=1.0)
vegan_mnb_base.fit(X_train, y_train_vegan)

dev_predictions = vegan_mnb_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev_vegan, dev_predictions)
dev_f1 = f1_score(y_dev_vegan, dev_predictions)
print("CRUELTY FREE -- MNB")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]

best_f1_score = 0
alpha = None
for value in params:
    vegan_mnb_tuning = MultinomialNB(alpha=value)
    vegan_mnb_tuning.fit(X_train, y_train_vegan)

    dev_predictions = vegan_mnb_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev_vegan, dev_predictions)

    if current_f1_score >= best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

vegan_mnb_model = MultinomialNB()
vegan_mnb_model.fit(X_train, y_train_vegan)

# dev set
dev_predictions_vegan = vegan_mnb_model.predict(X_dev)
dev_accuracy_vegan = accuracy_score(y_dev_vegan, dev_predictions_vegan)
dev_f1_vegan = f1_score(y_dev_vegan, dev_predictions_vegan)
# print("CRUELTY FREE - MNB")
# print("Development Set Accuracy:", dev_accuracy_vegan)
# print("Development Set F1 Score:", dev_f1_vegan)

# test set
test_predictions_vegan = vegan_mnb_model.predict(X_test)
test_accuracy_vegan = accuracy_score(y_test_vegan, test_predictions_vegan)
test_f1_vegan = f1_score(y_test_vegan, test_predictions_vegan)
# print("Test Set Accuracy:", test_accuracy_vegan)
# print("Test Set F1 Score:", test_f1_vegan)

# Output results
print("CRUELTY FREE -- MNB")
print(classification_report(y_test_vegan, test_predictions_vegan))


# Combine the true labels for all classifiers
y_true_combined = np.vstack(
    (testing_data['nut_free'], testing_data['meat_free'], testing_data['dairy_free'], testing_data['gluten_free'], testing_data['vegan'])).T

# Combine the predictions for all classifiers
y_pred_combined = np.vstack(
    (nut_test_predictions, meat_test_predictions, dairy_test_predictions, gluten_test_predictions, vegan_test_predictions)).T

# F1 scores for each MNB classifier
mnb_f1_scores = [
    test_f1_nut,  # Nut-free
    test_f1_meat,  # Meat-free
    test_f1_dairy,  # Dairy-free
    test_f1_gluten,  # Gluten-free
    test_f1_vegan  # Vegan
]

# Calculate the average Macro F1 score for MNB classifiers
combined_macro_f1_mnb = sum(mnb_f1_scores) / len(mnb_f1_scores)
print("Combined Macro F1 Score for MNB classifiers:", combined_macro_f1_mnb)


# XGBoost Classifiers

# NUT
y_train_nut = training_data['nut_free']
y_dev_nut = devset_data['nut_free']
y_test_nut = testing_data['nut_free']

#new base case
nut_xgb_base = xgb.XGBClassifier(eta=1.0)
nut_xgb_base.fit(X_train, y_train_nut)

dev_predictions = nut_xgb_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev_nut, dev_predictions)
dev_f1 = f1_score(y_dev_nut, dev_predictions)
print("NUT ZONE -- XGB")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new


#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]

best_f1_score = 0
alpha = None
for value in params:
    nut_xgb_tuning = xgb.XGBClassifier(eval_metric='logloss', eta=value)
    nut_xgb_tuning.fit(X_train, y_train_nut)

    dev_predictions = nut_xgb_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev_nut, dev_predictions)

    if current_f1_score >= best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

nut_xgb_model = xgb.XGBClassifier(eval_metric='logloss', eta=alpha)
nut_xgb_model.fit(X_train, y_train_nut)

# dev set
dev_predictions_nut = nut_xgb_model.predict(X_dev)
dev_accuracy_nut = accuracy_score(y_dev_nut, dev_predictions_nut)
dev_f1_nut = f1_score(y_dev_nut, dev_predictions_nut)
# print("NUT ZONE -- XGB")
# print("Development Set Accuracy:", dev_accuracy_nut)
# print("Development Set F1 Score:", dev_f1_nut)

# test set
test_predictions_nut = nut_xgb_model.predict(X_test)
test_accuracy_nut = accuracy_score(y_test_nut, test_predictions_nut)
test_f1_nut = f1_score(y_test_nut, test_predictions_nut)
# print("Test Set Accuracy:", test_accuracy_nut)
# print("Test Set F1 Score:", test_f1_nut)

# Output results
#print("NUT ZONE -- XGB")
print(classification_report(y_test_nut, test_predictions_nut))


# MEAT

y_train_meat = training_data['meat_free']
y_dev_meat = devset_data['meat_free']
y_test_meat = testing_data['meat_free']

#new base case
meat_xgb_base = xgb.XGBClassifier(eta=1.0)
meat_xgb_base.fit(X_train, y_train_meat)

dev_predictions = meat_xgb_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev_meat, dev_predictions)
dev_f1 = f1_score(y_dev_meat, dev_predictions)
print("MEAT FACTORY -- XGB")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]

best_f1_score = 0
alpha = None
for value in params:
    meat_xgb_tuning = xgb.XGBClassifier(eval_metric='logloss', eta=value)
    meat_xgb_tuning.fit(X_train, y_train_meat)

    dev_predictions = meat_xgb_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev_meat, dev_predictions)

    if current_f1_score >= best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

meat_xgb_model = xgb.XGBClassifier(eval_metric='logloss')
meat_xgb_model.fit(X_train, y_train_meat)



# dev set
dev_predictions_meat = meat_xgb_model.predict(X_dev)
dev_accuracy_meat = accuracy_score(y_dev_meat, dev_predictions_meat)
dev_f1_meat = f1_score(y_dev_meat, dev_predictions_meat)
# print("MEAT FACTORY -- XGB")
# print("Development Set Accuracy:", dev_accuracy_meat)
# print("Development Set F1 Score:", dev_f1_meat)

# test set
test_predictions_meat = meat_xgb_model.predict(X_test)
test_accuracy_meat = accuracy_score(y_test_meat, test_predictions_meat)
test_f1_meat = f1_score(y_test_meat, test_predictions_meat)
# print("Test Set Accuracy:", test_accuracy_meat)
# print("Test Set F1 Score:", test_f1_meat)

# Output results
#print("MEAT FACTORY -- XGB")
print(classification_report(y_test_meat, test_predictions_meat))


# DAIRY
y_train_dairy = training_data['dairy_free']
y_dev_dairy = devset_data['dairy_free']
y_test_dairy = testing_data['dairy_free']

#new base case
dairy_xgb_base = xgb.XGBClassifier(eta=1.0)
dairy_xgb_base.fit(X_train, y_train_dairy)

dev_predictions = dairy_xgb_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev_dairy, dev_predictions)
dev_f1 = f1_score(y_dev_dairy, dev_predictions)
print("DAIRY FARM -- XGB")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]

best_f1_score = 0
alpha = None
for value in params:
    dairy_xgb_tuning = xgb.XGBClassifier(eval_metric='logloss', eta=value)
    dairy_xgb_tuning.fit(X_train, y_train_dairy)

    dev_predictions = dairy_xgb_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev_dairy, dev_predictions)

    if current_f1_score > best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

dairy_xgb_model = xgb.XGBClassifier(eval_metric='logloss')
dairy_xgb_model.fit(X_train, y_train_dairy)

# dev set
dev_predictions_dairy = dairy_xgb_model.predict(X_dev)
dev_accuracy_dairy = accuracy_score(y_dev_dairy, dev_predictions_dairy)
dev_f1_dairy = f1_score(y_dev_dairy, dev_predictions_dairy)
# print("DAIRY FARM -- XGB")
# print("Development Set Accuracy:", dev_accuracy_dairy)
# print("Development Set F1 Score:", dev_f1_dairy)

# test set
test_predictions_dairy = dairy_xgb_model.predict(X_test)
test_accuracy_dairy = accuracy_score(y_test_dairy, test_predictions_dairy)
test_f1_dairy = f1_score(y_test_dairy, test_predictions_dairy)
# print("Test Set Accuracy:", test_accuracy_dairy)
# print("Test Set F1 Score:", test_f1_dairy)

# Output results
#print("DAIRY FARM -- XGB")
print(classification_report(y_test_dairy, test_predictions_dairy))


# GLUTEN
y_train_gluten = training_data['gluten_free']
y_dev_gluten = devset_data['gluten_free']
y_test_gluten = testing_data['gluten_free']

#new base case
gluten_xgb_base = xgb.XGBClassifier(eta=1.0)
gluten_xgb_base.fit(X_train, y_train_gluten)

dev_predictions = gluten_xgb_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev_gluten, dev_predictions)
dev_f1 = f1_score(y_dev_gluten, dev_predictions)
print("BAKERY -- XGB")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]

best_f1_score = 0
alpha = None
for value in params:
    gluten_xgb_tuning = xgb.XGBClassifier(eval_metric='logloss', eta=value)
    gluten_xgb_tuning.fit(X_train, y_train_gluten)

    dev_predictions = gluten_xgb_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev_gluten, dev_predictions)

    if current_f1_score > best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

gluten_xgb_model = xgb.XGBClassifier(eval_metric='logloss', eta=alpha)
gluten_xgb_model.fit(X_train, y_train_gluten)

# dev set
dev_predictions_gluten = gluten_xgb_model.predict(X_dev)
dev_accuracy_gluten = accuracy_score(y_dev_gluten, dev_predictions_gluten)
dev_f1_gluten = f1_score(y_dev_gluten, dev_predictions_gluten)
# print("BAKERY -- XGB")
# print("Development Set Accuracy:", dev_accuracy_gluten)
# print("Development Set F1 Score:", dev_f1_gluten)

# test set
test_predictions_gluten = gluten_xgb_model.predict(X_test)
test_accuracy_gluten = accuracy_score(y_test_gluten, test_predictions_gluten)
test_f1_gluten = f1_score(y_test_gluten, test_predictions_gluten)
# print("Test Set Accuracy:", test_accuracy_gluten)
# print("Test Set F1 Score:", test_f1_gluten)

# Output results
#print("BAKERY -- XGB")
print(classification_report(y_test_gluten, test_predictions_gluten))


# VEGAN
y_train_vegan = training_data['vegan']
y_dev_vegan = devset_data['vegan']
y_test_vegan = testing_data['vegan']

#new base case
vegan_xgb_base = xgb.XGBClassifier(eta=1.0)
vegan_xgb_base.fit(X_train, y_train_vegan)

dev_predictions = vegan_xgb_base.predict(X_dev)
dev_accuracy = accuracy_score(y_dev_vegan, dev_predictions)
dev_f1 = f1_score(y_dev_vegan, dev_predictions)
print("CRUELTY FREE -- XGB")
print("Development Set Accuracy with default:", dev_accuracy)
print("Development Set F1 Score with default:", dev_f1)
#new

#TUUUUNING
params = [0.01, 0.1, 0.5, 0.9, 1.0, 10, 12, 15, 50, 100, 1000]

best_f1_score = 0
alpha = None
for value in params:
    vegan_xgb_tuning = xgb.XGBClassifier(eval_metric='logloss', eta=value)
    vegan_xgb_tuning.fit(X_train, y_train_gluten)

    dev_predictions = vegan_xgb_tuning.predict(X_dev)

    current_f1_score = f1_score(y_dev_gluten, dev_predictions)

    if current_f1_score >= best_f1_score:
        best_f1_score = current_f1_score
        alpha = value
print("Best parameter was ", alpha, " with an F1 of: ", best_f1_score)

vegan_xgb_model = xgb.XGBClassifier(eval_metric='logloss', eta=alpha)
vegan_xgb_model.fit(X_train, y_train_vegan)

# dev set
dev_predictions_vegan = vegan_xgb_model.predict(X_dev)
dev_accuracy_vegan = accuracy_score(y_dev_vegan, dev_predictions_vegan)
dev_f1_vegan = f1_score(y_dev_vegan, dev_predictions_vegan)
# print("CRUELTY FREE -- XGB")
# print("Development Set Accuracy:", dev_accuracy_vegan)
# print("Development Set F1 Score:", dev_f1_vegan)

# test set
test_predictions_vegan = vegan_xgb_model.predict(X_test)
test_accuracy_vegan = accuracy_score(y_test_vegan, test_predictions_vegan)
test_f1_vegan = f1_score(y_test_vegan, test_predictions_vegan)
# print("Test Set Accuracy:", test_accuracy_vegan)
# print("Test Set F1 Score:", test_f1_vegan)

# Output results
#print("CRUELTY FREE -- XGB")
print(classification_report(y_test_vegan, test_predictions_vegan))


# Combine the true labels for all classifiers
y_true_combined = np.vstack(
    (testing_data['nut_free'], testing_data['meat_free'], testing_data['dairy_free'], testing_data['gluten_free'], testing_data['vegan'])).T

# Combine the predictions for all classifiers
y_pred_combined = np.vstack(
    (test_predictions_nut, test_predictions_meat, test_predictions_dairy, test_predictions_gluten, test_predictions_vegan)).T

# F1 scores for each XGB classifier
xgb_f1_scores = [
    test_f1_nut,  # Nut-free
    test_f1_meat,  # Meat-free
    test_f1_dairy,  # Dairy-free
    test_f1_gluten,  # Gluten-free
    test_f1_vegan  # Vegan
]

# Calculate the average Macro F1 score for XGB classifiers
combined_macro_f1_xgb = sum(xgb_f1_scores) / len(xgb_f1_scores)
print("Combined Macro F1 Score for XGB classifiers:", combined_macro_f1_xgb)
