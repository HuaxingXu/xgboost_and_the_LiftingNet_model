import numpy as np
import catboost as cb

train_data = np.random.randint(0, 100, size=(100, 10))
train_label = np.random.randint(0, 2, size=(100))
test_data = np.random.randint(0,100, size=(50,10))

model = cb.CatBoostClassifier(iterations=2, depth=2, learning_rate=0.5, loss_function='MultiClass',
                              logging_level='Verbose')
model.fit(train_data, train_label, cat_features=[0,2,5])
preds_class = model.predict(test_data)
preds_probs = model.predict_proba(test_data)
print('class = ',preds_class)
print('proba = ',preds_probs)
