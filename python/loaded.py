import pickle
import pandas as pd


test = pd.read_csv("../csv/test.csv")
filename = '../model/finalized_model_tree.sav'
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(test)
print(y_pred)
