import pandas as pd
from flask import Flask, jsonify, request
import pickle

"""
	app.py script handle requests 
"""
# Load Model
model = pickle.load(open("model.pkl", "rb"))

# App
app = Flask(__name__)

# Routes
@app.route('/', methods=['POST'])

def predict():
	# get data
	data = request.get_json(force=True)

	# convert data into dataframe
	data.update((x, [y]) for x, y in data.items())
	data_df = pd.DataFrame.from_dict(data)

	# Make predictions
	result = model.predict(data_df)

	# send back to browser
	output = {"results": int(result[0])}

	# return data as json format
	return jsonify(results=output)


if __name__ == '__main__':
	app.run()#port = 8080, debug = True)

