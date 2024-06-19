from flask import Flask, render_template, request, jsonify
from compressive_concrete import load_model as load_compressive_model, predict_concrete_strength
from slumpt import load_model as load_slump_model, predict_slump
from high_com_con import load_model as load_highconcrete_model, predict_highconcretestrength
from reverse import predict_and_calculate_ratios


app = Flask(__name__)



@app.route("/", methods=["GET"])
def home():
    return render_template(r"home.html")

@app.route("/concrete", methods=["GET", "POST"])
def concrete():
#     featureDict = {
#     'cement': 0,
#     'slag': 0,
#     'ash': 0,
#     'water': 0,
#     'superPlastic': 0,
#     'coarseAgg': 0,
#     'fineAgg': 0,
#     'age': 0,
# }
    modelPath = r'models/RandomForestRegressor.sav'
# Add entry to the database
    if request.method == "POST":
        # Check if the request contains JSON data
        if request.is_json:
            # Get the JSON data
            data = request.get_json()

            # Initialize an empty dictionary
            featureDict = {}

            # Extract data from JSON
            featureDict['cement'] = float(data.get("cement", 0))
            featureDict['slag'] = float(data.get("slag", 0))
            featureDict['ash'] = float(data.get("ash", 0))
            featureDict['water'] = float(data.get("water", 0))
            featureDict['superPlastic'] = float(data.get("superPlastic", 0))
            featureDict['coarseAgg'] = float(data.get("coarseAgg", 0))
            featureDict['fineAgg'] = float(data.get("fineAgg", 0))
            featureDict['age'] = float(data.get("age", 0))

            # Assume train_model and predict_concrete_strength are defined elsewhere
            model = load_compressive_model(modelPath)
            yHat = predict_concrete_strength(model, featureDict)
            
            # Prepare the response data
            response_data = {"predicted_strength": yHat}
            # print(response_data)
            return jsonify(response_data), 200
        else:
            return jsonify({"error": "Request must be JSON"}), 400

    return render_template(r"concrete.html")

@app.route("/slump", methods=["GET", "POST"])
def slump():

#     featureDict = {
#     'cement': 0,
#     'slag': 0,
#     'ash': 0,
#     'water': 0,
#     'superPlastic': 0,
#     'coarseAgg': 0,
#     'fineAgg': 0,
#     'silicafumes':0,
# }
    modelPath=r'models/grad_boost.sav'

# Add entry to the database
    if request.method == "POST":
        # Check if the request contains JSON data
        if request.is_json:
            # Get the JSON data
            data = request.get_json()

            # Initialize an empty dictionary
            featureDict = {}

            # Extract data from JSON
            featureDict['cement'] = float(data.get("cement", 0))
            featureDict['slag'] = float(data.get("slag", 0))
            featureDict['ash'] = float(data.get("ash", 0))
            featureDict['water'] = float(data.get("water", 0))
            featureDict['superPlastic'] = float(data.get("superPlastic", 0))
            featureDict['coarseAgg'] = float(data.get("coarseAgg", 0))
            featureDict['fineAgg'] = float(data.get("fineAgg", 0))
            featureDict['silicafumes'] = float(data.get("silicafumes", 0))

            # Assume train_model and predict_concrete_strength are defined elsewhere
            model = load_slump_model(modelPath)
            yHat = predict_slump(model, featureDict)
            
            # Prepare the response data
            response_data = {"predicted_strength": yHat}
            # print(response_data)
            return jsonify(response_data), 200
        else:
            return jsonify({"error": "Request must be JSON"}), 400

    return render_template(r"slump.html")

@app.route("/highconcrete", methods=["GET", "POST"])
def highconcrete():

#     featureDict = {
#     'cement': 0,
#     'slag': 0,
#     'ash': 0,
#     'water': 0,
#     'superPlastic': 0,
#     'coarseAgg': 0,
#     'fineAgg': 0,
#     'age': 0,
#     'silicafumes':0,
# }
    modelPath=r'models/linear.sav'

# Add entry to the database
    if request.method == "POST":
        # Check if the request contains JSON data
        if request.is_json:
            # Get the JSON data
            data = request.get_json()

            # Initialize an empty dictionary
            featureDict = {}

            # Extract data from JSON
            featureDict['cement'] = float(data.get("cement", 0))
            featureDict['slag'] = float(data.get("slag", 0))
            featureDict['ash'] = float(data.get("ash", 0))
            featureDict['water'] = float(data.get("water", 0))
            featureDict['superPlastic'] = float(data.get("superPlastic", 0))
            featureDict['silicafumes'] = float(data.get("silicafumes", 0))
            featureDict['coarseAgg'] = float(data.get("coarseAgg", 0))
            featureDict['fineAgg'] = float(data.get("fineAgg", 0))
            featureDict['age'] = float(data.get("age", 0))

            # Assume train_model and predict_concrete_strength are defined elsewhere
            model = load_highconcrete_model(modelPath)
            yHat = predict_highconcretestrength(model, featureDict)
            
            # Prepare the response data
            response_data = {"predicted_strength": yHat}
            # print(response_data)
            return jsonify(response_data), 200
        else:
            return jsonify({"error": "Request must be JSON"}), 400

    return render_template(r"highconcrete.html")


@app.route("/reverse", methods=["GET", "POST"])
def reverse():
# Add entry to the database
    if request.method == "POST":
        # Check if the request contains JSON data
        if request.is_json:
            # Get the JSON data
            data = request.get_json()


            # print(float(data.get("strength", 0)))
            # Assume train_model and predict_concrete_strength are defined elsewhere
            yHat = predict_and_calculate_ratios(float(data.get("strength", 0)))
            
            # Prepare the response data
            response_data = {"cement": round(yHat[0], 3),
                             "water": round(yHat[1], 3),
                             "fine_aggregate": round(yHat[2], 3),
                             "coarse_aggregate": round(yHat[3], 3),
                             }
            # print(response_data)
            return jsonify(response_data), 200
        else:
            return jsonify({"error": "Request must be JSON"}), 400

    return render_template(r"reverse.html")


if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)




