from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Build correct absolute path for model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'property-price-model.pkl')
model = joblib.load(model_path)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        Building_Class = float(request.form['Building_Class'])
        Lot_Size = float(request.form['Lot_Size'])
        Neighborhood = float(request.form['Neighborhood'])
        Overall_Material = float(request.form['Overall_Material'])
        House_Type = float(request.form['House_Type'])
        House_Condition = float(request.form['House_Condition'])
        Property_Shape = float(request.form['Property_Shape'])
        Property_age = float(request.form['Property_age'])
        Kitchen_Quality = float(request.form['Kitchen_Quality'])
        House_Design = float(request.form['House_Design'])
        Air_Conditioning = float(request.form['Air_Conditioning'])
        Total_living_area = float(request.form['Total_living_area'])
        Garage_Area = float(request.form['Garage_Area'])
        Sale_Condition = float(request.form['Sale_Condition'])

        # Prepare input for model
        features = np.array([[Building_Class, Lot_Size, Neighborhood, Overall_Material,
                              House_Type, House_Condition, Property_Shape, Property_age,
                              Kitchen_Quality, House_Design, Air_Conditioning,
                              Total_living_area, Garage_Area, Sale_Condition]])

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template('index.html',
                               prediction_text=f"🏠 Estimated Property Sale Price: ₹{prediction:,.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
