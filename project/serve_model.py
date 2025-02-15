from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder=".", static_url_path="")

# Build an absolute path to model.pkl relative to this file
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(model_path)

# List of one-hot encoded country columns used during training.
COUNTRY_COLUMNS = [
    'country_Albania', 'country_Algeria', 'country_Angola',
    'country_Antigua and Barbuda', 'country_Argentina', 'country_Armenia',
    'country_Australia', 'country_Austria', 'country_Azerbaijan',
    'country_Bahamas', 'country_Bahrain', 'country_Bangladesh',
    'country_Barbados', 'country_Belarus', 'country_Belgium',
    'country_Belize', 'country_Benin', 'country_Bermuda', 'country_Bhutan',
    'country_Bolivia', 'country_Bonaire; Sint Eustatius; Saba',
    'country_Bosnia and Herzegowina', 'country_Botswana', 'country_Brazil',
    'country_British Indian Ocean Territory', 'country_Brunei Darussalam',
    'country_Bulgaria', 'country_Burkina Faso', 'country_Burundi',
    'country_Cambodia', 'country_Cameroon', 'country_Canada',
    'country_Cape Verde', 'country_Cayman Islands', 'country_Chile',
    'country_China', 'country_Colombia', 'country_Congo',
    'country_Congo The Democratic Republic of The', 'country_Costa Rica',
    "country_Cote D'ivoire", 'country_Croatia (LOCAL Name: Hrvatska)',
    'country_Cuba', 'country_Curacao', 'country_Cyprus',
    'country_Czech Republic', 'country_Denmark', 'country_Djibouti',
    'country_Dominica', 'country_Dominican Republic', 'country_Ecuador',
    'country_Egypt', 'country_El Salvador', 'country_Estonia',
    'country_Ethiopia', 'country_European Union', 'country_Faroe Islands',
    'country_Fiji', 'country_Finland', 'country_France', 'country_Gabon',
    'country_Gambia', 'country_Georgia', 'country_Germany', 'country_Ghana',
    'country_Gibraltar', 'country_Greece', 'country_Guadeloupe', 'country_Guam',
    'country_Guatemala', 'country_Haiti', 'country_Honduras', 'country_Hong Kong',
    'country_Hungary', 'country_Iceland', 'country_India', 'country_Indonesia',
    'country_Iran (ISLAMIC Republic Of)', 'country_Iraq', 'country_Ireland',
    'country_Israel', 'country_Italy', 'country_Jamaica', 'country_Japan',
    'country_Jordan', 'country_Kazakhstan', 'country_Kenya',
    'country_Korea Republic of', 'country_Kuwait', 'country_Kyrgyzstan',
    "country_Lao People's Democratic Republic", 'country_Latvia',
    'country_Lebanon', 'country_Lesotho', 'country_Libyan Arab Jamahiriya',
    'country_Liechtenstein', 'country_Lithuania', 'country_Luxembourg',
    'country_Macau', 'country_Macedonia', 'country_Madagascar', 'country_Malawi',
    'country_Malaysia', 'country_Maldives', 'country_Malta', 'country_Mauritius',
    'country_Mexico', 'country_Moldova Republic of', 'country_Monaco',
    'country_Mongolia', 'country_Montenegro', 'country_Morocco',
    'country_Mozambique', 'country_Myanmar', 'country_Namibia', 'country_Nauru',
    'country_Nepal', 'country_Netherlands', 'country_New Caledonia',
    'country_New Zealand', 'country_Nicaragua', 'country_Niger',
    'country_Nigeria', 'country_Norway', 'country_Oman', 'country_Pakistan',
    'country_Palestinian Territory Occupied', 'country_Panama',
    'country_Papua New Guinea', 'country_Paraguay', 'country_Peru',
    'country_Philippines', 'country_Poland', 'country_Portugal',
    'country_Puerto Rico', 'country_Qatar', 'country_Reunion', 'country_Romania',
    'country_Russian Federation', 'country_Rwanda',
    'country_Saint Kitts and Nevis', 'country_Saint Martin', 'country_San Marino',
    'country_Saudi Arabia', 'country_Senegal', 'country_Serbia',
    'country_Seychelles', 'country_Singapore', 'country_Slovakia (SLOVAK Republic)',
    'country_Slovenia', 'country_South Africa', 'country_South Sudan',
    'country_Spain', 'country_Sri Lanka', 'country_Sudan', 'country_Sweden',
    'country_Switzerland', 'country_Syrian Arab Republic',
    'country_Taiwan; Republic of China (ROC)', 'country_Tajikistan',
    'country_Tanzania United Republic of', 'country_Thailand',
    'country_Trinidad and Tobago', 'country_Tunisia', 'country_Turkey',
    'country_Turkmenistan', 'country_Uganda', 'country_Ukraine',
    'country_United Arab Emirates', 'country_United Kingdom',
    'country_United States', 'country_Unknown', 'country_Uruguay',
    'country_Uzbekistan', 'country_Vanuatu', 'country_Venezuela',
    'country_Viet Nam', 'country_Virgin Islands (U.S.)', 'country_Yemen',
    'country_Zambia', 'country_Zimbabwe'
]

def process_input(data):
    """
    Process incoming JSON data to create a DataFrame for prediction.
    
    Expected keys include: user_id, signup_time, purchase_time, purchase_value,
    device_id, source, browser, sex, age, ip_address, and country.
    
    The function removes the "country" key from the data, sets all one-hot encoded
    country columns to 0, and then sets the column corresponding to the provided
    country to 1.
    """
    features = data.copy()
    
    # Extract the 'country' field and remove it from features
    country_input = features.pop("country", None)
    
    # Set all one-hot country columns to 0
    for col in COUNTRY_COLUMNS:
        features[col] = 0
    
    # If a country is provided, update its column to 1 (if it exists)
    if country_input:
        country_col = f"country_{country_input}"
        if country_col in COUNTRY_COLUMNS:
            features[country_col] = 1
        else:
            print(f"Warning: {country_input} is not recognized among the country columns.")
    
    # Convert the features dictionary into a DataFrame (one row)
    input_df = pd.DataFrame([features])
    return input_df

@app.route('/')
def home():
    # Serve the frontend (index.html) from the current directory
    return send_from_directory(os.path.join(os.path.dirname(__file__)), 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = process_input(data)
        prediction = model.predict(input_df)
        result = int(prediction[0])
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
