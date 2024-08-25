from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import openai
from openai import OpenAI
from model import Classifier, CategoricalEmbedding
import re
import json, requests

size = 43659

columns = {
    "smoker_type": [0.0, 1.0, 2.0],
    "education": [0.0, 1.0, 2.0, 3.0],
    "drinker_type": [0.0, 1.0, 2.0, 3.0],
    "actual_sports_ranked": [0.0, 1.0, 2.0, 3.0, 4.0],
    "second_hand_smoking": [0.0, 1.0, 2.0],
    "age_group": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "mood_disorder": [0.0, 1.0, 2.0],
    "anxiety_disorder": [0.0, 1.0, 2.0],
    "life_stress": [0.0, 1.0, 2.0, 3.0, 4.0],
    "marital_status": [0.0, 1.0, 2.0, 3.0],
    "balanced_food": [0.0, 1.0, 2.0, 3.0],
    "perceived_health": [0.0, 1.0, 2.0, 3.0],
    "perceived_mental_health": [0.0, 1.0, 2.0],
    "life_satisfaction": [0.0, 1.0, 2.0, 3.0],
    "health_care_provider": [0.0, 1.0, 2.0],
    "sex": [0.0, 1.0],
}

value_to_text_mapping = {
    "smoker_type": {1.0: "Current or former smoker", 2.0: "Experimental or Lifetime non smoker"},
    "perceived_health": {1.0: "Excellent/Very Good", 2.0: "Good", 3.0: "Fair/Poor"},
    "perceived_mental_health": {1.0: "Excellent/ Very Good", 2.0: "Good", 3.0: "Fair/Poor"},
    "education": {1.0: "Less than secondary school graduation", 2.0: "Secondary school graduation, no post-secondary education", 3.0: "Post-secondary certificate diploma or univ degree"},
    "drinker_type": {1.0: "Regular or Occasional drinker", 2.0: "Did not drink"},
    "actual_sports_ranked": {1.0: "Ranked 0 - 25th percentile", 2.0: "Ranked 25 -50th percentile", 3.0: "Ranked 50 - 75th percentile", 4.0: "Ranked 75 - 100th percentile"},
    "health_care_provider": {1.0: "No", 2.0: "Yes"},
    "second_hand_smoking": {1.0: "No", 2.0: "Yes"},
    "age_group": {0.0: "18 - 24", 1.0: "25 - 34", 2.0: "35 - 44", 3.0: "45 - 54", 4.0: "54 - 65", 5.0: "above 65"},
    "mood_disorder": {1.0: "No", 2.0: "Yes"},
    "anxiety_disorder": {1.0: "No", 2.0: "Yes"},
    "life_stress": {1.0: "Not at all stressful", 2.0: "Not very stressful", 3.0: "A bit stressful", 4.0: "Quite a bit/Extremely stressful"},
    "marital_status": {1.0: "Married/ Common-Law", 2.0: "Widowed/Separated/Divorced", 3.0: "Single Never Married"},
	"life_satisfaction": {1.0: "Very satisfied / Satisfied", 2.0: "Neutral", 3.0: "Very dissatisfied / Dissatisfied"},
    "balanced_food": {1.0: "Often True", 2.0: "Sometimes True", 3.0: "Never True"},
    "sex": {0.0: "Male", 1.0: "Female"},
}

feature_order = ["smoker_type", "education", "drinker_type", "actual_sports_ranked", "second_hand_smoking", "age_group", "mood_disorder", "anxiety_disorder", "life_stress", "marital_status", "balanced_food", "perceived_health", "perceived_mental_health", "life_satisfaction", "health_care_provider", "sex"]

df = pd.DataFrame({col: np.random.choice(values, p=[0.05] + [0.95/(len(values)-1)]*(len(values)-1), size=size) for col, values in columns.items()})

metabolic_syndrome_risk_values = [0.0]*33844 + [2.0]*9815
np.random.shuffle(metabolic_syndrome_risk_values)

df["metabolic_syndrome_risk"] = metabolic_syndrome_risk_values

embedding_dimensions = 4
unique_dim = 0
attention_count = 2
param_expan_factor = 2
maximum_input_length = 16
output_dim = 1  
dp_rate = 0.15 
num_layers = 3

df_new = df.copy()
df_new = df_new.drop(columns=['metabolic_syndrome_risk'])
df_new = df_new.astype(int)
embeddings = {col: CategoricalEmbedding(df_new[col].max() + 1, embedding_dimensions, unique_dim) for col in df_new.columns}

from flask import Flask
import torch

app = Flask(__name__)

model = Classifier(embeddings, maximum_input_length, embedding_dimensions, attention_count, param_expan_factor, num_layers, dp_rate, output_dim)
state_dict = torch.load('model.pt')
model.load_state_dict(state_dict, strict=False)
model.eval()

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            features = [data.get(col, "") for col in feature_order]

            if "" in features:
                return jsonify({'prediction_text': 'Please fill in all fields before predicting.', 'risk_level': ''})

            actual_sports_hours = float(data.get('actual_sports_ranked', 0.0))
            if actual_sports_hours <= 15.0:
                actual_sports_class = 1.0
            elif actual_sports_hours <= 250.0:
                actual_sports_class = 2.0
            elif actual_sports_hours <= 360.0:
                actual_sports_class = 3.0
            else:
                actual_sports_class = 4.0
            
            features[feature_order.index('actual_sports_ranked')] = actual_sports_class
			
            features = torch.tensor([[float(feature) for feature in features]])
			
            with torch.no_grad():
                output = model(features)
            
            probability = torch.sigmoid(output)

            classification = (probability > 0.5).float()

            risk_level = "High Risk" if classification.item() == 1.0 else "Low Risk"

            high_risk_probability = probability.item() * 100
            low_risk_probability = (1 - probability.item()) * 100

            feature_values = features.tolist()[0]

            features_dict = dict(zip(feature_order, feature_values))
			
            textual_features = {col: value_to_text_mapping[col][value] for col, value in features_dict.items()}

            description = f"I am predicted to have {risk_level} for metabolic syndrome. "

            description += f"I am a {textual_features['sex']} in the age group of {textual_features['age_group']}. I'm {textual_features['marital_status']}. I have achieved {textual_features['education']}. I am a {textual_features['smoker_type']} and my perceived health is {textual_features['perceived_health']}. My perceived mental health is {textual_features['perceived_mental_health']}. I am a {textual_features['drinker_type']}. My physical activities are ranked as {textual_features['actual_sports_ranked']}. I have a health provider: {textual_features['health_care_provider']}, and I am exposed to second hand smoking: {textual_features['second_hand_smoking']}. I have mood disorder: {textual_features['mood_disorder']}, and anxiety disorder: {textual_features['anxiety_disorder']}. My life stress is {textual_features['life_stress']}. I am {textual_features['life_satisfaction']} to life and if can't afford balanced food: {textual_features['balanced_food']}."

            description += " Could you provide some general medical suggestion for my case?"

            print(description)

            client = OpenAI(api_key='OPEN AI API KEY')

            completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful professional medical assistant."},
                {"role": "user", "content": description}
                ],
                max_tokens=128,
                )
            
            content = completion.choices[0].message.content
            wellness_tips_string = re.sub(r'[#*]', '', content)
            print(wellness_tips_string)

            prediction_text = f'Model\'s output (classification): <b>{risk_level}</b>. The probability of High Risk is <b>{high_risk_probability:.3f}%</b>.<br>{wellness_tips_string}'
            return jsonify({'prediction_text': prediction_text, 'risk_level': risk_level})

        except Exception as e:
            return jsonify({'error': str(e)})
    
    return render_template('predict.html')

if __name__ == "__main__":
    app.run()
