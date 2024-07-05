from flask import Flask, request, render_template
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)

# Definisikan variabel fuzzy
pregnancies = ctrl.Antecedent(np.arange(0, 20, 1), 'pregnancies')
glucose = ctrl.Antecedent(np.arange(0, 201, 1), 'glucose')
blood_pressure = ctrl.Antecedent(np.arange(0, 200, 1), 'blood_pressure')
skin_thickness = ctrl.Antecedent(np.arange(0, 100, 1), 'skin_thickness')
insulin = ctrl.Antecedent(np.arange(0, 846, 1), 'insulin')
bmi = ctrl.Antecedent(np.arange(0, 70, 1), 'bmi')
diabetes_pedigree_function = ctrl.Antecedent(np.arange(0, 2.5, 0.1), 'diabetes_pedigree_function')
age = ctrl.Antecedent(np.arange(20, 100, 1), 'age')
risk = ctrl.Consequent(np.arange(0, 2, 1), 'risk')

# Definisikan membership functions
pregnancies.automf(3)
glucose.automf(3)
blood_pressure.automf(3)
skin_thickness.automf(3)
insulin.automf(3)
bmi.automf(3)
diabetes_pedigree_function.automf(3)
age.automf(3)

risk['low'] = fuzz.trimf(risk.universe, [0, 0, 0.5])
risk['high'] = fuzz.trimf(risk.universe, [0.5, 1, 1])

# Definisikan aturan fuzzy
rule1 = ctrl.Rule(glucose['poor'] & bmi['poor'] & age['poor'] & insulin['poor'], risk['low'])
rule2 = ctrl.Rule(glucose['average'] & bmi['average'] & age['average'] & insulin['average'], risk['high'])
rule3 = ctrl.Rule(glucose['good'] & bmi['good'] & age['good'] & insulin['good'], risk['high'])

risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
risk_simulation = ctrl.ControlSystemSimulation(risk_ctrl)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pregnancies_value = float(request.form['pregnancies'])
        glucose_value = float(request.form['glucose'])
        blood_pressure_value = float(request.form['blood_pressure'])
        skin_thickness_value = float(request.form['skin_thickness'])
        insulin_value = float(request.form['insulin'])
        bmi_value = float(request.form['bmi'])
        diabetes_pedigree_function_value = float(request.form['diabetes_pedigree_function'])
        age_value = float(request.form['age'])
        
        risk_simulation.input['pregnancies'] = pregnancies_value
        risk_simulation.input['glucose'] = glucose_value
        risk_simulation.input['blood_pressure'] = blood_pressure_value
        risk_simulation.input['skin_thickness'] = skin_thickness_value
        risk_simulation.input['insulin'] = insulin_value
        risk_simulation.input['bmi'] = bmi_value
        risk_simulation.input['diabetes_pedigree_function'] = diabetes_pedigree_function_value
        risk_simulation.input['age'] = age_value
        
        risk_simulation.compute()
        risk_value = risk_simulation.output['risk']
        result = 'High Risk' if risk_value > 0.5 else 'Low Risk'
        
        return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
