#################################################
# Anti-Microbial Peptide Neural Network Predictions
#################################################

# Import dependencies
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.io as pio
import AMP_functions

# Set Display Options to prevent Exponential Notation
pd.set_option('display.float_format', '{:.2f}'.format)

#################################################
# Flask Setup
#################################################

app = Flask(__name__)

#################################################
# Flask Routes
#################################################

# Takes FASTA Input from User
@app.route('/', methods=['GET'])
def input_text():
    return render_template('index.html')

#################################################

# Converts FASTA Input into Pandas DF, Calculates AMP Score using TF Model and Plots Data
@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        user_text = request.form['text_input']
        
        # Process FASTA input from User into Pandas DataFrame
        user_text_df = AMP_functions.process_fasta(user_text)

        # Calculate Molecular Weight of Peptides
        user_text_df['MW (kDa)'] = AMP_functions.calc_molecular_weight(user_text_df['Sequence'])

        # Calculate Isoelectric Point (pI) of Peptides
        user_text_df['Isoelectric Point (pI)'] = AMP_functions.calculate_pI(user_text_df['Sequence'])
        
        # Calculate Hydrophobicity (Kyte-Doolittle Scores) for each Amino Acid
        user_text_df['Hydrophobicity'] = user_text_df['Sequence'].apply(AMP_functions.list_hydrophobicities)
        
        # Creae One-Hot Encoded Sequence Column
        user_text_df['One_Hot_Encoded'] = user_text_df['Sequence'].apply(AMP_functions.one_hot_encode)
        
        # Set Max Amino Acid Sequence Length
        max_length = 198

        # Pad 'One_Hot_Encoded' Column so that all Matrices are the same size (198 x 20)
        user_text_df['Padded_One_Hot'] = user_text_df['One_Hot_Encoded'].apply(lambda arr_list: AMP_functions.pad_arrays(arr_list, max_length))

        # Take Values for TensorFlow CNN Model
        user_text_sequences = user_text_df['Padded_One_Hot'].values

        # Convert Input Data to Numpy Array
        user_text_sequences = np.array([np.array(val) for val in user_text_sequences])

        # Load Trained Model
        model = tf.keras.models.load_model('HDF5_files/convolutional_nn_1.h5')

        # Make Predictions
        predictions = model.predict(user_text_sequences)
        predictions = predictions.ravel().tolist()

        # Create DataFrame containing Prediction Results for Plotting
        results_df = pd.DataFrame({'ID': user_text_df['Sequence_ID'],
                                'Sequence': user_text_df['Sequence'],
                                'MW (kDa)': user_text_df['MW (kDa)'],
                                'Isoelectric Point (pI)': user_text_df['Isoelectric Point (pI)'],
                                'AMP Score': predictions}).reset_index(drop=True)

        # Create Classification column based on AMP Score column
        results_df['Classification'] = np.where(results_df['AMP Score'] < 0.5, 'Non-AMP', 'AMP')
        
        # Check DataFrame in Terminal
        print(results_df)
        
        # Plot Bubble Chart
        bubble_chart = AMP_functions.create_bubble_chart(results_df)
        
        # Create Plotly Bubble Chart for HTML
        chart_div = pio.to_html(bubble_chart, full_html=False)

        return render_template('result.html', tables=[results_df.to_html(classes='data', index=False)], titles=results_df.columns.values, chart_div=chart_div)
    
    except Exception as e:
        # Handle Exception/Error
        error_message = f"Please ensure that your amino acid sequence(s) are in FASTA format and \
            contain no more than 198 residues per peptide."
        return render_template('error.html', error_message=(error_message))
        
#################################################

if __name__ == '__main__':
    app.run(debug=True)