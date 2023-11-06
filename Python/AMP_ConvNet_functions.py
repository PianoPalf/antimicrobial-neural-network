#################################################
# FUNCTIONS for Anti-Microbial Peptide Neural Network
#################################################

# Import Dependencies
import pandas as pd
import numpy as np
from Bio import SeqIO
from io import StringIO
from Bio.SeqUtils import molecular_weight
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import plotly.graph_objs as go

#########################################

# FUNCTION: Parses FASTA Input from User and converts it to Pandas DataFrame
def process_fasta(user_input):
    fasta_text = user_input

    # Use StringIO to simulate a file-like object for SeqIO
    fasta_file = StringIO(fasta_text)

    # Parse the FASTA text
    records = list(SeqIO.parse(fasta_file, "fasta"))
    sequence_ids = []
    sequences = []
    # Access sequence information
    # Access sequence information
    for record in records:
        sequence_ids.append(record.id)
        sequences.append(str(record.seq.upper()))
    
    data = {'Sequence_ID': sequence_ids, 'Sequence': sequences}
    df = pd.DataFrame(data)

    return df

#########################################

# Reduce complexity of data by Binning Amino Acids based on physical & chemical properties

# FUNCTION: Bins Amino Acid Sequence based on Amino Acid Properties
def bin_aa(sequence):
    # Create Dictionary containing Amino Acid Bins 
    # Note that other Bins can be created eg. Hydrophobic/Hydrophilic, Murphy 10 etc. 
    aa_bins = { # Murphy 8 Categories
        'H': 'H', # H to H
        'P': 'P', # P to P
        'L': 'L', 'V': 'L', 'I': 'L', 'M': 'L', 'C': 'L', # L, V, I, M, C to L 
        'A': 'A', 'G': 'A', # A, G to A
        'S': 'S', 'T': 'S', # S, T to S
        'F': 'F', 'Y': 'F', 'W': 'F', # F, Y, W to F
        'E': 'E', 'D': 'E', 'N': 'E', 'Q': 'E', # E, D, N, Q to E
        'K': 'K', 'R': 'K' # K, R to K
    }   
    return ''.join(aa_bins.get(base, base) for base in sequence)

#########################################

# FUNCTION: Calculates Molecular Weights (in kDa) of Amino Acid Sequences (as pd.Series)
def calc_molecular_weight(records):
    molecular_weights = []
    
    for record in records:
        seq = Seq(record)
        weight = molecular_weight(seq, "protein")
        weight = (weight/1000)
        weight = round(weight, 2)
        molecular_weights.append(weight)
    
    return molecular_weights

#########################################

# Generate Kyte-Doolittle Scores for Amino Acid Sequences

# Create Kyte-Doolittle (Hydrophobicity) score Dictionary
kyte_doolittle_scores = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# FUNCTION: Calculate Kyte-Doolittle scores for Amino Acid Sequence
def list_hydrophobicities(sequence):
    return [kyte_doolittle_scores.get(aa, 0) for aa in sequence]

# FUNCTION: Calculate total Kyte-Doolittle score for Amino Acid Sequence
def sum_hydrophobicities(sequence):
    return sum([kyte_doolittle_scores.get(aa, 0) for aa in sequence])

#########################################

# FUNCTION: Calculates Isoelectric Point (pI) of Amino Acid Sequence (as pd.Series)
def calculate_pI(series):
    pI_values = []

    for sequence in series:
        seq = Seq(sequence)
        protein_analysis = ProteinAnalysis(str(seq))
        pI = protein_analysis.isoelectric_point()
        pI_values.append(pI)
    
    return pI_values

#########################################

# FUNCTION: One-Hot Encodes Amino Acid Sequence
def one_hot_encode(sequence):
    # Amino Acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    # Map unique Amino Acids to Column indices
    amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    one_hot_matrix = np.zeros((len(sequence), len(amino_acids)), dtype=int)
    for i, aa in enumerate(sequence):
        if aa in amino_acid_to_index:
            one_hot_matrix[i, amino_acid_to_index[aa]] = 1
    return one_hot_matrix

#########################################

# FUNCTION: Pads One-Hot-Encoded Array Sequences
def pad_arrays(arr_list, desired_len):
    while len(arr_list) < desired_len:
        arr_list = np.vstack((arr_list, np.zeros(arr_list[0].shape, dtype=np.int64)))
    return arr_list

#########################################

# FUNCTION: Creates Plotly Bubble Chart to visualise AMP Score, MW and pI
def create_bubble_chart(df):
    # Define a color scale for the bubbles
    color_scale = ['#f65b74', '#23b0bd']
    font_color = '#333333'
    font = 'Helvetica Neue'
    mw_kDa = (df['MW (kDa)'])
    fig = go.Figure()

    scatter_trace = go.Scatter(
        x=df['AMP Score'],
        y=df['Isoelectric Point (pI)'],
        mode='markers',
        marker=dict(
            size=mw_kDa,
            sizeref=0.1,
            color=df['AMP Score'],
            colorscale=color_scale,
            cmin=0,
            cmax=1,
            colorbar=dict(title='AMP Score')
        ),
        text=df['ID'],
        hovertemplate='<b>%{text}</b><br>MW: %{marker.size:.2f} kDa<br>AMP Score: %{marker.color:.2f}'
    )

    fig.add_trace(scatter_trace)

    fig.update_layout(
        title='AMP Score, Molecular Weight and pI Visualised',
        xaxis_title='AMP Score',
        yaxis_title='Isoelectric Point (pI)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(242, 242, 242, 0.8)',
        title_font=dict(family=font, color=font_color),
        xaxis=dict(
            title_font=dict(family=font, color=font_color),
            tickfont=dict(family=font, color=font_color)
        ),
        yaxis=dict(
            title_font=dict(family=font, color=font_color),
            tickfont=dict(family=font, color=font_color)
        ),
        legend=dict(
            title_font=dict(family=font, color=font_color),
            font=dict(family=font, color=font_color)
        )
    )
    return fig