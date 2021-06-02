import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from modlamp.plot import helical_wheel
import plotly.express as px
from PIL import Image
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modlamp.descriptors import GlobalDescriptor
from modlamp.descriptors import PeptideDescriptor
import os

cwd = os.getcwd()

VHSE = {"A" : [ 0.15 , -1.11 , -1.35 , -0.92 ,  0.02 , -0.91 ,  0.36 , -0.48],
        "R" : [-1.47 ,  1.45 ,  1.24 ,  1.27 ,  1.55 ,  1.47 ,  1.30 ,  0.83],
        "N" : [-0.99 ,  0.00 , -0.37 ,  0.69 , -0.55 ,  0.85 ,  0.73 , -0.80],
        "D" : [-1.15 ,  0.67 , -0.41 , -0.01 , -2.68 ,  1.31 ,  0.03 ,  0.56],
        "C" : [ 0.18 , -1.67 , -0.46 , -0.21 ,  0.00 ,  1.20 , -1.61 , -0.19],
        "Q" : [-0.96 ,  0.12 ,  0.18 ,  0.16 ,  0.09 ,  0.42 , -0.20 , -0.41],
        "E" : [-1.18 ,  0.40 ,  0.10 ,  0.36 , -2.16 , -0.17 ,  0.91 ,  0.02],
        "G" : [-0.20 , -1.53 , -2.63 ,  2.28 , -0.53 , -1.18 ,  2.01 , -1.34],
        "H" : [-0.43 , -0.25 ,  0.37 ,  0.19 ,  0.51 ,  1.28 ,  0.93 ,  0.65],
        "I" : [ 1.27 , -0.14 ,  0.30 , -1.80 ,  0.30 , -1.61 , -0.16 , -0.13],
        "L" : [ 1.36 ,  0.07 ,  0.26 , -0.80 ,  0.22 , -1.37 ,  0.08 , -0.62],
        "K" : [-1.17 ,  0.70 ,  0.70 ,  0.80 ,  1.64 ,  0.67 ,  1.63 ,  0.13],
        "M" : [ 1.01 , -0.53 ,  0.43 ,  0.00 ,  0.23 ,  0.10 , -0.86 , -0.68],
        "F" : [ 1.52 ,  0.61 ,  0.96 , -0.16 ,  0.25 ,  0.28 , -1.33 , -0.20],
        "P" : [ 0.22 , -0.17 , -0.50 ,  0.05 , -0.01 , -1.34 , -0.19 ,  3.56],
        "S" : [-0.67 , -0.86 , -1.07 , -0.41 , -0.32 ,  0.27 , -0.64 ,  0.11],
        "T" : [-0.34 , -0.51 , -0.55 , -1.06 , -0.06 , -0.01 , -0.79 ,  0.39],
        "W" : [ 1.50 ,  2.06 ,  1.79 ,  0.75 ,  0.75 , -0.13 , -1.01 , -0.85],
        "Y" : [ 0.61 ,  1.60 ,  1.17 ,  0.73 ,  0.53 ,  0.25 , -0.96 , -0.52],
        "V" : [ 0.76 , -0.92 , -0.17 , -1.91 ,  0.22 , -1.40 , -0.24 , -0.03],
        "B" : [ 0.00 ,  0.00 ,  0.00 ,  0.00 ,  0.00 ,  0.00 ,  0.00 ,  0.00],
        "-" : [ 0.00 ,  0.00 ,  0.00 ,  0.00 ,  0.00 ,  0.00 ,  0.00 ,  0.00]}
alphabet = 'BCDSQKIPTFNGHLRWAVEYM-'

def chopping (data, lim = 48):

  chopped = []
  for seq in data:
    if len(seq)<= lim:
      chopped.append(seq)

  return chopped

def padding (data, begin_token = '', end_token = '-' , lim = 48):

  padded = []
  for seq in data:
    temp = begin_token + seq + end_token * (lim - len(seq))
    padded.append(temp)

  return padded

def onehot_encoding (data):

  onehot_encoded = []
  for seq in data:
    temp = [[0 for i in alphabet] for j in seq]
    for j, aminoacid in enumerate(seq):
      for k, location in enumerate(alphabet):
        if aminoacid == location:
          temp[j][k] = 1
    onehot_encoded.append(temp)

  return onehot_encoded

def onehot_decoding (data):

  onehot_decoded = []
  for array in data:
    temp = ''
    for i, seq in enumerate(array):
      for j, aminoacid in enumerate(seq):
        if aminoacid > 0.9 :
          temp += alphabet[j]
    onehot_decoded.append(temp)

  return onehot_decoded

def vhse_encoding (data):
  vhse_encoded = []
  for seq in data:
    pep = []
    for aa in seq:
      pep.append(VHSE[aa])
    vhse_encoded.append(pep)
  return vhse_encoded

def prepare_VAE_data(data, csv = True):
  if csv:
    seq = pd.read_csv(data)["sequence"].tolist()
  else:
    seq = data
  seq = chopping(seq, lim = 48)
  seq = padding(seq, begin_token="", lim = 48)
  seq = onehot_encoding(seq)
  seq = np.asarray(seq)
  #seq = np.reshape(seq, (-1, 48*22))
  return seq

st.write("""
# AMP-VAE
> This application demonstrates the results of models trained on
a dataset of natural Anti-Microbial Peptides that are used for
activity prediction and *de-novo* peptide design
""")

st.sidebar.write("""
## Select Models
""")

generator = st.sidebar.selectbox('select generator model', ["VAE_v8"])
regressor = st.sidebar.selectbox('select regressor model', ["CNN_v3", "CNN_v4"])

st.sidebar.write("""
---
## Latent Dimensions
""")
dim = np.zeros(50)
for i in range(50):
    dim[i] = st.sidebar.slider("dimension_"+str(i), value = 0., min_value = -0.5, max_value = 0.5, step = 0.05)

st.write("""
---
## MIC prediction
Enter the peptide sequence to get the predicted Minimum Inhibitory Concentration
for E.coli
""")

text = st.text_input('seuqnce')
if text != '':
    seq = [text]
    seq = padding(seq)
    seq = vhse_encoding(seq)
    seq = np.asarray(seq)

    model = keras.models.load_model(regressor)
    mic = model.predict(seq)

    if mic >= 3:
        st.write("""
        **non-AMP**
        """)
    else:
        st.write("""
        **AMP**
        """)
        st.write("predicted MIC (uM):", 10**mic)

st.write("""
---
## Latent Space Visualization
Here we use t-SNE to see how the Variational Auto-Encoder represents AMP and non-AMP sequences
on the multi-dimensional latent space
""")

encoder = keras.models.load_model(generator+"_encoder")
decoder = keras.models.load_model(generator+"_decoder")
df = pd.read_csv(generator+"_latent_space.csv")

fig = px.scatter(
    df, x="tsne_1", y="tsne_2",
    size = "length", size_max = 7 ,color = "AMP", hover_name="sequence",
    )
st.plotly_chart(fig, use_container_width=True)

st.write("""
---
## Latent Space Interpolation
Enter a peptide sequence and change the value of every latent dimension to see how the generated peptide changes
""")

text = st.text_input('peptide')
if text != '':
    seq = [text]
    seq = prepare_VAE_data(seq, csv = False)

    _,_,latent = encoder.predict(seq)
    latent_arr = np.array(latent)

    interpolated_arr = latent_arr + dim
    generated_peptide = decoder.predict(interpolated_arr)
    generated_peptide = np.reshape(generated_peptide,(-1,48,22))
    generated_peptide = onehot_decoding(generated_peptide)
    generated_peptide = generated_peptide[0].strip("-")
    st.write(">", generated_peptide)

    bio_generated_peptide = ProteinAnalysis(generated_peptide)
    modlamp_generated_peptide = GlobalDescriptor(generated_peptide)

    length = len(generated_peptide)
    molecular_weight = bio_generated_peptide.molecular_weight()
    modlamp_generated_peptide.calculate_charge()
    charge = modlamp_generated_peptide.descriptor[0][0]
    modlamp_generated_peptide.charge_density()
    charge_density = modlamp_generated_peptide.descriptor[0][0]
    isoelectric_point = bio_generated_peptide.isoelectric_point()
    gravy = bio_generated_peptide.gravy()
    aromaticity = bio_generated_peptide.aromaticity()

    col1, col2 = st.beta_columns(2)
    with col1:
        helical_wheel(generated_peptide, moment = True, filename = os.path.join(cwd, "wheel.png"))
        image = Image.open(os.path.join(cwd, "wheel.png"))
        st.image(image,use_column_width="auto")

    with col2:
        st.write("length :", length)
        st.write("molecular_weight :" , molecular_weight)
        st.write("charge :", charge)
        st.write("charge_density :", charge_density)
        st.write("isoelectric_point :", isoelectric_point)
        st.write("gravy :", gravy)
        st.write("aromaticity :", aromaticity)
