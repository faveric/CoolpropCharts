import pandas as pd
import streamlit as st
from CoolProp.CoolProp import AbstractState
import plotly.graph_objs as go
from functions import *
import numpy as np

# Import your Diagram and Line classes
from classes import Diagram, Line  # Adjust the import based on where the classes are defined

# Page configuration
page_config()

# Initialize session state for diagrams, points, and counters if they don't exist
if 'diagrams' not in st.session_state:
    st.session_state['diagrams'] = []
if 'selected_fluid' not in st.session_state:
    st.session_state['selected_fluid'] = 'n-Propane'

# Define fluids, properties and units
fluids = get_available_fluids()
properties, property_units = get_properties()

# Streamlit Sidebar Mode Select
st.session_state['operating_mode']=st.sidebar.radio('Select Operating Mode', ['Main', 'Reversed Thermodynamic Cycle'])

# Readme
page_info()

# Streamlit app title
page_title()

st.subheader('PLOT SETUP')
with st.container(border=True):
    page_instructions()
    # Fluid selection
    selected_fluid = st.selectbox("Select Fluid", fluids, index=fluids.index("n-Propane"))

    #Create Columns
    if st.session_state['operating_mode'] == 'Main':
        dia_col, point_col, line_col = st.columns([1/3, 1/3, 1/3])
    elif st.session_state['operating_mode'] == 'Reversed Thermodynamic Cycle':
        dia_col, point_col = st.columns([1/3, 2/3])

    # Add new diagram section
    with dia_col:
        page_diagram_handler(selected_fluid)


    # Add points to the active diagram
    with point_col:
        page_point_handler()

    # Add points to the active diagram
    if st.session_state['operating_mode'] == 'Main':
        with line_col:
            page_line_handler()

# Table with The plotted points
if st.session_state['active_diagram']:
    points_table(st.session_state['active_diagram'])
# Plot the active diagram with its points

if st.session_state['active_diagram']:
    plot_diagrams()