import streamlit as st
import CoolProp.CoolProp as CP
from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from classes import *

#############################
### THERMODYNAM FUNCTIONS ###
#############################
def get_available_fluids() -> List[str]:
    """Return a sorted list of available CoolProp fluids."""
    return sorted(CP.get_global_param_string("fluids_list").split(','))

def get_properties():
    property_units = {
        'P': 'Pa', 'T': 'K', 'H': 'kJ/kg', 'S': 'kJ/(kg K)', 'D': 'kg/m³', 'Q': '-',
        'viscosity_dynamic': 'Pa·s', 'viscosity_kinematic': 'm²/s', 'conductivity': 'W/(m K)',
        'speed_of_sound': 'm/s', 'Prandtl': '-', 'cp': 'kJ/(kg K)', 'cv': 'kJ/(kg K)',
        'surface_tension': 'N/m', 'internal_energy': 'kJ/kg', 'Gibbs_mass': 'kJ/kg',
        'Helmholtz_mass': 'kJ/kg', 'molar_mass': 'g/mol', 'Hmolar': 'kJ/mol',
        'Smolar': 'kJ/(mol K)', 'Dmolar': 'mol/m³', 'cp_molar': 'J/(mol K)', 'cv_molar': 'J/(mol K)'
    }
    properties = list(property_units.keys())
    return properties, property_units

#############################
### STREAMLIT PAGE BLOCKS ###
#############################
def page_title():
    tit1, tit2 = st.columns(2, vertical_alignment="bottom")
    with tit1:
        st.image(".img/CoolPropLogo.png")
    with tit2:
        st.title("CoolProp Charts")
        st.markdown("Calculate thermodynamic properties and plot them using CoolProp")

def page_instructions():
    with st.expander("instructions"):
        st.markdown("""\n
        1. Select Working Mode on the sidebar\n
        2. Create a new diagram or select an existing diagram\n
        3. Add Points to the diagrams with two independent properties in main mode or with Thermodynamic cycle parameters\n
        4. Add Lines to the diagrams connecting points with thermo-physical processes""")

def page_diagram_handler(selected_fluid):
    with st.container(key='add_diagram_form', border=True):
        st.subheader("Diagram")
        st.markdown("Create a New Diagram or Select one from the list")
        new_diag_name = st.text_input("Enter new diagram name")
        add_diagram_button = st.button("Add new diagram")

        if add_diagram_button and new_diag_name:
            # Create a new Diagram object and store it
            new_diagram = Diagram(new_diag_name, "HEOS", selected_fluid)
            st.session_state['diagrams'].append(new_diagram)
            st.success(f"Diagram '{new_diag_name}' added.")
            # List of diagram names for selection
        diagram_names = [diagram.name for diagram in st.session_state['diagrams']]
        if diagram_names:
            active_diagram_name = st.selectbox('Select Diagram', diagram_names)
            # Find the selected diagram object by its name
            st.session_state['active_diagram'] = next(
                diagram for diagram in st.session_state['diagrams'] if diagram.name == active_diagram_name)
        else:
            st.info("No diagrams available. Please create a new diagram first.")
            st.session_state['active_diagram'] = None

def page_point_handler():
    properties, property_units = get_properties()
    if st.session_state['active_diagram']:
        with st.container(key='add_point_form', border=True):
            if st.session_state['operating_mode'] == 'Main':
                st.subheader('Points')
                st.markdown(f'Add points to active diagram: {st.session_state["active_diagram"].name}')
                # State point calculation input
                point_name = st.text_input('Enter point name')

                col1, col2 = st.columns(2)

                # Input for property 1
                with col1:
                    prop1 = st.selectbox("Select First Property", properties, key="prop1")
                    unit1 = property_units[prop1]
                    value1 = st.number_input(f"Enter {prop1} [{unit1}]", key="value1")

                # Input for property 2
                with col2:
                    remaining_properties = [p for p in properties if p != prop1]
                    prop2 = st.selectbox("Select Second Property", remaining_properties, key="prop2")
                    unit2 = property_units[prop2]
                    value2 = st.number_input(f"Enter {prop2} [{unit2}]", key="value2")

                # Submit button for point creation
                add_point_button = st.button("Save point")

                if add_point_button and point_name:
                    # Add the point to the active diagram
                    st.session_state['active_diagram'].add_point(point_name, **{prop1: value1, prop2: value2})
                    st.success(f"Point '{point_name}' added to diagram '{st.session_state['active_diagram'].name}'.")
            elif st.session_state['operating_mode'] == 'Reversed Thermodynamic Cycle':
                st.subheader('Cycle Parameters')
                st.markdown(f'Add points to active diagram: {st.session_state["active_diagram"].name}')

                # Input for evaporation and condensation temperatures
                T_evaporation = st.number_input("Enter Evaporation Temperature [K]", key="T_evaporation")
                T_condensation = st.number_input("Enter Condensation Temperature [K]", key="T_condensation")
                Superheat = st.number_input("Enter Superheat [K]", key="Superheat")
                Pump_Isentropic_Efficiency = st.number_input("Enter Pump Isentropic Efficiency", 0.0, 1.0, step=0.01)
                Subcooling = st.number_input("Enter Subcooling [K]", key="Subcooling")

                # Calculate Points
                if st.button("Calculate Cycle"):
                    st.session_state['active_diagram'].calculate_reversed_cycle(T_evaporation,
                                                                                T_condensation,
                                                                                Superheat,
                                                                                Pump_Isentropic_Efficiency,
                                                                                Subcooling)
                    st.success(f"Thermodynamic Cycle '{st.session_state['active_diagram'].name}' properties calculated.")

def page_line_handler():
    if st.session_state['active_diagram']:
        with st.container(key='add_line_form', border=True):
            st.subheader('Lines')
            st.markdown(f'Add Lines to active diagram: {st.session_state["active_diagram"].name}')
            # State point calculation input
            line_name = st.text_input('Enter line name')

            points = list(st.session_state['active_diagram'].points.keys())
            col1, col2 = st.columns(2)

            # Input for property 1
            with col1:
                point1 = st.selectbox("Select First Point", points, key="point1")

            # Input for property 2
            with col2:
                remaining_points = [p for p in points if p != point1]
                point2 = st.selectbox("Select Second Point", points, key="point2")

            # Submit button for point creation
            process = ['isothermal', 'isentropic', 'isobaric', 'isochoric', 'isoenthalpic', 'line']
            selected_process = st.selectbox("Select Process", process, key="process")
            add_line_button = st.button("Save line")

            if add_line_button and line_name:
                # Add the point to the active diagram
                try:
                    st.session_state['active_diagram'].add_line(line_name, point1, point2, selected_process)
                    st.success(f"Line '{line_name}' added to diagram '{st.session_state['active_diagram'].name}'.")
                except ValueError as e:
                    st.error(e)
def points_table(active_diagram):
    st.subheader('TABLE')
    with st.container(border=True):
        df = active_diagram.get_point_df()
        column_config = {
            col: st.column_config.NumberColumn(col, format="%.2e")
            for col in df.columns
        }

        # Display DataFrame with column configuration
        st.dataframe(df, column_config=column_config)

def plot_diagrams():
    properties, property_units = get_properties()
    st.subheader('PLOT')
    diagram_names = [diagram.name for diagram in st.session_state['diagrams']]
    with st.container(border=True):
        cd1, cd2 = st.columns(2, vertical_alignment="bottom")
        with cd1:
            prop_y = st.selectbox("Select y-axis property", properties, key="propy", index=properties.index("P"))
        with cd2:
            prop_x = st.selectbox("Select x-axis property", properties, key="propx", index=properties.index("H"))

        c1, c2 = st.columns(2, vertical_alignment="bottom")
        with c1:
            selected_diagrams = st.multiselect("Select Diagrams to plot", diagram_names)
        with c2:
            do_plot = st.button(f'Plot Selected Diagrams')
            if do_plot:
                fig = go.Figure()
                fig.update_layout(
                    plot_bgcolor='white',
                    template="plotly_white"
                )
                plot_diagrams = [diagram for diagram in st.session_state['diagrams'] if
                                 diagram.name in selected_diagrams]
                num_colors = len(plot_diagrams)
                colors = qualitative.Plotly[:num_colors]  # Get a color palette from Plotly's qualitative colors
                for sel_diag, color in zip(plot_diagrams, colors):
                    sel_diag.plot(fig, prop_y, prop_x, color)

        if do_plot:
            st.plotly_chart(fig)