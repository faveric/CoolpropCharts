from classes import Diagram, Line
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import plotly.graph_objs as go


def propane_cycle_example():

    # Create a new diagram
    diagram = Diagram("Propane Refrigeration Cycle", "HEOS", "Propane")

    # Convert temperatures to Kelvin
    T_evap = 273.15 + 2  # 2°C to K
    T_cond = 273.15 + 35  # 35°C to K
    T_sh = T_evap + 15  # Superheat temperature (15K above evaporation)

    # Create a temporary state to get pressures
    state = AbstractState("HEOS", "Propane")

    # Get evaporator pressure (at saturated conditions)
    state.update(CP.QT_INPUTS, 0, T_evap)
    P_evap = state.p()

    # Get condenser pressure (at saturated conditions)
    state.update(CP.QT_INPUTS, 0, T_cond)
    P_cond = state.p()

    # Point 1: After evaporator (superheated vapor)
    diagram.add_point("1", T=T_sh, P=P_evap)

    # Point 2: After isentropic compression
    diagram.add_point("2", P=P_cond, S=diagram.points["1"].smass())

    # Point 3: Saturated liquid after condenser
    diagram.add_point("3", T=T_cond, Q=0)

    # Point 4: After expansion valve (same enthalpy as point 3)
    diagram.add_point("4", P=P_evap, H=diagram.points["3"].hmass())

    # Add lines connecting the points
    diagram.add_line("1-2", "1", "2",  "isentropic")  # Isentropic compression
    diagram.add_line("2-3", "2", "3",  "isobaric")  # Isobaric condensation
    diagram.add_line("3-4", "3", "4",  "isoenthalpic")  # Isenthalpic expansion
    diagram.add_line("4-1", "4", "1",  "isobaric")  # Isobaric evaporation

    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='white',
        template="plotly_white"
    )
    diagram.plot(fig)
    fig.show()

    # Calculate COP and other cycle parameters
    def calculate_cycle_parameters(diagram):
        # Get states
        state1 = diagram.points["1"]

if __name__ == "__main__":
    propane_cycle_example()