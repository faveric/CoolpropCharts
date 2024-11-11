import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import CoolProp
from CoolProp.CoolProp import AbstractState
from typing import Dict, Tuple, Optional
import pandas as pd


class Line:
    """Class to represent a thermodynamic process line"""

    def __init__(self,
                 name: str,
                 start_state: AbstractState,
                 end_state: AbstractState,
                 constant_property: str,
                 variable_property: str,
                 fluid: str,
                 backend: str = 'HEOS'):
        self.name = name
        self.start = start_state
        self.end = end_state
        self.constant_property = constant_property.upper()
        self.variable_property = variable_property.upper()
        self.process_points = []  # List of AbstractState points along the line
        self.fluid = fluid
        self.backend = backend

        self.property_map = {
            'P': 'p',  # Pressure [Pa]
            'T': 'T',  # Temperature [K]
            'H': 'hmass',  # Specific enthalpy [J/kg]
            'S': 'smass',  # Specific entropy [J/kg/K]
            'D': 'rhomass',  # Density [kg/m³]
            'Q': 'Q',  # Vapor quality [-]
            'viscosity_dynamic': 'viscosity',  # Dynamic viscosity [Pa-s]
            'viscosity_kinematic': lambda state: state.viscosity() / state.rhomass(),  # Kinematic viscosity [m²/s]
            'conductivity': 'conductivity',  # Thermal conductivity [W/m/K]
            'speed_of_sound': 'speed_sound',  # Speed of sound [m/s]
            'Prandtl': 'Prandtl',  # Prandtl number [-]
            'cp': 'cpmass',  # Specific heat capacity at constant pressure [J/kg/K]
            'cv': 'cvmass',  # Specific heat capacity at constant volume [J/kg/K]
            'surface_tension': 'surface_tension',  # Surface tension [N/m]
            'internal_energy': 'umass',  # Internal energy [J/kg]
            'Gibbs_mass': 'gibbsmass',  # Gibbs free energy [J/kg]
            'Helmholtz_mass': 'helmholtzmass',  # Helmholtz free energy [J/kg]
            'molar_mass': 'molar_mass',  # Molar mass [kg/mol]
            'Hmolar': 'hmolar',  # Molar enthalpy [J/mol]
            'Smolar': 'smolar',  # Molar entropy [J/mol/K]
            'Dmolar': 'rhomolar',  # Molar density [mol/m³]
            'cp_molar': 'cpmolar',  # Molar specific heat capacity at constant pressure [J/mol/K]
            'cv_molar': 'cvmolar',  # Molar specific heat capacity at constant volume [J/mol/K]
        }

        # Update state using the two input properties
        if constant_property == "Noconstant":
            constant_property_name = 'p'
            use_line = True
        else:
            constant_property_name = self.property_map[constant_property]
            use_line = False


        variable_property_name = self.property_map[variable_property]

        # Generate the input pair using getattr to access CoolProp.iX dynamically
        constant_property_attr = getattr(CoolProp, f'i{constant_property_name.capitalize()}')
        variable_property_attr = getattr(CoolProp, f'i{variable_property_name.capitalize()}')

        # Check if the constant property is equal for start and end points
        constant_start = getattr(start_state, constant_property_name)()
        constant_end = getattr(end_state, constant_property_name)()

        # Create intermediate points along the variable property using linspace
        num_points = 50
        variable_start = getattr(start_state, variable_property_name)()
        variable_end = getattr(end_state, variable_property_name)()

        if use_line == True:
            for variable, constant in zip(np.linspace(variable_start, variable_end, num_points),
                                          np.linspace(constant_start, constant_end, num_points)):
                state = AbstractState(self.backend, self.fluid)
                ipair = CoolProp.CoolProp.generate_update_pair(constant_property_attr,
                                                               constant,
                                                               variable_property_attr,
                                                               variable)
                state.update(ipair[0], ipair[1], ipair[2])
                self.process_points.append(state)
        else:
            for variable in np.linspace(variable_start, variable_end, num_points):
                # Create a new AbstractState for each intermediate point
                state = AbstractState(self.backend, self.fluid)
                ipair = CoolProp.CoolProp.generate_update_pair(constant_property_attr,
                                                       constant_start,
                                                       variable_property_attr,
                                                       variable)
                state.update(ipair[0], ipair[1], ipair[2])
                self.process_points.append(state)

class Diagram:
    """Class to represent a thermodynamic diagram"""

    def __init__(self, name: str, backend: str, fluid: str):
        self.name = name
        self.fluid = fluid
        self.backend = backend
        self.points: Dict[str, AbstractState] = {}
        self.lines: Dict[str, Line] = {}

        # Property map for human-readable inputs
        self.property_map = {
            'P': 'p',  # Pressure [Pa]
            'T': 'T',  # Temperature [K]
            'H': 'hmass',  # Specific enthalpy [J/kg]
            'S': 'smass',  # Specific entropy [J/kg/K]
            'D': 'rhomass',  # Density [kg/m³]
            'Q': 'Q',  # Vapor quality [-]
            'viscosity_dynamic': 'viscosity',  # Dynamic viscosity [Pa-s]
            'conductivity': 'conductivity',  # Thermal conductivity [W/m/K]
            'speed_of_sound': 'speed_sound',  # Speed of sound [m/s]
            'Prandtl': 'Prandtl',  # Prandtl number [-]
            'cp': 'cpmass',  # Specific heat capacity at constant pressure [J/kg/K]
            'cv': 'cvmass',  # Specific heat capacity at constant volume [J/kg/K]
            'surface_tension': 'surface_tension',  # Surface tension [N/m]
            'internal_energy': 'umass',  # Internal energy [J/kg]
            'Gibbs_mass': 'gibbsmass',  # Gibbs free energy [J/kg]
            'Helmholtz_mass': 'helmholtzmass',  # Helmholtz free energy [J/kg]
            'molar_mass': 'molar_mass',  # Molar mass [kg/mol]
            'Hmolar': 'hmolar',  # Molar enthalpy [J/mol]
            'Smolar': 'smolar',  # Molar entropy [J/mol/K]
            'Dmolar': 'rhomolar',  # Molar density [mol/m³]
            'cp_molar': 'cpmolar',  # Molar specific heat capacity at constant pressure [J/mol/K]
            'cv_molar': 'cvmolar',  # Molar specific heat capacity at constant volume [J/mol/K]
        }

    def get_pressure_at_temperature(self, temperature: float) -> float:
        """Calculate the pressure at a given temperature for the fluid (using CoolProp)."""
        state = AbstractState(self.backend, self.fluid)
        state.update(CoolProp.QT_INPUTS,1.0, temperature)  # Use an arbitrary pressure to calculate the saturation pressure
        return state.p()  # Return the saturation pressure at the given temperature

    def add_point(self, name: str, **kwargs):
        """Add a state point to the diagram using any two properties"""
        try:
            # Create new AbstractState instance
            state = AbstractState(self.backend, self.fluid)

            # Convert input property names to CoolProp names
            input_props = [(k.upper(), v) for k, v in kwargs.items()]
            if len(input_props) != 2:
                raise ValueError("Exactly two properties must be specified")

            # Update state using the two input properties
            prop1_name = self.property_map[input_props[0][0]]
            prop2_name = self.property_map[input_props[1][0]]

            # Generate the input pair using getattr to access CoolProp.iX dynamically
            prop1_const = getattr(CoolProp, f'i{prop1_name.capitalize()}')
            prop2_const = getattr(CoolProp, f'i{prop2_name.capitalize()}')
            ipair = CoolProp.CoolProp.generate_update_pair(prop1_const,
                                                           input_props[0][1],
                                                           prop2_const,
                                                           input_props[1][1])
            # Update state with the generated ipair and specified values
            state.update(ipair[0], ipair[1], ipair[2])

            # Store the state in points dictionary
            self.points[name] = state
        except ValueError as e:
            print(f'Unable to set point: {e}')

    def get_point_coords(self, state: AbstractState, x_property: str, y_property: str) -> Tuple[float, float]:
        """Get coordinates for a point based on specified properties"""
        x = getattr(state, self.property_map[x_property.upper()])()
        y = getattr(state, self.property_map[y_property.upper()])()
        return x, y

    def add_line(self, name: str, start_point_name: str, end_point_name: str, process: str):
        """Add a process line between two existing points, with process constraints"""
        if start_point_name not in self.points or end_point_name not in self.points:
            raise ValueError("Both start and end points must exist in the diagram")

        start_point = self.points[start_point_name]
        end_point = self.points[end_point_name]

        # Define constant properties and second property for linspace
        if process == "isothermal":
            constant_property = "T"
            variable_property = "H"
        elif process == "isobaric":
            constant_property = "P"
            variable_property = "H"
        elif process == "isochoric":
            constant_property = "D"
            variable_property = "H"
        elif process == "isentropic":
            constant_property = "S"
            variable_property = "H"
        elif process == "isoenthalpic":
            constant_property = "H"
            variable_property = "P"
        elif process == "line":  # Simple line for custom values
            constant_property = "Noconstant"
            variable_property = "H"
        else:  # Simple line for custom values
            constant_property = "Noconstant"
            variable_property = "H"

        # Add the line with all process points
        self.lines[name] = Line(name,
                                start_point,
                                end_point,
                                constant_property,
                                variable_property,
                                self.fluid,
                                self.backend)

    def plot(self, fig=None, y_property: str = 'P', x_property: str = 'H', color: str ='red',
             show_saturation: bool = True):
        """Plot the diagram with specified properties on x and y axes using Plotly"""

        # Initialize Plotly figure
        if fig is None:
            fig = go.Figure()

        # Prepare lists to store the values of x and y properties from points and lines only
        x_values = []
        y_values = []

        # Plot saturation curves if requested, but exclude them from axis limit calculations
        if show_saturation:
            state = AbstractState(self.backend, self.fluid)

            # Generate temperature points for saturation curves
            T_crit = state.T_critical()
            T_triple = state.Ttriple()
            T_range = np.linspace(T_triple, T_crit, 1000)

            x_sat_liq = []
            y_sat_liq = []
            x_sat_vap = []
            y_sat_vap = []

            for T in T_range:
                try:
                    # Saturated liquid point
                    state.update(CoolProp.QT_INPUTS, 0, T)
                    x_sat_liq.append(getattr(state, self.property_map[x_property.upper()])())
                    y_sat_liq.append(getattr(state, self.property_map[y_property.upper()])())

                    # Saturated vapor point
                    state.update(CoolProp.QT_INPUTS, 1, T)
                    x_sat_vap.append(getattr(state, self.property_map[x_property.upper()])())
                    y_sat_vap.append(getattr(state, self.property_map[y_property.upper()])())
                except:
                    continue

            # Add saturated liquid and vapor curves as traces
            fig.add_trace(go.Scatter(x=x_sat_liq, y=y_sat_liq, mode='lines', name='Saturated liquid',
                                     line=dict(dash='dash', color=color)))
            fig.add_trace(go.Scatter(x=x_sat_vap, y=y_sat_vap, mode='lines', name='Saturated vapor',
                                     line=dict(dash='dash', color=color)))

        # Plot points with annotations and custom hover text
        for name, state in self.points.items():
            x, y = self.get_point_coords(state, x_property, y_property)
            x_values.append(x)
            y_values.append(y)
            hover_text = f"{name}<br>{x_property}: {x:.2f}<br>{y_property}: {y:.2f}"
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text', name=name,
                text=[name], textposition='top right',
                marker=dict(color=color, size=8),
                hovertext=hover_text,
                hoverinfo="text"
            ))

            # Plot lines by using precomputed process points
        for name, line in self.lines.items():
            line_x = []
            line_y = []

            for state in line.process_points:
                x, y = self.get_point_coords(state, x_property, y_property)
                line_x.append(x)
                line_y.append(y)
                x_values.append(x)
                y_values.append(y)

            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name=f'{name} ({line.constant_property})',
                hoverinfo="x+y",
                hovertemplate=f"{x_property}: {{x:.2f}}<br>{y_property}: {{y:.2f}}",
                line=dict(color=color)  # Set line color to blue
            ))

            # Set axis limits based on the range of x_values and y_values
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        x_delta = min([(x_max - x_min) * 0.25, 0.1 * x_min])
        y_delta = min([(y_max - y_min) * 0.25, 0.1 * y_min])

        fig.update_xaxes(range=[x_min - x_delta, x_max + x_delta])
        fig.update_yaxes(range=[y_min - y_delta, y_max + y_delta])

        fig.update_layout(
            title=f'{y_property}-{x_property} PLOT',
            xaxis_title=f'{x_property} [SI units]',
            yaxis_title=f'{y_property} [SI units]',
            showlegend=True,
            template="plotly_white"
        )

        return fig

    def get_point_df(self):
        """Retrieves a DataFrame of all points with their thermodynamic properties, including units."""

        # Mapping of properties to units
        units_map = {
            'P': 'Pa',
            'T': 'K',
            'H': 'J/kg',
            'S': 'J/kg/K',
            'D': 'kg/m³',
            'Q': '-',
            'viscosity_dynamic': 'Pa·s',
            'conductivity': 'W/m/K',
            'speed_of_sound': 'm/s',
            'Prandtl': '-',
            'cp': 'J/kg/K',
            'cv': 'J/kg/K',
            'surface_tension': 'N/m',
            'internal_energy': 'J/kg',
            'Gibbs_mass': 'J/kg',
            'Helmholtz_mass': 'J/kg',
            'molar_mass': 'kg/mol',
            'Hmolar': 'J/mol',
            'Smolar': 'J/mol/K',
            'Dmolar': 'mol/m³',
            'cp_molar': 'J/mol/K',
            'cv_molar': 'J/mol/K'
        }

        def get_point_properties(point):
            """Retrieve properties for a single point with units in column names."""
            properties = {}
            for display_name, attr in self.property_map.items():
                unit = units_map.get(display_name, "")
                col_name = f"{display_name} [{unit}]" if unit else display_name
                try:
                    if callable(attr):
                        # If the map entry is a function, call it with the state
                        properties[col_name] = attr(point)
                    else:
                        # Otherwise, use the attribute name to get the value
                        properties[col_name] = getattr(point, attr)()
                except:
                    properties[col_name] = np.nan
            return properties

        # Build a dictionary of all point properties
        points_data = {
            name: get_point_properties(point) for name, point in self.points.items()
        }

        # Convert to a DataFrame
        return pd.DataFrame.from_dict(points_data, orient='index')

    def calculate_reversed_cycle(self, T_evaporation: float, T_condensation: float,
                 Superheat: float, Pump_Isentropic_Efficiency: float, Subcooling: float):
        """Calculate the four points of the reversed cycle and add them to the diagram."""

        p_evap = self.get_pressure_at_temperature(T_evaporation)
        p_cond = self.get_pressure_at_temperature(T_condensation)

        # Point 1: P = p at T_evaporation
        self.add_point("Point_1", **{"T": T_evaporation + max(Superheat,1e-3), "P": p_evap})
        # Point 2: Intermediate point (Point_2_start) is calculated with P = p at T_condensation, s = s at point 1
        point_1 = self.points["Point_1"]
        s1 = point_1.smass()  # Entropy at Point 1
        self.add_point("Point_2", **{"S": s1, "P": p_cond})
        point_2_iso = self.points["Point_2"]

        # Calculate h_2 using the pump isentropic efficiency
        h1 = point_1.hmass()  # Enthalpy at Point 1
        h2_iso = point_2_iso.hmass()  # The enthalpy at Point 2 if isentropic
        # Solve for h_2 based on the pump isentropic efficiency
        h2 = h1 + (h2_iso - h1) /Pump_Isentropic_Efficiency
        self.add_point("Point_2", **{"P": p_cond, "H":h2})

        # Point 3: p = p2, T = T_condensation - Subcooling
        point_2 = self.points["Point_2"]
        T3 = T_condensation - Subcooling
        self.add_point("Point_3", **{"P": point_2.p(), "T": T3})

        # Point 4: h = h3, p = p1
        point_3 = self.points["Point_3"]
        self.add_point("Point_4", **{"P": p_evap, "H": point_3.hmass()})

        # Add lines representing the processes
        self.add_line("Line_1-2", "Point_1", "Point_2", process="line")
        self.add_line("Line_2-3", "Point_2", "Point_3", process="isobaric")
        self.add_line("Line_3-4", "Point_3", "Point_4", process="isoenthalpic")
        self.add_line("Line_4-1", "Point_4", "Point_1", process="isobaric")