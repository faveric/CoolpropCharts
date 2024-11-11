# CoolpropCharts
Streamlit application which uses Coolprop to calculate Thermodynamic properties and plot Thermodynamic Charts

## Overview
This app allows users to visualize thermodynamic cycles by plotting specific points and connecting them with thermodynamic process lines. Additionally, users can generate diagrams for both standard and reversed thermodynamic cycles.

## Key Features:
- Fluid Selection
- Diagram Creation
- Point Addition with Two Independent Properties
- Line Addition with Specified Process Types
- Alternate Mode: Reversed Thermodynamic Cycle
- Plot one or more diagrams in Thermodynamic Charts

## Usage Instructions
1. Select a Fluid
Begin by choosing a working fluid from the provided list, which is powered by CoolProp. This fluid will be used to calculate thermodynamic properties for all points and lines on the chart.

2. Create a New Diagram
Click on "Create New Diagram" to start a new chart. This will clear any previously plotted points or lines and prepare the workspace for a fresh cycle visualization.

3. Add Points with Two Independent Properties
To add a thermodynamic point, specify two independent properties (such as temperature and pressure) in the input fields. The app will compute the remaining properties for that point based on the selected fluid and display the calculated state in the chart.

4. Add Lines with Specified Thermodynamic Processes
You can connect two points by selecting them and choosing the type of thermodynamic process to represent (e.g., isothermal, isobaric, isentropic). This step adds clarity by visually distinguishing different processes within the cycle.

5. Reversed Thermodynamic Cycle Mode
This mode allows you to define a reversed thermodynamic cycle (e.g., refrigeration or heat pump cycle) by inputting the cycle parameters directly:

- Evaporation Temperature: Temperature at the evaporator.
- Condensation Temperature: Temperature at the condenser.
- Superheat: Degree of superheat at the evaporator outlet.
- Subcooling: Degree of subcooling at the condenser outlet.
- Pump Isentropic Efficiency: Efficiency of the pump in the cycle.

When you activate this mode, the app automatically generates all necessary points and process lines for a complete reversed cycle on the diagram.

6. Plot
- Select x and y axis properties for the chart
- Select one or more diagrams to plot
- Plot, control plot pan&zoom and save plots as png images

