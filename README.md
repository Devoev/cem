This repo contains the code for the TU Darmstadt lecture [04-10-0611-vu Computational Electromagnetics](https://www.tucan.tu-darmstadt.de/scripts/mgrqispi.dll?APPNAME=CampusNet&PRGNAME=COURSEDETAILS&ARGUMENTS=-N339515486233916,-N000274,-N384332886947432,-N386713054610324,-N386713065523433,-N0,-N0).

## Project structure

### FEM code
The simulations in this project are done using a [finite element simulation](https://en.wikipedia.org/wiki/Finite_element_method). 
Functions for matrix assembly, basis function evaluation and a data structure for a 2D triangular mesh are found in the `fem/` folder.

### Capacitor model
The file for an electrostatic simulation of a parallel-plate capacitor. The main script and files for the mesh generation are found under `capacitor/` 

![capacitor](https://github.com/Devoev/cem/assets/28957846/ef502040-8e1a-4eee-af92-db2d87f45278)

### Examples
Other example scripts are found in the `examples/` folder.
