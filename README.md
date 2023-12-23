# Introduction
Reimplementation of the slicing strategy introduced by [Wüthrich et al.](https://www.researchgate.net/publication/354726760_A_Novel_Slicing_Strategy_to_Print_Overhangs_without_Support_Material)



```bibtex
@Article{app11188760,
AUTHOR = {Wüthrich, Michael and Gubser, Maurus and Elspass, Wilfried J. and Jaeger, Christian},
TITLE = {A Novel Slicing Strategy to Print Overhangs without Support Material},
JOURNAL = {Applied Sciences},
VOLUME = {11},
YEAR = {2021},
NUMBER = {18},
ARTICLE-NUMBER = {8760},
URL = {https://www.mdpi.com/2076-3417/11/18/8760},
ISSN = {2076-3417},
ABSTRACT = {Fused deposition modeling (FDM) 3D printers commonly need support material to print overhangs. A previously developed 4-axis printing process based on an orthogonal kinematic, an additional rotational axis around the z-axis and a 45° tilted nozzle can print overhangs up to 100° without support material. With this approach, the layers are in a conical shape and no longer parallel to the printing plane; therefore, a new slicer strategy is necessary to generate the paths. This paper describes a slicing algorithm compatible with this 4-axis printing kinematics. The presented slicing strategy is a combination of a geometrical transformation with a conventional slicing software and has three basic steps: Transformation of the geometry in the .STL file, path generation with a conventional slicer and back transformation of the G-code. A comparison of conventionally manufactured parts and parts produced with the new process shows the feasibility and initial results in terms of surface quality and dimensional accuracy.},
DOI = {10.3390/app11188760}
}
```
# Getting Started

You will need:
- PrusaSlicer (currently this is the only supported slicer).
- Python with the packages listed in `requirements.txt`:
  - vtkplotlib~=2.1.0
  - tqdm~=4.66.1
  - numpy~=1.26.2

The script should work cross-platform but was only tested on Windows.


## Setup Python
To setup your python environment you can either set up a virtual environment 
(my personal preference), or use your system interpreter:

To use a virtual environment:
1. Create a virtual environment `python -m venv venv` (the second `venv` is the name of the environment; you can choose your own)
2. Activate the virtual environment (`venv/Scripts/activate`).
3. Install the required python modules: `python -m pip install -r requirements.txt`
4. Make sure everything is OK by calling `python conical_slice.py --help`. If all modules are installed, you should see the help for the CLI.

To use your system python, do steps 3 and 4.

# How to use the script

First things first, if you're using a venv, make sure it's activated, or you use the python executable in your `<venv>` 
folder (usually `Scripts/python.exe`).

You will also need a PrusaSlicer Config file. To do this, open PrusaSlicer, select your printer, filament and layer height as normal. Then export by File > Export > Export Config (or CTRL+E).

To see the available options, call `python conical_slice.py --help`; that's the source of truth for the command line interface.

Example usage:

```shell
python conical_slice.py \
  -i tree.stl \
  -o my_gcode.gcode \
  --prusa-slicer-config config.ini \
  --cone-origin 0 0 \
  --angle 10 \
  --preview
```