@echo off
set FILES=../RXMesh/input/sphere3.obj ../RXMesh/input/sphere5.obj 

set PYTHON_SCRIPT=smoothing_ev.py

for %%F in (%FILES%) do (
    echo Processing %%F
    python %PYTHON_SCRIPT% %%F
)