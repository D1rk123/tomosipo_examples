# tomosipo_examples
Examples using the Tomosipo library, made for the tutorial given at the CWI on 21-Jun-2021.

## Tutorial slides
The slides of the tutorial that was given at the CWI are included in the repository as Tomosipo_tutorial_slides.pdf.

## Examples
1. **2d_projection.py** Minimal example of how to use Tomosipo, showing how to project and backproject with a 2D parallel beam geometry.
2. **vector_geometry.py** Example showing how to make an unconventional CT setup with a spinning box moving along a very wide fan beam detector. It uses the vector geometry and transform functionality from Tomosipo, and the SIRT reconstruction algorithm from ts_algorithms. It also shows how to create animations from geometries.
3. **flexray_fdk.py** Example showing how to reconstruct data from the FleX-ray scanner at the CWI using the FDK reconstruction algorithm. It is designed not to make unnecessary copies to limit memory usage.

## Getting started
To make a Conda environment which should be able to run the example scripts, execute all lines in the **create_environment.txt** file. If you have an RTX30XX GPU, use the **create_environment_RTX30XX.txt** file instead.

## Related Repositories
- https://github.com/ahendriksen/tomosipo
- https://github.com/ahendriksen/ts_algorithms
