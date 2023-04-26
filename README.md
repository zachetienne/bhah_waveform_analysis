Public release of BH@H data.

Simulation is the same as in

https://einsteintoolkit.org/gallery/bbh/index.html

but with BH@H and BH@H numerical grids. Also the initial data TwoPunctures resolution was increased
from 30x30x16 to 48x48x20, so that the numerical errors from the initial data become negligible
early in the simulation.

Warning: This is a preliminary release, primarily for community feedback. A couple of known issues
remain with the simulation, including most importantly attenuation in higher-order modes near merger
due to suboptimal grid structure in the wavezone. This issue and its solution (not implemented in
this particular simulation) are discussed around slide 17 of my April 2023 APS talk:

https://docs.google.com/presentation/d/1HUoF0EhO7FZG1ZJ-jLkYb4vNTioRWaQbZYi8GvsTOnM/edit?usp=sharing