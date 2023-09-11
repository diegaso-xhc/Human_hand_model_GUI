# Human hand model GUI

## Overview of the repository
<div align="justify">
Human hands are crucial for the manipulation dexterity that humans exhibit on a daily basis. This is because of their high performance and remarkable flexibility. Understanding how humans work is crucial for a myriad of fields, including robotics, rehabilitation and biomechanics.
Sharing state-of-the-art digital hand models with the curious minds of the world would surely expand our knowledge of these marvelous instruments we have. 

This repository uses one of such state-of-the-art models, namely, the MANO hand model. 
<br />
<br /> 
<p align="center">
   <img src="/Visualizations/Hand_1.png" width="600" />
</p>
<sup> *Romero, Javier, Dimitrios Tzionas, and Michael J. Black. "Embodied hands: Modeling and capturing hands and bodies together." ACM Transactions on Graphics (ToG) 36.6 (2017): 1-17.</sup>

## Understanding repository

The repository contains two files. One of them contains a GUI which can be used to move the 45 degrees of freedom of the hand (3 rotations per each joint), as well as adjust the shape of the hand model via a shape vector (size of 10). The other file can depicts the hand in a flat configuration for visualization purposes. Both files export the meshes of the resulting hands in a given configuration. The meshes are saved in .mat files which can be imported via MATLAB or python. 
<br />

### Hand model GUI
<p align="center">
   <img src="/Visualizations/GUI.png" width="700" />
</p>
<br />

<br />
<strong>Go ahead and explore the functionalities of this repository!</strong>
<br />

</div>

## Contributions

The contributions of this repository can be summarized as follows:

```
- A GUI capable of manipulating the MANO hand model and give the user a feeling of the utlities of the model.
- Exporting functions which save the mesh models (vertices and faces) of the hand model in a given configuration. Visualizations can be customized.
```

## License

Developed by Diego Hidalgo C. (2023). This repository is intended for research purposes only. If you wish to use any parts of the provided code for commercial purposes, please contact the author at hidalgocdiego@gmail.com.
