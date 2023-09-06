# Human hand model GUI

## Overview of the repostitory
<div align="justify">
Human hands are crucial for the manipulation dexterity that humans exhibit on a daily basis. This is because of their high performance and remarkable flexibility. Understanding how humans work is crucial for a myriad of fields, including robotics, rehabilitation and biomechanics.
Sharing state-of-the-art digital hand models with the curious minds of the world would surely expand our knowledge of these marvelous instruments we have. 

This repository uses one of such state-of-the-art models, namely, the MANO hand model. 
<br />
<br /> 
<p align="center">
   <img src="/Visualizations/Hand_1.png" width="500" />
</p>
<sup> *Romero, Javier, Dimitrios Tzionas, and Michael J. Black. "Embodied hands: Modeling and capturing hands and bodies together." ACM Transactions on Graphics (ToG) 36.6 (2017): 1-17.</sup>
## Understanding repository

The reository contains two files, namely:
```
- toolkits: Python file containing all classes and controllers to connect, use, and control the robotic hand.
- main: A main file containing samples on how to use the functions on toolkits.
```
The classes are written in a way that facilitates the connection with the hand. The repository also reduces the complexity of handling bytes transmission from and to the hand. This allows the user to focus directly on high level controllers and experiment within different applications.
<br />
Whenever using position, current or force controllers, you will be able to see and extract the error and response on each one of the fingers you desired to control. The following images correspond to the position error and response signals of the motion on the gif above:
<br />


### Position error vs. Time
<p align="center">
   <img src="/Visualizations/GUI.png" width="700" />
</p>
<br />


### Position error vs. Time
<p align="center">
   <img src="/Visualizations/Response_response.png" width="750" />
</p>

<br />
<strong>Go ahead and explore the functionalities of this repository!</strong>
<br />

</div>

## Contributions

The contributions of this repository can be summarized as follows:

```
- A class that facilitates the connection via serial port to a device.
- Classes that handle the byte transmission to and from the robotic hand.
- Most of the functionalities explained in the manual of the ih2 azzurra hand (you won't need to develop things on your own).
- Ready to use controllers for position, current and force (P,PI,PD,PID).
- Visualization functions for analysis of the controllers responses.
```

## License

Developed by Diego Hidalgo C. (2021). This repository is intended for research purposes only. If you wish to use any parts of the provided code for commercial purposes, please contact the author at hidalgocdiego@gmail.com.
