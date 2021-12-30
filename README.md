The sole function of the problem is to allow the user to determine the mass and width of a Z_0 boson.
User is prompted to provide multiple files containing experimental results in the format of Energy(GeV), Cross section (nb) and Cross section uncertainity(nb) as per industry standards. 
The user can change within the script, the properties of the file, a csv file with no headers and comments is set as default.
The script prompts the user to choose the decay product from a range of possible products stored within the script as a dicionary.

Scipy-Curvefit is used to fit the data to the desired number of parameters, various statistical methods are used to preprocess the data for cleaning.
A 2D or 3D contour plot is stored within the directory of the script and the fitting results (with uncertainities) is outputted.
