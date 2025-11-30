# Smart Spectral Analysis: Quantification of CO<sub>2</sub> and H<sub>2</sub>O

The repository contains code for a project I developed for the spectroscopy team at
University College Cork. The program analyses spectral data files produced by a Fourier
Transform Infrared Spectrometer and estimates the gas phase number densities of CO<sub>2</sub>
and H<sub>2</sub>O present in the sample chamber.

Spectral features' properties are estimated by fitting pseudo-Voigt profiles to them using
non-linear least squares and the L-BFGS-b algorithm. After that, they are classified by
matching peak centres with HITRAN data and their area is used to estimate concentration of
the gases. A document explaining the program's algorithm in detail can be provided upon
request.

The project is based on Python, and its code has been set up in a way that allows for easy
installation using Pip.

### Requirements to run and install the project

1. Python (version >= 3.12)
2. Pip (latest version)
3. An IDE like PyCharm to run (optional)
4. Additional dependencies are automatically installed with the project
5. Make sure "python" and "pip" are added to the PATH variable of your system after
   installation.

### Installing the project

There are two ways to install this project:

#### Directly install from GitHub without cloning

With this method, you would be installing the project directly into your default Python
packages folder. It's installed as the "FTIR-Gas-Density-Estimation" package. The
repository will
not be cloned into your system. If the code needs to be checked locally, you will need to
go to your Python packages folder.

Steps to install:

1. Execute this command in your terminal/command prompt:

   ```pip install git+https://github.com/lakshyasuri/FTIRGasDensityEstimation.git```
2. To update the package to the latest version, if needed later, run this command:

   ```pip install --upgrade git+https://github.com/lakshyasuri/FTIRGasDensityEstimation.git```

#### Clone the repository and then install

With this method, you would be cloning the repository into a directory of your choice and
then installing it as a package. It's installed "FTIR-Gas-Density-Estimation".

Steps to install:

1. Go to the directory where you want to clone the repository by using this command:

   ```cd your/directory/path ```
2. Run this command to clone this repository:

   ```git clone https://github.com/lakshyasuri/FTIRGasDensityEstimation.git```
3. Go inside the repository:

   ```cd path/to/repository ```
4. Install the project as a package using the command:

   ```pip install .```
5. To update the package to the latest version, if needed later, run this command from the
   repository's directory:

   ```git pull```
   ```pip install .```
6. If you want to update the code yourself and don't want to run the install command again
   and again, you can install it in developer mode using this command:

   ```pip install -e .```

### Running the project

#### Without an IDE

The installation directly adds the runner script - "run.py" - to your PATH variable. Which
means, you can run it from any location in your computer using the terminal/command
prompt. The command to run the project is:

```spectral-analyser "path/to/file/sample.dpt"```

Make sure that the file path is inside double quotes. The path is the first and compulsory
argument of the entry script. It can either be a directory or a file path. If it's a
directory, all the "*.dpt" files will processed by the program sequentially.

The program also has the following optional arguments which can be provided after the
mandatory file path/directory argument:

1. ```--compute_baseline```: Compute a new optimal baseline if this argument is used, else
   use the previous one. Use this option only when you see that the input spectrum has a
   visibly different shape than the previous input. It takes some time to calculate the
   best baseline and if you want to analyse multiple similar files, avoid computing the
   baseline more than once. After getting an estimate of the concentration, you can use
   this argument to get a more accurate result. For the first run of this project, a new
   optimal baseline will be calculated regardless of this argument.
2. ```--plot```: Argument to show plots. The program produces various plots for an input
   file to visually show various stages of its processing. Useful for manual analysis and
   result confirmation
3. ```--lbfgs```: Fit Pseudo-Voigt profile to absorption peaks using L-BFGS-b algorithm.
   If the argument is not provided, the default non-linear least squares algorithm is
   used. Usage is not really required as non-linear least squares routine performed better
   in almost every case during testing.

Here's few sample commands with optional arguments:

```spectral-analyser "/Users/lakshya/Downloads/Data_01082025/Spectra with humidification/I.0000.dpt --plot```

```spectral-analyser "/Users/lakshya/Downloads/Data_01082025/Spectra with humidification/I.0140.dpt --plot --compute_baseline```

#### With an IDE like PyCharm

Simply run the "run.py" script inside the "spectral_analyser" folder by providing the required and optional arguments using its "Edit Configurations" option.

