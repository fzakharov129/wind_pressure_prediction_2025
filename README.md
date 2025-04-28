The development of this AI model is of a research nature.
The research team aims to create a model capable of accurately predicting the wind pressure coefficient on high-rise building façades.
Although the topic of wind pressure prediction for buildings has been extensively studied, our research group has its own vision of the model architecture that we intend to implement.

At the current stage:

A baseline architecture based on MLP (Multilayer Perceptron) is being developed, which will serve as a starting point for further transition to more advanced and modern architectures.

About the Data
The model is trained on a proprietary dataset based on the results of wind tunnel tests of building models:

Building Models:

Both the primary building and the interfering building are made of ABS plastic.

Model dimensions: 30.48 × 30.48 × 182.88 cm.

Measurement Points:

Pressure was measured at uniformly distributed points on the building façades.

Test Conditions:

Measurements were taken at wind attack angles ranging from 0° to 180° with 10° increments.

Configurations:

32 different placements of the interfering building relative to the primary building were considered.

Data Preparation
The final base dataset has been formed and saved in the file windloading_interference_base.csv.

It is used for model training and testing.

Generation of Pressure Distribution Maps:

Visualizations of the mean pressure field (Mean) and standard deviation (StdDev) were created.

The maps were saved in the directory results/figures, with façade boundaries highlighted for clarity.

Feature and Target Description:

Target variable: Pressure coefficient at a point (Mean).

Features:

Coordinates of the interfering building (X_int, Y_int);

Coordinates of the measurement points on the façade (X_fac, Y_fac);

Wind direction angle (Ang).


To run the project, navigate to the root folder wind_pressure_prediction_2025, activate the virtual environment (source environment/linux_wpp/bin/activate for Linux or source environment/windows_wpp/Scripts/activate for Windows), install the dependencies from the requirements.txt file, and then execute the command python3 -m src.training.train_mlp. Upon execution, the project will automatically load the dataset, train the MLP model based on the specified parameters, save the model weights, training configuration, loss log file, and evaluation metrics on the test set, and also display result graphs.