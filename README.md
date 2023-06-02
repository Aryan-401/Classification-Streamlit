# Classification Datasets

This code provides a Streamlit application for classifying different datasets using various classifiers. It allows the user to select a dataset, choose a classifier, and customize the classifier's parameters. The code then trains the selected classifier on the chosen dataset, evaluates its performance using several metrics, and visualizes the dataset using Principal Component Analysis (PCA).

## Dependencies

Make sure you have the following dependencies installed in your environment:

- numpy
- streamlit
- scikit-learn
- matplotlib

You can install them using pip:

```bash
pip install numpy streamlit scikit-learn matplotlib
```

## Usage

1. Run the code using the following command:

```bash
streamlit run main.py
```


2. The Streamlit application will open in your browser.

3. Use the sidebar to select a dataset and a classifier.

4. Adjust the parameters specific to the chosen classifier.

5. Explore the dataset visualization and the classifier's performance metrics displayed in a table.

6. The application also provides a plot of the dataset using PCA.

Note: The code assumes that the dataset files (e.g., Iris, Breast Cancer, Wine Dataset) are available in scikit-learn's datasets module. If any of the datasets are missing, make sure to install them separately.

## Additional Information

The code consists of the following main components:

- Loading and preparing the dataset: The `get_dataset` function retrieves the selected dataset, and the `graphing_dataset` function creates a scatter plot to visualize the dataset.

- Selecting and customizing the classifier: The `add_parameter_ui` function allows the user to customize the parameters of the selected classifier using the Streamlit sidebar. The `get_classifier` function returns the appropriate classifier based on the user's choices.

- Classification and performance evaluation: The code splits the dataset into training and testing sets using `train_test_split`, trains the selected classifier on the training data, and evaluates its performance using accuracy, precision, recall, and F1 score.

- Visualization using PCA: The code applies PCA to the dataset and creates a scatter plot to visualize the dataset in two dimensions.

Feel free to explore and modify the code according to your needs and preferences.
