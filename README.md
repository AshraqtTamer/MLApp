# Machine Learning App

This project is a web-based application for building machine learning models, supporting tasks like **Regression**, **Classification**, and **Clustering**. It allows users to upload datasets and select the type of model and task they want to perform, making it suitable for various data analysis needs.

## Features

- **Task Selection**: Users can choose from Regression, Classification, or Clustering tasks.
- **Dynamic Inputs**: Upload your own dataset in CSV format and select the appropriate target variable.
- **Model Options**: Choose between different machine learning models (Linear Regression, Random Forest, Logistic Regression, etc.).
- **Clustering Visualization**: Visualize clusters and centroids with interactive scatter plots.
- **Performance Metrics**: Display metrics such as Mean Squared Error (MSE), R² score, Accuracy, etc., based on the selected task.

## How It Works

### Input Fields:
1. **Upload Dataset**: Upload your dataset in CSV format.
2. **Select Task**: Choose from Regression, Classification, or Clustering.
3. **Target Variable**: For regression or classification, select the target column from the dataset.
4. **Model Selection**: Select a model for the task (e.g., Linear Regression, Random Forest).
5. **Clustering Parameters**: For clustering, specify the number of clusters.

### Preprocessing:
- **Handling Missing Data**: Optionally drop rows with missing values.
- **Feature and Target Separation**: Automatically separates features and target variable based on the task.

### Prediction:
- **Regression**: Fit a model (e.g., Linear Regression or Random Forest Regressor) and predict the target variable.
- **Classification**: Fit a classification model (e.g., Logistic Regression or Random Forest Classifier) and predict the class labels.
- **Clustering**: Perform K-Means clustering and visualize the clusters along with centroids.

### Results:
- **Regression**: Displays Mean Squared Error (MSE) and R² score.
- **Classification**: Displays Accuracy score.
- **Clustering**: Visualizes clustered data with a scatter plot and centroids.

## Prerequisites

- Python 3.7+
- Libraries:
  - `streamlit`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/yield-prediction-app.git
   cd yield-prediction-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your dataset in the project folder. It should be in CSV format with numerical or categorical features and one target variable column for regression or classification tasks.
   
## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the URL displayed in the terminal to access the app.

3. Upload your dataset and select the task to get started.

4. Perform the task:

- **For Regression**, select a regression model and view results such as MSE and R² score.
- **For Classification**, choose a classification model and see the accuracy score.
- **For Clustering**, specify the number of clusters and visualize the data and centroids

## Dataset

This app allows you to upload your own dataset in CSV format. The dataset should contain numerical or categorical features, with one column designated as the target variable for regression or classification tasks.

## File Structure

- `app.py`: Main Streamlit application file.
- `requirements.txt`: A file that lists all the Python packages and their versions needed to run the project.
- `Classified Data.csv`: Dataset used classification.
- `Mall_Customersfinal.csv`: Dataset used for Clustering.
- `USA_HousingFinal.csv`: Dataset used for Regression.

## Future Improvements

- Support for additional machine learning models, including more advanced algorithms.
- Enhanced data visualization features for better insights and interactive graphs.
- Integration of model performance comparison tools for easier evaluation of multiple models.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- scikit-learn for machine learning models.
- streamlit for the interactive web application framework.
- matplotlib and seaborn for data visualization.
  
