import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Cache for loading data
@st.cache_data
def load_data(file_path):
    """Load the CSV file and preprocess numeric columns."""
    data = pd.read_csv(file_path)

    # Columns that may contain non-numeric symbols
    numeric_columns = ['price', 'area', 'total_rooms', 'age', 'value_per_sqft']
    for col in numeric_columns:
        if col in data.columns:
            # Convert to string and clean non-numeric characters
            data[col] = data[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
            data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, invalid values become NaN

    return data

# Cache for training the model
@st.cache_resource
def train_model(X_train, y_train):
    """Train the RandomForest model."""
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# App title and description
st.title("House Price Prediction App")
st.markdown("""
Upload a dataset to train a house price prediction model.  
Explore interactive visualizations and predict prices based on input features.
""")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load data
    data = load_data(uploaded_file)

    # Ensure the dataset has a 'price' column
    if 'price' not in data.columns:
        st.error("The dataset must contain a 'price' column.")
    else:
        st.write("### Dataset Preview")
        st.dataframe(data)

        # Drop rows with missing or non-numeric values in essential columns
        data = data.dropna(subset=['price'])

        # Region Filter Section
        st.sidebar.header("Filters")

        # Filter by Region using dropdown
        if 'region_name' in data.columns:
            unique_regions = data['region_name'].dropna().unique()
            selected_region = st.sidebar.selectbox(
                "Select a Region", ['All Regions'] + list(unique_regions)
            )
            if selected_region != 'All Regions':
                data = data[data['region_name'] == selected_region]

        # Filter by House Type
        if 'house_type' in data.columns:
            unique_house_types = data['house_type'].dropna().unique()
            selected_house_type = st.sidebar.selectbox(
                "Select a House Type", ['All Types'] + list(unique_house_types)
            )
            if selected_house_type != 'All Types':
                data = data[data['house_type'] == selected_house_type]

        # Locality Comparison Tool using Selectbox
        st.write("### Property Comparison Tool by Locality")
        if 'locality_name' in data.columns:
            # Allow user to select a locality for comparison using selectbox
            selected_locality = st.selectbox(
                "Select a Locality to Compare", data['locality_name'].unique()
            )

            # Filter the data based on selected locality
            filtered_data = data[data['locality_name'] == selected_locality]

            if not filtered_data.empty:
                # Display the properties for comparison
                st.write(f"### Comparison of Properties in {selected_locality}")
                st.dataframe(filtered_data)

                # Data Visualization: Price Distribution by Locality
                st.write("#### Price Distribution in Selected Locality")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(filtered_data['price'], bins=30, kde=True, ax=ax, color='blue')
                ax.set_title(f"Price Distribution in {selected_locality}")
                ax.set_xlabel("Price")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

                # Data Visualization: Price vs Area
                st.write("#### Price vs Area (Scatter Plot)")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(filtered_data['area'], filtered_data['price'], alpha=0.7, c='orange')
                ax.set_title(f"Price vs Area for {selected_locality}")
                ax.set_xlabel("Area (sq.ft.)")
                ax.set_ylabel("Price")
                st.pyplot(fig)

            else:
                st.info(f"No properties found for the selected locality: {selected_locality}.")

        # Display Filtered Data
        st.write("### Filtered Data")
        st.dataframe(data)

        # Preprocessing for Model Training
        data = data.dropna()  # Drop any rows with missing values

        # Ensure all numeric columns are cleaned and converted to valid numbers
        numeric_columns = ['price', 'area', 'total_rooms', 'age', 'value_per_sqft']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')  # Converts invalid entries to NaN

        # Fill any remaining NaN values with the mean (or another method you prefer)
        data = data.fillna(data.mean())

        # Convert categorical columns to numerical using one-hot encoding
        categorical_cols = [
            'locality_name', 'region_name', 'construction_status', 'house_type', 'new_resale'
        ]
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # Split data into features and target
        X = data.drop('price', axis=1)
        y = data['price']

        if len(data) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            st.warning("Dataset is too small to split. Using the entire dataset for both training and testing.")
            X_train, X_test, y_train, y_test = X, X, y, y

        # Train the model
        if not X_train.empty and not y_train.empty:
            model = train_model(X_train, y_train)

            # Evaluate the model
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.metric(label="Model Mean Squared Error", value=f"{mse:.2f}")

            # Sidebar for user input
            st.sidebar.header("Input Features")
            user_input = {}

            for col in X.columns:
                if col in X.select_dtypes(include=['uint8']).columns:  # Categorical
                    user_input[col] = st.sidebar.selectbox(col, [0, 1])
                else:  # Numerical
                    user_input[col] = st.sidebar.number_input(col, value=float(X[col].mean()))

            user_data = pd.DataFrame(user_input, index=[0])

            # Prediction button
            if st.sidebar.button("Predict Price"):
                prediction = model.predict(user_data)
                st.write("### Predicted House Price")
                st.success(f"${prediction[0]:,.2f}")
        else:
            st.error("Insufficient data to train the model.")
