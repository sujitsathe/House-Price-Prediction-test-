# Check if the uploaded dataset is valid
if uploaded_file:
    # Load the data
    data = load_data(uploaded_file)

    # Verify the required 'price' column exists
    if 'price' not in data.columns:
        st.error("The uploaded dataset must contain a 'price' column. Please upload a valid dataset.")
    else:
        # Ensure numeric columns are correctly processed
        numeric_columns = ['price', 'area', 'total_rooms', 'age', 'value_per_sqft']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Drop rows with missing 'price' values
        data = data.dropna(subset=['price'])

        # Check if the dataset is empty after preprocessing
        if data.empty:
            st.error("The dataset contains no valid rows after processing. Please check your data.")
        else:
            # Proceed with the app logic
            st.write("### Dataset Preview")
            st.dataframe(data)

            # Add filters, visualizations, and modeling logic here as per the original code
