def to_snake_case(column_name):
    # Convert to lowercase
    lowercase_name = column_name.lower()
    # Replace spaces and special characters with underscores
    snake_case_name = lowercase_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    return snake_case_name

# Apply the function to all column names
new_columns = [to_snake_case(col) for col in df0.columns]

# Rename columns in the DataFrame
df0.columns = new_columns
