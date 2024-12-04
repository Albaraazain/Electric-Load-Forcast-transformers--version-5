import pandas as pd

def convert_to_hourly(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file, parse_dates=['utc_timestamp'])
    
    # Set timestamp as index
    df.set_index('utc_timestamp', inplace=True)
    
    # Resample to hourly frequency and calculate sum (not mean)
    hourly_df = df.resample('H').sum()
    
    # Reset index to make timestamp a column again
    hourly_df.reset_index(inplace=True)
    
    # Save to new CSV file
    hourly_df.to_csv(output_file, index=False)
    
    return hourly_df


# Usage
if __name__ == "__main__":
    input_file = "data/energy_consumption.csv"
    output_file = "hourly_data.csv"
    hourly_data = convert_to_hourly(input_file, output_file)
    print("First few rows of converted data:")
    print(hourly_data.head())