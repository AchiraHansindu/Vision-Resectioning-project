import pandas as pd

# Replace 'your_file.h5' with your actual file path
df = pd.read_hdf('interlaken_00_c_events_left/rectify_map.h5')

# Display the contents of the HDF5 file
print(df.head())
