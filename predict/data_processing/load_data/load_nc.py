from netCDF4 import Dataset
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_nc_data(file_path: str):
    """
    Read data from NetCDF file and convert to Pandas DataFrame.
    
    :param file_path: Path to NetCDF file
    :return: List of DataFrames containing data for each variable. Returns empty list if file reading fails.
    """
    try:
        # Open NetCDF file
        nc_file = Dataset(file_path, 'r')
        
        # Print variable list
        logger.info(f"Variables in the netCDF file: {list(nc_file.variables)}")
        
        # Create a list to store DataFrames for each variable
        dataFrames = []
        
        # Iterate through variables, read data and convert to DataFrame
        for var_name in nc_file.variables:
            var_data = nc_file.variables[var_name][:]
            
            # Skip empty data
            if var_data.size == 0:
                continue
                
            # Process variable data and generate DataFrame
            if len(var_data.shape) == 1:
                df = pd.DataFrame(var_data, columns=[var_name])
                dataFrames.append(df)
            elif len(var_data.shape) == 2:
                df = pd.DataFrame(var_data, columns=[f'{var_name}_col_{i}' for i in range(var_data.shape[1])])
                dataFrames.append(df)
            elif len(var_data.shape) == 3:
                df = pd.DataFrame(var_data.reshape(-1, var_data.shape[-1]), columns=[f'{var_name}_dim_{i}' for i in range(var_data.shape[-1])])
                dataFrames.append(df)
        
        # Close file
        nc_file.close()
        
        # Return the first DataFrame if there's only one, otherwise concatenate all
        if not dataFrames:
            return pd.DataFrame()  # Return empty DataFrame instead of empty list
        elif len(dataFrames) == 1:
            return dataFrames[0]
        else:
            return pd.concat(dataFrames, axis=1)
            
    except OSError as e:
        logger.error(f"Cannot read NetCDF file {file_path}: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame instead of empty list
    except Exception as e:
        logger.error(f"Error processing NetCDF file {file_path}: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame instead of empty list