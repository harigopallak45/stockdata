import pandas as pd
import logging

def get_company_info(file_path, filename):
    try:
        # Load the CSV file with additional options
        df = pd.read_csv(file_path, delimiter=',', quotechar='"', on_bad_lines='skip')
        
        # Check for required columns
        if 'filename' not in df.columns or 'company_name' not in df.columns or 'description' not in df.columns:
            return {'error': 'Required columns not found in the dataset.'}

        # Extract company information based on the filename
        company_info = df[df['filename'] == filename]

        if not company_info.empty:
            company_info_dict = company_info.iloc[0].to_dict()
            return {
                'company_name': company_info_dict.get('company_name', 'Unknown'),
                'description': company_info_dict.get('description', 'No Description'),
                'Ticker': company_info_dict.get('filename', 'No Ticker')  # Ensure you have a ticker or filename
            }
        else:
            return {'error': 'Company information not found for the provided file.'}

    except FileNotFoundError:
        return {'error': f'File not found: {file_path}'}
    except pd.errors.EmptyDataError:
        return {'error': 'No data in the file.'}
    except pd.errors.ParserError as e:
        logging.error(f'Parser error: {e}')
        return {'error': f'Error parsing the file: {e}'}
    except Exception as e:
        logging.error(f'Error: {e}')
        return {'error': str(e)}
