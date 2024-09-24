import pandas as pd
import logging

def get_company_info(csv_path, filename, delimiter=',', quotechar='"'):
    try:
        # Load the CSV file while handling potential parsing errors
        company_info_df = pd.read_csv(csv_path, delimiter=delimiter, quotechar=quotechar)

        if 'filename' not in company_info_df.columns or 'company_name' not in company_info_df.columns:
            return {'error': "Required columns not found in the CSV file."}

        # Initialize default values
        company_name = 'Unknown'
        description = 'No Description'
        ticker = 'No Ticker'

        # Loop through each row to find the matching filename
        for _, row in company_info_df.iterrows():
            if row['filename'] == filename:
                company_name = row.get('company_name', 'Unknown')
                description = row.get('description', 'No Description')
                ticker = row.get('filename', 'No Ticker')
                break

        return {
            'company_name': company_name,
            'description': description,
            'Ticker': ticker
        }

    except FileNotFoundError:
        return {'error': f'File not found: {csv_path}'}
    except pd.errors.EmptyDataError:
        return {'error': 'No data in the file.'}
    except pd.errors.ParserError as e:
        logging.error(f'Parser error: {e}')
        return {'error': f'Error parsing the file: {e}'}
    except Exception as e:
        logging.error(f"Error retrieving company information: {e}")
        return {'error': str(e)}
