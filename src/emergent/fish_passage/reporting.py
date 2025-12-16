import os
from typing import Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def output_excel(records: Dict[str, pd.DataFrame], model_dir: str, model_name: str) -> str:
    """Export a mapping of name->DataFrame to an Excel workbook.

    Returns the path to the written file.
    """
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, f'output_{model_name}.xlsx')
    try:
        try:
            with pd.ExcelWriter(output_path) as writer:
                for sheet_name, df in records.items():
                    df.to_excel(writer, sheet_name=str(sheet_name))
            try:
                logger.info('records exported. check output excel file: %s', output_path)
            except Exception:
                pass
            return output_path
        except ModuleNotFoundError:
            # Excel writer backend not available (e.g., openpyxl). Fall back
            # to writing per-sheet CSV files into a folder.
            fallback_dir = os.path.join(model_dir, f'output_{model_name}_csvs')
            os.makedirs(fallback_dir, exist_ok=True)
            for sheet_name, df in records.items():
                safe_name = str(sheet_name).replace('/', '_')
                csv_path = os.path.join(fallback_dir, f'{safe_name}.csv')
                df.to_csv(csv_path, index=False)
            try:
                logger.info('Excel backend missing; wrote CSVs to %s', fallback_dir)
            except Exception:
                pass
            return fallback_dir
    except Exception:
        logger.exception('Failed to export records to excel or CSV fallback')
        raise
