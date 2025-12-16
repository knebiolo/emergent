import pandas as pd
import tempfile
import os

from emergent.fish_passage.reporting import output_excel


def test_output_excel_writes_file(tmp_path):
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    records = {'gen1': df1}
    out = output_excel(records, str(tmp_path), 'testmodel')
    assert os.path.exists(out)
    # Accept either an xlsx file (if openpyxl is installed)
    # or a fallback directory containing CSV files.
    if out.endswith('.xlsx'):
        assert os.path.isfile(out)
    else:
        assert os.path.isdir(out)
        files = [f for f in os.listdir(out) if f.endswith('.csv')]
        assert len(files) > 0
