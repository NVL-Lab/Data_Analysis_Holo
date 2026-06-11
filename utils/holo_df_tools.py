__author__ = 'Saul'

import pandas as pd

def get_suite2p_filters() -> dict:
    return {
        'session_date': '191001',
        'mice_name': ['NVI12']
    }

def get_nwb_filters() -> dict:
    return {
        'session_date': '190930',
        'mice_name': 'NVI13'
    }

def get_data_indexes(df_dir, purpose) -> list:
    df = pd.read_parquet(df_dir).reset_index(drop=True)
    if purpose == 'suite2p':
        filters = get_suite2p_filters()
    elif purpose == 'nwb':
        filters = get_nwb_filters()

    for label in filters:
        if isinstance(filters[label], list):
            df = df[df[label].isin(filters[label])]
        elif isinstance(filters[label], str):
            df = df[df[label].eq(filters[label])]
        else:
            raise TypeError('Column values are of incorrect type')

    
    df_flagged = df[df.filter(like='Flag').notna().any(axis=1)]
    
    if not df_flagged.empty:
        print('**The following datasets are flagged**')
        print(df_flagged)

    # Uncomment to not run flagged datasets
    #df = df[df.filter(like='Flag').isna().all(axis=1)]
    filtered_indexes = df.index
    print('**The following datasets will be ran**')
    print(df)
    print(filtered_indexes)

    return filtered_indexes
