import dataframe_sessions

if __name__ == '__main__':
    sessions, issues = dataframe_sessions.get_sessions()
    
    sessions.to_csv('holobmi_df.csv', index=False)
    issues.to_csv('holobmi_df_issues.csv', index=False)

    sessions.to_parquet('holobmi_df.parquet', index=False)
    issues.to_parquet('holobmi_df_issues.parquet', index=False)
    print('dfs downloaded')
