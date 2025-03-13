# utils.py

import streamlit as st
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import pandas as pd

def connect_to_search_console(access_token, refresh_token, client_id, client_secret, scopes):
    try:
        creds = Credentials(token=access_token,
                            refresh_token=refresh_token,
                            token_uri='https://oauth2.googleapis.com/token',
                            client_id=client_id,
                            client_secret=client_secret,
                            scopes=scopes)

        service = build('searchconsole', 'v1', credentials=creds)
        print("Successfully connected to Google Search Console API.")
        return service

    except Exception as e:
        print(f"An error occurred during connection: {e}")
        return None

def load_gsc_query_data(
    service, site_url: str, start_date: str, end_date: str,
    dimensions: list = ["query"], search_type: str = "web"
) -> pd.DataFrame:
    """
    Retrieves query-level data from Google Search Console for a specified site,
    date range, and dimension set. Returns a DataFrame with columns:
      ['Query', 'Clicks', 'Impressions', 'CTR', 'Position'].
    """
    query_data = []
    request_body = {
        'startDate': start_date,
        'endDate': end_date,
        'dimensions': dimensions,
        'searchType': search_type
    }

    try:
        while True:
            response = service.searchanalytics().query(
                siteUrl=site_url, body=request_body
            ).execute()

            if 'rows' in response:
                for row in response['rows']:
                    query_val = row['keys'][0]  # Because 'dimensions' = ['query']
                    clicks = row['clicks']
                    impressions = row['impressions']
                    ctr = row['ctr']
                    position = row['position']
                    query_data.append({
                        'Query': query_val,
                        'Clicks': clicks,
                        'Impressions': impressions,
                        'CTR': ctr,
                        'Position': position
                    })
            else:
                st.info("No query data found for the selected range.")
                break

            # Paginate if nextPageToken exists
            if 'nextPageToken' in response:
                request_body['pageToken'] = response['nextPageToken']
            else:
                break

        if query_data:
            return pd.DataFrame(query_data)
        else:
            return pd.DataFrame(columns=['Query', 'Clicks', 'Impressions', 'CTR', 'Position'])

    except Exception as e:
        st.error(f"Error fetching query data: {e}")
        return pd.DataFrame(columns=['Query', 'Clicks', 'Impressions', 'CTR', 'Position'])




def load_gsc_query_data_alt(
    service, site_url: str, start_date: str, end_date: str, page_add: str,
    dimensions: list = ["query"], search_type: str = "web"
) -> pd.DataFrame:
    """
    Retrieves query-level data from Google Search Console for a specified site,
    date range, and dimension set. Returns a DataFrame with columns:
      ['Query', 'Clicks', 'Impressions', 'CTR', 'Position'].
    """
    query_data = []
    request_body = {
        'startDate': start_date,
        'endDate': end_date,
        'dimensions': dimensions,
        'searchType': search_type,
        "dimensionFilterGroups": [
            {
                "groupType": "and",
                "filters": [
                    {
                        "dimension": "page",
                        "operator": "contains",
                        "expression": page_add
                    }
                ]
            }
        ]
    }

    try:
        while True:
            response = service.searchanalytics().query(
                siteUrl=site_url, body=request_body
            ).execute()

            if 'rows' in response:
                for row in response['rows']:
                    query_val = row['keys'][0]  # Because 'dimensions' = ['query']
                    clicks = row['clicks']
                    impressions = row['impressions']
                    ctr = row['ctr']
                    position = row['position']
                    query_data.append({
                        'Query': query_val,
                        'Clicks': clicks,
                        'Impressions': impressions,
                        'CTR': ctr,
                        'Position': position
                    })
            else:
                st.info("No query data found for the selected range.")
                break

            # Paginate if nextPageToken exists
            if 'nextPageToken' in response:
                request_body['pageToken'] = response['nextPageToken']
            else:
                break

        if query_data:
            return pd.DataFrame(query_data)
        else:
            return pd.DataFrame(columns=['Query', 'Clicks', 'Impressions', 'CTR', 'Position'])

    except Exception as e:
        st.error(f"Error fetching query data: {e}")
        return pd.DataFrame(columns=['Query', 'Clicks', 'Impressions', 'CTR', 'Position'])
