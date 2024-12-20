import os
from google.ads.googleads.client import GoogleAdsClient
from google.oauth2 import service_account
import altair as alt
import pandas as pd

# Replace with the path to your service account JSON key file
key_file_path = 'nmw-t-1-e01bb49718d1.json'
developer_token = "p2Of8yLD6yKNWn7NrtlR3g"
login_customer_id = "8882181823"  # Your Google Ads Customer ID (MCC if applicable)

try:
    # Load credentials from the service account JSON key file
    credentials = service_account.Credentials.from_service_account_file(key_file_path)

    # Configuration dictionary for Google Ads API client
    config = {
        "developer_token": developer_token,
        "use_proto_plus": True,
        "json_key_file_path": key_file_path,
    }
    if login_customer_id:
        config["login_customer_id"] = login_customer_id

    # Initialize the Google Ads client
    client = GoogleAdsClient.load_from_dict(config_dict=config)
    print("Client successfully initialized")

    # The services we need
    keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
    google_ads_service = client.get_service("GoogleAdsService")

    # Define your seed keyword
    seed_keyword = input("Enter your seed keyword: ")

    # Define location and language targeting based on the official snippet
    location_ids = ["1023191"]  # New York, NY as an example
    language_id = "1000"  # English

    # Function to map location IDs to resource names
    geo_target_constant_service = client.get_service("GeoTargetConstantService")
    def map_locations_ids_to_resource_names(location_ids):
        return [
            geo_target_constant_service.geo_target_constant_path(location_id)
            for location_id in location_ids
        ]

    location_rns = map_locations_ids_to_resource_names(location_ids)
    language_rn = google_ads_service.language_constant_path(language_id)

    # Prepare the GenerateKeywordIdeasRequest
    request = client.get_type("GenerateKeywordIdeasRequest")
    request.customer_id = login_customer_id
    request.language = language_rn
    request.geo_target_constants.extend(location_rns)
    request.include_adult_keywords = False
    # Use the GOOGLE_SEARCH_AND_PARTNERS network as in the official snippet
    request.keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS

    # Set the seed keyword
    request.keyword_seed.keywords.append(seed_keyword)

    # Generate keyword ideas
    keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(request=request)
    keyword_data = []

    for idea in keyword_ideas:
        competition_value = idea.keyword_idea_metrics.competition.name
        avg_monthly_searches = idea.keyword_idea_metrics.avg_monthly_searches or 0
        keyword_data.append({
            "Keyword Text": idea.text,
            "Average Monthly Searches": avg_monthly_searches,
            "Competition": competition_value
        })

    # Create a DataFrame
    df = pd.DataFrame(keyword_data)

    # Save the DataFrame to a CSV file
    csv_file_path = "keyword_ideas.csv"
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved at {csv_file_path}")

    # Display the first few rows of the DataFrame
    print(df)
    # prompt: Using dataframe df: plot the number of words in each "Keyword Text" value against the number of average monthly searches 

    # Import necessary libraries

    # Calculate the number of words in each "Keyword Text" value
    df['word_count'] = df['Keyword Text'].apply(lambda x: len(x.split()))

    # Create the scatter plot
    chart = alt.Chart(df).mark_point().encode(
        x='word_count',
        y='Average Monthly Searches',
        tooltip=['Keyword Text', 'Average Monthly Searches', 'word_count']
    )

    # Display the chart
    chart.show()

except Exception as e:
    print(f"An error occurred: {e}")