import plotly.express as px
from bertopic import BERTopic
import pandas as pd
import os

# --- Configuration ---
# IMPORTANT: Update this path to your Excel file
EXCEL_FILE_PATH = 'Data for Test Task_ML Engineer_Data Science UA (1) (1).xlsx'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODELS_DIR = "bertopic_models"
INTL_NEWS_FOLDERS = "bertopic_model_eng"
UKR_NEWS_FOLDERS = "bertopic_model_ukr"
os.makedirs(MODELS_DIR, exist_ok=True)

from preprocessing import (
columns_to_keep,
remove_exact_duplicates,
remove_short_and_uninformative,
filter_by_source_popularity,
prepare_text_and_metadata_for_bertopic
)

from bertopic_module import (
ukrainian_stop_words,
configure_and_train_bertopic)


# --- Helper Function for Normalized Visualization
def visualize_topics_over_time_normalized(model, topics_over_time_df, topic_names_map, title, top_n_topics=20, colors=None):
    """
    Visualizes the relative frequency of topics over time.
    Takes a topics_over_time DataFrame, a map of topic IDs to names, and other parameters.
    """
    normalized_df = topics_over_time_df.copy()

    # Exclude topic -1 (outliers) as it's usually not semantically meaningful for trends
    normalized_df = normalized_df[normalized_df['Topic'] != -1]

    total_frequency_per_timestamp = normalized_df.groupby("Timestamp")["Frequency"].sum()
    normalized_df["TotalFrequency"] = normalized_df["Timestamp"].map(total_frequency_per_timestamp)

    # Calculate relative frequencies (proportion of documents belonging to each topic per time bin)
    normalized_df["RelativeFrequency"] = normalized_df["Frequency"] / normalized_df["TotalFrequency"]

    # Apply the OpenAI-generated topic names to the DataFrame for better labels
    normalized_df['TopicName'] = normalized_df['Topic'].map(topic_names_map)

    # If top_n_topics is specified, filter the DataFrame to show only the most frequent topics
    if top_n_topics is not None:
        # Get the IDs of the top N topics based on their overall frequency (excluding -1)
        top_n_topic_ids = normalized_df.groupby('Topic')['Frequency'].sum().nlargest(top_n_topics).index.tolist()
        normalized_df = normalized_df[normalized_df['Topic'].isin(top_n_topic_ids)]
        # Re-map topic names in case filtering changed the set of topics present
        normalized_df['TopicName'] = normalized_df['Topic'].map(topic_names_map)

    fig = px.line(
        normalized_df,
        x="Timestamp",
        y="RelativeFrequency",
        color="TopicName", # Use the OpenAI-generated names for color legend
        line_group="Topic", # Group lines by Topic ID
        hover_name="TopicName", # Show topic name on hover
        title=title,
        labels={"RelativeFrequency": "Relative Frequency", "Timestamp": "Date", "TopicName": "Topic"},
        height=600,
        color_discrete_sequence=colors if colors else px.colors.qualitative.Dark24
    )

    # Update hover information to show both relative and absolute frequencies, and Topic ID
    fig.update_traces(
        mode="lines",
        hovertemplate="<b>Topic:</b> %{hovertext}<br><b>ID:</b> %{customdata[0]}<br><b>Date:</b> %{x}<br><b>Relative Frequency:</b> %{y:.2%}<br><b>Absolute Frequency:</b> %{customdata[1]}<extra></extra>",
        customdata=normalized_df[['Topic', 'Frequency']]
    )

    # Set Y-axis range from 0 to 1 (0% to 100%) for relative frequencies
    fig.update_yaxes(range=[0, 1])
    fig.show()
    return fig

# Helper function to unwrap list from OpenAI topic names
def unwrap_openai_name(name_entry):
    if isinstance(name_entry, list):
        if name_entry:
            return name_entry[0]
        else:
            return "Untitled/Empty Topic"
    return str(name_entry)

def main():
    print("--- Starting News Analysis ---")

    # 1. Load and Split Data
    print("Loading and splitting data...")
    full_excel_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=None) # Read all sheets into a dictionary of DataFrames

    if 'Україна' in full_excel_data and 'Міжнародні' in full_excel_data:
        ukr_df_raw = full_excel_data['Україна']
        intl_df_raw = full_excel_data['Міжнародні']
    else:
        # Fallback if sheet names are different, e.g., assume first sheet is Ukrainian, second is International
        print("Warning: 'Українські' or 'Міжнародні' sheets not found. Assuming first two sheets are Ukl/Intl respectively.")
        sheet_names = list(full_excel_data.keys())
        if len(sheet_names) >= 2:
            ukr_df_raw = full_excel_data[sheet_names[0]]
            intl_df_raw = full_excel_data[sheet_names[1]]
        else:
            raise ValueError("Not enough sheets found in the Excel file. Expected at least two.")

    # 2. Preprocess Data
    print("\n--- Preprocessing Ukrainian News ---")
    ua_df = ukr_df_raw[columns_to_keep].copy()
    ua_df = remove_exact_duplicates( ua_df )
    ua_df = remove_short_and_uninformative( ua_df, min_chars=100, min_words=10 )
    # Filtering sources: removing the least (bottom 20%) and most (top 5%) popular to reduce noise and prevent dominance.
    ua_df_filtered = filter_by_source_popularity( ua_df, lower_quantile=0.20, upper_quantile=0.95 )
    ukr_df_final = prepare_text_and_metadata_for_bertopic( ua_df_filtered.copy() )
    ukr_documents = ukr_df_final['document'].tolist()
    ukr_timestamps = ukr_df_final['Дата'].tolist()

    print("\n--- Preprocessing International News ---")
    intl_df = intl_df_raw[columns_to_keep].copy()
    intl_df = remove_exact_duplicates( intl_df )
    intl_df = remove_short_and_uninformative( intl_df, min_chars=100, min_words=10 )
    # Filtering sources: removing the least (bottom 1%) and most (top 5%) popular to reduce noise and prevent dominance.
    intl_df_filtered = filter_by_source_popularity( intl_df, lower_quantile=0.01, upper_quantile=0.95 )
    intl_df_final = prepare_text_and_metadata_for_bertopic( intl_df_filtered.copy() )
    intl_documents = intl_df_final['document'].tolist()
    intl_timestamps = intl_df_final['Дата'].tolist()

    # 3. Train or Load BERTopic Models
    print("\n--- Training/Loading Ukrainian Model ---")
    ukr_model_path = os.path.join(MODELS_DIR, "bertopic_model_ukr")
    ukr_topic_model = None # Initialize to None
    if os.path.exists(ukr_model_path):
        print(f"Loading existing Ukrainian model from: {ukr_model_path}...")
        try:
            ukr_topic_model = BERTopic.load(ukr_model_path)
            print("Ukrainian model loaded successfully.")
        except Exception as e:
            print(f"Error loading Ukrainian model: {e}. Retraining...")
            ukr_topic_model, topics_ukr, probs_ukr = configure_and_train_bertopic(
                documents=ukr_documents,
                language_code='uk',
                stop_words_list=ukrainian_stop_words,
                openai_api_key=OPENAI_API_KEY
            )
            ukr_df_final['topic'] = topics_ukr
            embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
            ukr_topic_model.save(UKR_NEWS_FOLDERS, serialization="safetensors",
                                   save_ctfidf=True, save_embedding_model=embedding_model )
    else:
        print("Training new Ukrainian model...")
        ukr_topic_model, topics_ukr, probs_ukr = configure_and_train_bertopic(
            documents=ukr_documents,
            language_code='uk',
            stop_words_list=ukrainian_stop_words,
            openai_api_key=OPENAI_API_KEY
        )
        embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
        ukr_topic_model.save( UKR_NEWS_FOLDERS, serialization="safetensors",
                              save_ctfidf=True, save_embedding_model=embedding_model )

    print("\n--- Training/Loading International Model ---")
    intl_model_path = os.path.join(MODELS_DIR, "bertopic_model_eng")
    intl_topic_model = None # Initialize to None
    if os.path.exists(intl_model_path):
        print(f"Loading existing International model from: {intl_model_path}...")
        try:
            intl_topic_model = BERTopic.load(intl_model_path)
            print("International model loaded successfully.")
        except Exception as e:
            print(f"Error loading International model: {e}. Retraining...")
            intl_topic_model, topics_intl, probs_intl = configure_and_train_bertopic(
                documents=intl_documents,
                language_code='en', # Or 'multilingual' if international news is in various languages
                stop_words_list='english',
                openai_api_key=OPENAI_API_KEY
            )
            intl_df_final['topic'] = topics_intl
            embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
            intl_topic_model.save(INTL_NEWS_FOLDERS, serialization="safetensors",
                                  save_ctfidf=True, save_embedding_model=embedding_model )
    else:
        print("Training new International model...")
        intl_topic_model, topics_intl, probs_intl = configure_and_train_bertopic(
            documents=intl_documents,
            language_code='en',
            stop_words_list='english',
            openai_api_key=OPENAI_API_KEY
        )
        intl_df_final['topic'] = topics_intl
        embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
        intl_topic_model.save(INTL_NEWS_FOLDERS, serialization="safetensors",
                              save_ctfidf=True, save_embedding_model=embedding_model )

    # Check if models were successfully loaded/trained before proceeding
    if ukr_topic_model is None or intl_topic_model is None:
        print("Error: One or both topic models could not be loaded or trained. Exiting.")
        return # Exit the main function if models are not available

    # 4. Get Topic Names from OpenAI
    # Ensure 'OpenAI' column has single strings, not lists, for both newly trained and loaded models.
    print("\n--- Getting Topic Info and Formatting OpenAI Names for Ukrainian Model ---")
    ukr_topic_info = ukr_topic_model.get_topic_info()
    ukr_topic_info['OpenAI'] = ukr_topic_info['OpenAI'].apply(unwrap_openai_name)
    if -1 in ukr_topic_info['Topic'].values:
        if ukr_topic_info.loc[ukr_topic_info['Topic'] == -1, 'OpenAI'].iloc[0] == "Untitled/Empty Topic":
            ukr_topic_info.loc[ukr_topic_info['Topic'] == -1, 'OpenAI'] = "Outlier Topic"

    print("\n--- Getting Topic Info and Formatting OpenAI Names for International Model ---")
    intl_topic_info = intl_topic_model.get_topic_info()
    intl_topic_info['OpenAI'] = intl_topic_info['OpenAI'].apply(unwrap_openai_name)
    if -1 in intl_topic_info['Topic'].values:
        if intl_topic_info.loc[intl_topic_info['Topic'] == -1, 'OpenAI'].iloc[0] == "Untitled/Empty Topic":
            intl_topic_info.loc[intl_topic_info['Topic'] == -1, 'OpenAI'] = "Outlier Topic"


    # Create maps from Topic ID to OpenAI name for visualization
    ukr_topic_name_map = ukr_topic_info.set_index('Topic')['OpenAI'].to_dict()
    intl_topic_name_map = intl_topic_info.set_index('Topic')['OpenAI'].to_dict()


    # 5. Visualize Topics Over Time (using your custom normalized function)
    print("\n--- Visualizing Topics Over Time for UKRAINIAN News ---")
    ukr_topics_over_time = ukr_topic_model.topics_over_time(ukr_documents, ukr_timestamps, nr_bins=20)
    visualize_topics_over_time_normalized(
        ukr_topic_model,
        ukr_topics_over_time,
        ukr_topic_name_map,
        "Relative Topic Frequency in UKRAINIAN News Over Time",
        top_n_topics=20,
        colors=px.colors.qualitative.Dark24
    )

    print("\n--- Visualizing Topics Over Time for INTERNATIONAL News ---")
    intl_topics_over_time = intl_topic_model.topics_over_time(intl_documents, intl_timestamps, nr_bins=20)
    visualize_topics_over_time_normalized(
        intl_topic_model,
        intl_topics_over_time,
        intl_topic_name_map,
        "Relative Topic Frequency in INTERNATIONAL News Over Time",
        top_n_topics=20,
        colors=px.colors.qualitative.Dark24
    )

    # 6. Visualize Topic Interrelationships (Semantic Similarity and Hierarchy)
    print("\n--- Visualizing Topic Interrelationships (UKRAINIAN) ---")
    # Use the pre-prepared map for custom labels for all visualizations
    # This directly uses the clean string names from OpenAI
    ukr_topic_model.set_topic_labels(ukr_topic_name_map)
    ukr_topic_model.visualize_topics( custom_labels=True ).show()
    ukr_topic_model.visualize_hierarchy( custom_labels=True ).show()


    print("\n--- Visualizing Topic Interrelationships (INTERNATIONAL) ---")
    # Use the pre-prepared map for custom labels for all visualizations
    # This directly uses the clean string names from OpenAI
    intl_topic_model.set_topic_labels(intl_topic_name_map)
    intl_topic_model.visualize_topics( custom_labels=True ).show()
    intl_topic_model.visualize_hierarchy( custom_labels=True ).show()

if __name__ == "__main__":
    main()