import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

columns_to_keep = ["Дата", "Джерело", "Заголовок", "Опис", "Популярність джерела"]

def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes exact duplicates based on the combination of 'Заголовок' (Headline) and 'Опис' (Description).

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with exact duplicates removed.
    """
    initial_rows = len(df)

    # Create a combined text column for duplicate detection
    # Fill NaN values with empty strings to prevent errors during concatenation
    df["combined_text"] = df["Заголовок"].fillna("") + " " + df["Опис"].fillna("")

    # Drop duplicates based on the 'combined_text' column
    df_cleaned = df.drop_duplicates(subset=["combined_text"]).copy()
    df_cleaned = df_cleaned.drop(columns=["combined_text"])
    removed_rows = initial_rows - len(df_cleaned)
    print(f"Removed {removed_rows} exact duplicates.")

    return df_cleaned


def clean_text_for_length(text: str) -> str:
    """
    Cleans text by converting to lowercase, removing URLs, and normalizing whitespace.
    Used internally for length-based filtering.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove URLs if present
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Normalize whitespace (replace multiple spaces with a single one and strip leading/trailing)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_short_and_uninformative(df: pd.DataFrame, min_chars: int = 100, min_words: int = 10) -> pd.DataFrame:
    """
    Removes entries where the combined headline and description text is too short or uninformative
    after basic cleaning.

    Args:
        df (pd.DataFrame): The input DataFrame.
        min_chars (int): Minimum number of characters required for the combined text.
        min_words (int): Minimum number of words required for the combined text.

    Returns:
        pd.DataFrame: A new DataFrame with short/uninformative entries removed.
    """
    initial_rows = len(df)

    # Create a cleaned combined text column for length analysis
    df["cleaned_combined_text_for_length"] = df.apply(
        lambda row: clean_text_for_length(row["Заголовок"]) + " " + clean_text_for_length(row["Опис"]), axis=1
    )

    # Filter out entries where the text is too short
    df_filtered = df[
        (df["cleaned_combined_text_for_length"].str.len() >= min_chars) &
        (df["cleaned_combined_text_for_length"].apply(lambda x: len(x.split())) >= min_words)
    ].copy()

    # Remove the temporary cleaned text column
    df_filtered = df_filtered.drop(columns=["cleaned_combined_text_for_length"])

    removed_rows = initial_rows - len(df_filtered)
    print(f"Removed {removed_rows} short or uninformative entries (min_chars={min_chars}, min_words={min_words}).")
    return df_filtered


def plot_source_popularity_distribution(df, col="Популярність джерела", bins=50, show_kde=True, title="Source Popularity Distribution"):
    """
    Plot a histogram of values in the specified numerical column (typically 'Source Popularity').

    Parameters:
    - df: pandas DataFrame
    - col: name of the numerical column to visualize (default = 'Популярність джерела')
    - bins: number of histogram bins (default = 50)
    - show_kde: whether to show the KDE (density curve) overlay (default = True)
    - title: title of the plot (default = 'Source Popularity Distribution')
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col], bins=bins, kde=show_kde, color="steelblue", edgecolor="black")
    plt.title(title, fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def filter_by_source_popularity(df: pd.DataFrame, col: str = "Популярність джерела", lower_quantile: float = 0.20,
                                upper_quantile: float = 0.95) -> pd.DataFrame:
    """
    Filters the DataFrame based on the popularity of the source, removing outliers.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): Name of the popularity column.
        lower_quantile (float): The lower percentile threshold (e.g., 0.20 to keep data above 20th percentile).
        upper_quantile (float): The upper percentile threshold (e.g., 0.95 to keep data below 95th percentile).

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    initial_rows = len( df )

    # Describe the popularity column to get statistical overview
    print( "\nSource Popularity Statistics:" )
    print( df[col].describe() )

    # Calculate thresholds based on quantiles
    low_thresh = df[col].quantile( lower_quantile )
    high_thresh = df[col].quantile( upper_quantile )

    print( f"\nFiltering '{col}' based on quantiles:" )
    print( f"  Lower threshold ({lower_quantile * 100:.0f}th percentile): {low_thresh:.2f}" )
    print( f"  Upper threshold ({upper_quantile * 100:.0f}th percentile): {high_thresh:.2f}" )

    # Apply the filter
    df_filtered = df[
        (df[col] >= low_thresh) &
        (df[col] <= high_thresh)
        ].copy()

    removed_rows = initial_rows - len( df_filtered )
    print( f"Removed {removed_rows} entries outside the popularity range." )
    print( f"Remaining entries after popularity filtering: {len( df_filtered )}" )
    return df_filtered


def clean_text_for_topic_modeling(text: str) -> str:
    """
    Applies basic text preprocessing suitable for BERT-based models:
    - Converts to lowercase.
    - Removes punctuation (keeping only letters, numbers, and spaces).
    - Removes digits if they are not considered informative for topic modeling.
    - Normalizes whitespace.

    Args:
        text (str): Input text from 'Заголовок' or 'Опис'.

    Returns:
        str: Cleaned text.
    """
    if pd.isna( text ):
        return ""

    text = str( text )  # Ensure text is string

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation and special characters (keep letters, numbers, and spaces)
    # Keeping numbers here, as they might be part of specific events/products
    text = re.sub( r'[^a-zA-Zа-яА-ЯіІїЇєЄґҐ\d\s]', '', text )

    # 3. Normalize whitespace (replace multiple spaces with a single one and strip leading/trailing)
    text = re.sub( r'\s+', ' ', text ).strip()

    return text


def prepare_text_and_metadata_for_bertopic(df: pd.DataFrame, date_col: str = 'Дата') -> pd.DataFrame:
    """
    Prepares the DataFrame for BERTTopic by:
    - Combining 'Заголовок' (Headline) and 'Опис' (Description) into a 'document' column.
    - Applying basic text cleaning to the 'document' column.
    - Converting the date column to datetime objects.
    - Removing any empty documents or rows with invalid dates after processing.

    Args:
        df (pd.DataFrame): The input DataFrame (already filtered by popularity/duplicates).
        date_col (str): The name of the date column.

    Returns:
        pd.DataFrame: DataFrame with 'document' and processed 'Дата' columns, ready for BERTTopic.
    """
    initial_rows = len( df )

    # Create the combined text column
    df['document'] = df['Заголовок'].fillna( '' ) + ' ' + df['Опис'].fillna( '' )

    # Apply text cleaning to the 'document' column
    print( "Applying text cleaning to 'document' column..." )
    df['document'] = df['document'].apply( clean_text_for_topic_modeling )

    # Remove any documents that became empty after cleaning
    df = df[df['document'].str.len() > 0].copy()
    if len( df ) < initial_rows:
        print( f"Removed {initial_rows - len( df )} documents that became empty after cleaning." )
        initial_rows = len( df )  # Update initial_rows for date check

    # Convert date column to datetime
    if date_col in df.columns:
        print( f"Converting '{date_col}' column to datetime..." )
        df[date_col] = pd.to_datetime( df[date_col], errors='coerce' )

        # Remove rows where date conversion failed
        df.dropna( subset=[date_col], inplace=True )
        if len( df ) < initial_rows:
            print( f"Removed {initial_rows - len( df )} rows due to invalid date values in '{date_col}'." )
    else:
        print( f"Warning: Date column '{date_col}' not found in DataFrame. Cannot prepare timestamps." )

    return df