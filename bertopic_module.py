import openai
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

ukrainian_stop_words = [
    "а", "або", "але", "без", "біля", "більш", "був", "була", "були", "було", "бути", "вам", "вас", "весь", "він",
    "від", "вона", "вони", "все", "всі", "вся", "геть", "де", "для", "до", "ж", "же", "за", "звичайно", "зовсім",
    "і", "із", "им", "інші", "інший", "їй", "їм", "їх", "й", "коли", "кому", "котра", "котрий", "котре", "котрі",
    "крім", "куди", "лише", "може", "можна", "ми", "на", "навіть", "над", "наприклад", "нас", "не", "нею", "нижче",
    "них", "ні", "ніж", "нього", "о", "один", "одна", "одне", "одні", "от", "ось", "під", "після", "по", "поки",
    "поруч", "потім", "при", "про", "проте", "раз", "разом", "та", "таке", "такий", "також", "там", "те", "тепер",
    "тим", "тільки", "то", "той", "тому", "тут", "у", "хоча", "хто", "цей", "ця", "це", "ці", "час", "часто",
    "чого", "через", "чи", "чий", "чия", "чиє", "чиї", "що", "щоб", "як", "яка", "який", "яке", "які", "якраз",
    "якщо", "ясно", "я", "їй", "їм", "їх", "тощо", "таке", "таких", "це", "це", "ці", "цим", "цих", "яка", "які",
    "що", "на", "про", "для", "та", "не", "до", "за",
]

# Prompts for OpenAI representation models
prompt_urk = """
Завдання: Сформулювати заголовок для новинної теми.

Надані матеріали для аналізу:
- Документи, що входять до теми:
[DOCUMENTS]
- Ключові слова, що описують тему:
[KEYWORDS]

Спираючись на вищезазначену інформацію, створіть коротку, але максимально змістовну та описову назву або інформаційний привід для цієї новинної теми. Заголовок має бути схожим на реальний новинний заголовок і не повинен перевищувати 8 слів.

Формат відповіді:
topic: <ваш заголовок теми>
"""

prompt_eng = """
Task: Formulate a headline for a news topic.

Provided materials for analysis:
- Documents included in the topic:
[DOCUMENTS]
- Keywords describing the topic:
[KEYWORDS]

Based on the information above, create a short but highly meaningful and descriptive title or news event brief for this news topic. The headline should resemble a real news headline and should not exceed 8 words.

Response format:
topic: <your topic headline>
"""

def get_representation_model(current_language: str, openai_api_key: str = None) -> dict:
    """
    Initializes and returns a dictionary of BERTopic representation models
    according to the specified language.

    Args:
        current_language (str): The language for model setup ('uk' for Ukrainian, 'en' for English).
        openai_api_key (str, optional): Your OpenAI API key. Required if using OpenAI representation. Defaults to None.

    Returns:
        dict: A dictionary of BERTopic representation models.
    """
    print(f"\nSetting up representation models for language: {current_language.upper()}")

    # KeyBERT (universal model)
    keybert_model = KeyBERTInspired()

    # MMR (universal model)
    mmr_model = MaximalMarginalRelevance(diversity=0.3)

    # Part-of-Speech (language-dependent)
    pos_model = None
    try:
        if current_language == 'uk':
            pos_model_name = "uk_core_news_sm"
        elif current_language == 'en':
            pos_model_name = "en_core_web_sm"
        else:
            print(f"Warning: Unsupported language '{current_language}' for POS model. POS representation will be skipped.")
            pos_model_name = None # Set to None if language is not supported

        if pos_model_name:
            pos_model = PartOfSpeech(pos_model_name)
            print(f"SpaCy model '{pos_model_name}' loaded successfully for PartOfSpeech.")

    except OSError as e:
        print(f"Error loading SpaCy model for language {current_language}: {e}")
        print(f"Please ensure you have run '!python -m spacy download {pos_model_name}' and it completed successfully.")
        pos_model = None # Set to None if loading fails

    # OpenAI (language-dependent and requires API key)
    openai_model = None
    if openai_api_key:
        openai_selected_prompt = None
        if current_language == 'uk':
            openai_selected_prompt = prompt_urk
        elif current_language == 'en':
            openai_selected_prompt = prompt_eng
        else:
            print(f"Warning: Unsupported language '{current_language}' for OpenAI prompt. OpenAI representation will be skipped.")

        if openai_selected_prompt:
            try:
                client = openai.OpenAI(api_key=openai_api_key)
                openai_model = OpenAI(client, model="gpt-4o-mini", exponential_backoff=True, prompt=openai_selected_prompt)
                print(f"OpenAI model initialized with prompt for {current_language.upper()}.")
            except Exception as e:
                print(f"Error initializing OpenAI model: {e}. OpenAI representation will be skipped.")
                openai_model = None # Set to None if initialization fails
    else:
        print("OpenAI API key not provided. OpenAI representation will be skipped.")


    # Collect all representation models into a dictionary
    representation_models_dict = {
        "KeyBERT": keybert_model,
        "MMR": mmr_model,
    }

    if pos_model: # Add POS only if the model was successfully loaded
        representation_models_dict["POS"] = pos_model

    if openai_model: # Add OpenAI only if the model was successfully initialized
        representation_models_dict["OpenAI"] = openai_model

    print(f"Final representation models for {current_language.upper()}: {list(representation_models_dict.keys())}")
    return representation_models_dict

def configure_and_train_bertopic(
    documents: list[str],
    language_code: str,
    stop_words_list: list = None, # Can be a list of words or the string 'english'
    openai_api_key: str = None, # Pass API key for OpenAI Representation
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    umap_min_dist: float = 0.0,
    umap_metric: str = 'cosine',
    hdbscan_min_cluster_size: int = 20,
    hdbscan_metric: str = 'euclidean',
    hdbscan_cluster_selection_epsilon: float = 0.5,
    vectorizer_ngram_range: tuple = (1, 2)
):
    """
    Configures and trains a BERTopic model for a given language and dataset.

    Args:
        documents (list[str]): List of preprocessed text documents.
        timestamps (list): List of corresponding timestamps for each document.
        language_code (str): The language code ('uk' or 'en') for the BERTopic model and representation.
        stop_words_list (list/str, optional): List of stop words or 'english'.
                                              If None, CountVectorizer default is used.
        openai_api_key (str, optional): Your OpenAI API key. Required if using OpenAI representation. Defaults to None.
        embedding_model_name (str): Name of the SentenceTransformer model.
        umap_n_neighbors (int): Number of neighbors for UMAP.
        umap_n_components (int): Number of components for UMAP.
        umap_min_dist (float): Minimum distance for UMAP.
        umap_metric (str): Metric for UMAP.
        hdbscan_min_cluster_size (int): Minimum cluster size for HDBSCAN.
        hdbscan_metric (str): Metric for HDBSCAN.
        hdbscan_cluster_selection_epsilon (float): Cluster selection epsilon for HDBSCAN.
        vectorizer_ngram_range (tuple): N-gram range for CountVectorizer.

    Returns:
        BERTopic: The trained BERTopic model.
        list: List of topic assignments for each document.
        list: List of topic probabilities for each document.
    """

    print(f"\n--- Training BERTopic Model for {language_code.upper()} News ---")

    # 1. Embedding Model (universal for both languages)
    embedding_model = SentenceTransformer(embedding_model_name)

    # 2. UMAP (Dimensionality Reduction)
    umap_model = UMAP(n_neighbors=umap_n_neighbors,
                      n_components=umap_n_components,
                      min_dist=umap_min_dist,
                      metric=umap_metric,
                      random_state=4747) # For reproducibility

    # 3. HDBSCAN (Clustering)
    hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                            metric=hdbscan_metric,
                            cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
                            prediction_data=True) # Required for `transform` method later

    # 4. CountVectorizer (Topic Representation)
    if stop_words_list == 'english':
        vectorizer_model = CountVectorizer(stop_words='english', ngram_range=vectorizer_ngram_range)
    elif isinstance(stop_words_list, list):
        vectorizer_model = CountVectorizer(stop_words=stop_words_list, ngram_range=vectorizer_ngram_range)
    else:
        print(f"Warning: Stop words not specified or recognized for {language_code}. Using default CountVectorizer.")
        vectorizer_model = CountVectorizer(ngram_range=vectorizer_ngram_range)

    # 5. Representation Model (obtained using our universal function)
    representation_model = get_representation_model(current_language=language_code, openai_api_key=openai_api_key)

    # Initialize BERTopic
    topic_model = BERTopic(
        language="multilingual",
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics="auto", # Let HDBSCAN decide the number of topics
        calculate_probabilities=True,
        verbose=True # Show progress
    )

    # Fit the model
    topics, probs = topic_model.fit_transform(documents)

    print(f"Number of topics found for {language_code.upper()} News: {len(topic_model.get_topics()) - 1}") # -1 for outlier topic

    return topic_model, topics, probs