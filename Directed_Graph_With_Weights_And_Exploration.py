import wikipedia
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import networkx as nx

# Set language to English
wikipedia.set_lang("en")

# Load MiniLM model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get embeddings for a given text
def get_embeddings(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sentence_embeddings = torch.sum(token_embeddings * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

# Calculate cosine similarity between two sets of embeddings
def cosine_similarity(embeddings1, embeddings2):
    return torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)

# Get first paragraph from Wikipedia article
def get_first_paragraph(article_title):
    try:
        summary = wikipedia.summary(article_title)
        first_paragraph = summary.split('\n')[0]
        return first_paragraph
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
        return None

# Function to collect related articles with a maximum number
def collect_related_articles(article_title, max_related_articles):
    try:
        related_articles = wikipedia.page(article_title).links[:max_related_articles]
        return related_articles
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
        return []

# Perform DFS crawling with exploration of related articles for each node
def dfs_with_exploration(seed_article_title, max_nodes, max_related_articles_per_article, similarity_threshold):
    # Create a directed graph
    graph = nx.DiGraph()

    # Initialize stack for DFS crawling
    stack = [(seed_article_title, 1)]  # (article_title, depth)

    # Keep track of visited nodes
    visited_nodes = set()

    # Perform DFS crawling
    while stack:
        current_article, depth = stack.pop()
        if current_article in visited_nodes:
            continue

        visited_nodes.add(current_article)

        # Get main text and embeddings for the current article
        main_text = get_first_paragraph(current_article)
        main_embeddings = get_embeddings(main_text)

        # Collect related articles for the current article
        related_articles = collect_related_articles(current_article, max_related_articles_per_article)

        # Add node to the graph
        graph.add_node(current_article)

        # Explore related articles
        for related_article in related_articles:
            if related_article not in visited_nodes:
                # Get text and embeddings for the related article
                related_text = get_first_paragraph(related_article)
                if related_text:
                    related_embeddings = get_embeddings(related_text)

                    # Calculate similarity
                    similarity = cosine_similarity(main_embeddings, related_embeddings).item()

                    # Add related article as a node if similarity meets the threshold
                    if similarity >= similarity_threshold:
                        graph.add_node(related_article)
                        graph.add_edge(current_article, related_article, weight=similarity)
                        stack.append((related_article, depth + 1))

        # Terminate if maximum nodes limit reached
        if len(graph) >= max_nodes:
            break

    return graph

# Define parameters
seed_article_title = "Graph theory"
max_nodes = 500
max_related_articles_per_article = 50
similarity_threshold = 0.2

# Perform DFS crawling with exploration of related articles for each node
graph = dfs_with_exploration(seed_article_title, max_nodes, max_related_articles_per_article, similarity_threshold)

# Save the graph to a Gephi file
nx.write_gexf(graph, "Directed_Graph_With_Weights_And_Exploration.gexf")
