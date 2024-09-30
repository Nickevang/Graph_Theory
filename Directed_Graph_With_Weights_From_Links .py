import wikipedia
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import networkx as nx
import random

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

# Define the Wikipedia article
seed_article_title = "Graph theory"
max_nodes = 5000

# Create a directed graph
graph = nx.DiGraph()

# Add seed article as a node
graph.add_node(seed_article_title)

# Queue for BFS crawling
queue = [seed_article_title]

# Maximum number of related articles for each article
max_related_articles_per_article = 20

# Similarity threshold
similarity_threshold = 0.2

# Keep track of added nodes
added_nodes = set([seed_article_title])

# Counter for added nodes
node_count = 1

# Perform BFS crawling
while queue and node_count < max_nodes:
    current_article = queue.pop(0)
    related_articles = collect_related_articles(current_article, max_related_articles_per_article)
    main_text = get_first_paragraph(current_article)
    main_embeddings = get_embeddings(main_text)
    for related_article in related_articles:
        if related_article not in added_nodes:
            related_text = get_first_paragraph(related_article)
            if related_text:
                related_embeddings = get_embeddings(related_text)
                similarity = cosine_similarity(main_embeddings, related_embeddings).item()
                if similarity >= similarity_threshold:
                    graph.add_node(related_article)
                    graph.add_edge(current_article, related_article, weight=similarity)
                    queue.append(related_article)
                    added_nodes.add(related_article)
                    node_count += 1
                    if node_count >= max_nodes:
                        break

# Save the graph to a Gephi file
nx.write_gexf(graph, "Directed_Graph_With_Weights_From_Links.gexf")
