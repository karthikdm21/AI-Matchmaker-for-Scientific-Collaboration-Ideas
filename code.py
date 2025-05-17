import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
 
# Read the file line by line to avoid memory issues 
data = [] 
with open("arxiv-metadata-oai-snapshot.json", 'r', encoding='utf-8') as f: 
    for i, line in enumerate(f): 
        if i >= 1000:  # limit to 1000 samples for performance 
            break 
        data.append(pd.read_json(line, typ='series')) 
 
# Create DataFrame and select relevant fields 
df = pd.DataFrame(data) 
df = df[['title', 'abstract', 'authors', 'categories']].dropna() 
 
# Vectorize abstracts using TF-IDF 
vectorizer = TfidfVectorizer(max_features=5000) 
embeddings = vectorizer.fit_transform(df['abstract'].tolist()) 
 
# Compute similarity matrix 
similarity_matrix = cosine_similarity(embeddings) 
 
# Function to get top N similar papers 
def get_top_matches(similarity_matrix, top_n=3): 
    top_matches = [] 
    for idx, row in enumerate(similarity_matrix): 
        similar_indices = np.argsort(row)[::-1][1:top_n+1]  # Skip self-match at [0] 
        top_matches.append(similar_indices) 
    return top_matches 
 
# Get top matches for each paper 
top_matches = get_top_matches(similarity_matrix) 
 
# Calculate collaboration scores for all top matches 
collaboration_scores = [] 
for idx, matches in enumerate(top_matches): 
    for match_idx in matches: 
        score = similarity_matrix[idx][match_idx] * 100 
        collaboration_scores.append({ 
            'Paper 1 Title': df.iloc[idx]['title'], 
            'Paper 1 Authors': df.iloc[idx]['authors'], 
            'Paper 2 Title': df.iloc[match_idx]['title'], 
            'Paper 2 Authors': df.iloc[match_idx]['authors'], 
            'Collaboration Score': round(score, 2) 
        }) 
 
# Store all matches in a DataFrame 
results_df = pd.DataFrame(collaboration_scores) 
 
# Function to check category difference (interdisciplinary) 
def get_category_difference(paper1_idx, paper2_idx): 
    return df.iloc[paper1_idx]['categories'] != df.iloc[paper2_idx]['categories'] 
 
# Filter for interdisciplinary matches only 
interdisciplinary_matches = [] 
for idx, matches in enumerate(top_matches): 
    for match_idx in matches: 
        if get_category_difference(idx, match_idx): 
            score = similarity_matrix[idx][match_idx] * 100 
            interdisciplinary_matches.append({ 
                'Paper 1 Title': df.iloc[idx]['title'], 
                'Paper 1 Category': df.iloc[idx]['categories'], 
                'Paper 2 Title': df.iloc[match_idx]['title'], 
                'Paper 2 Category': df.iloc[match_idx]['categories'], 
                'Collaboration Score': round(score, 2) 
            }) 
 
 
# Show top 10 interdisciplinary matches directly in the notebook 
pd.set_option('display.max_colwidth', None)  # So long titles don't get cut off 
display(final_df.head(10))
