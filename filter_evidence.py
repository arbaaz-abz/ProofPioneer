import json
import sqlite3
import nltk
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')


def load_unwanted_patterns(config_path: str) -> List[re.Pattern]:
    with open(config_path, "r") as f:
        patterns = json.load(f)
    return [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]


def retrieve_all_rows(db_path: str = "outputs/index.db") -> List[Dict[str, Any]]:
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()            
            cursor.execute("SELECT * FROM index_table")
            rows = cursor.fetchall()                        
            return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"Error accessing the database: {e}")
        return []

def retrieve_row_by_key(key: str, db_path: str = "outputs/index.db") -> Dict[str, Any]:
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM index_table WHERE key = ?", (key,))
            row = cursor.fetchone()
            return dict(row) if row else None
    except sqlite3.Error as e:
        print(f"Error accessing the database: {e}")
        return None

def extract_relevant_sentences(sentences: List[str], model: SentenceTransformer, claim_embedding: Any, threshold: float = 0.65) -> (List[int], List[float]):
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    similarity_scores = util.cos_sim(claim_embedding, sentence_embeddings).squeeze().tolist()
    del sentence_embeddings
    if isinstance(similarity_scores, float):
        similarity_scores = [similarity_scores]
    relevant_indices = [i for i, score in enumerate(similarity_scores) if score >= threshold]
    return relevant_indices, similarity_scores

def build_segments(sentences: List[str], relevant_indices: List[int], model: SentenceTransformer, claim_embedding: Any, context_size: int = 2, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
    segments = []
    for idx in relevant_indices:
        start = max(idx - context_size, 0)
        end = min(idx + context_size + 1, len(sentences))
        segment_text = ' '.join(sentences[start:end])
        segments.append({
            'segment_start': start,
            'segment_end': end,
            'text': segment_text
        })

    segments.sort(key=lambda x: x['segment_start'])
    # refined_segments = []
    # for seg in segments:
    #     seg_sentences = nltk.sent_tokenize(seg['text'])
    #     seg_embeddings = model.encode(seg_sentences, convert_to_tensor=True)
    #     seg_similarity = util.cos_sim(claim_embedding, seg_embeddings).mean().item()
    #     if seg_similarity >= similarity_threshold:
    #         refined_segments.append(seg)
    
    return segments
    # return refined_segments

def clean_and_check_text(webpage_text: str, unwanted_patterns: List[re.Pattern]) -> str:
    if webpage_text == "NA":
        return webpage_text, True

    # Replace two or more newlines with a single newline
    cleaned_text = re.sub(r'\n{2,}', '\n', webpage_text)

    if cleaned_text == "":
        return webpage_text, True

    # Check if any of the unwanted patterns matches the text
    flag = False
    for pattern in unwanted_patterns:
        if pattern.search(cleaned_text):
            flag = True
            break
    return cleaned_text, flag

def process_claims(search_results: Dict[str, Any], database_path: str, prompt_claim_entities_path: str, prompt_entity_relationship_path: str):   
    # Initialize sentence transformer model
    # embeddings_model = SentenceTransformer('all-MiniLM-L12-v2')
    embeddings_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    use_segments = False
    # ombine_evidence_pieces = True

    compiled_patterns = load_unwanted_patterns("unwanted_config.json")
    
    filtered_articles = {}
    for claim_index, claim in enumerate(search_results):

        # if claim_index > 25: # 15 made it break
        #     break

        print(f"\nProcessing Claim ID: {claim_index}\n")
        filtered_articles[claim] = {}
        claim_embedding = embeddings_model.encode(claim, convert_to_tensor=True)
        
        claim_object = search_results[claim]
        for query_index, query in enumerate(claim_object.keys()):
            for page_index in claim_object[query].keys():
                for article_index, article_metadata in enumerate(claim_object[query][page_index]):

                    # if article_index > 10:
                    #     break

                    # TODO: Use article_metadata for first level filtering

                    search_key = f"{claim_index}_{query_index}_{page_index}_{article_index}"
                    if search_key in filtered_articles:
                        print("Article already processed")
                        continue

                    row = retrieve_row_by_key(search_key, database_path)
                    
                    if row:
                        article_path = row['path']
                        print(f"Loading Article: {article_path}")
                        with open(article_path, "r", encoding='utf-8') as f:
                            data = json.load(f)
                            title = data['title'] if 'title' in data else ""
                            article_text, flag = clean_and_check_text(data['text'], compiled_patterns)
                            article_text = f"{title}\n{article_text}"

                            # This article is not useful
                            if flag:
                                print(f"Skipping article: {article_path}")
                                continue
                    else:
                        print("Row not found for key:", search_key)
                        continue

                    if article_text == "NA":
                        continue
                    
                    # Tokenize sentences
                    sentences = nltk.sent_tokenize(article_text)

                    # Remove large sentences (context length limit is 8192 tokens)
                    # A helpful rule of thumb is that one token generally corresponds to ~4 characters of text for common English text. This translates to roughly ¾ of a word (so 100 tokens ~= 75 words).
                    # sentences = [sentence for sentence in sentences if len(sentence.split()) <= 15000]

                    print(f"Number of Sentences (before filtering): {len(sentences)}")

                    # Remove sentences that are less than 10 words
                    # Trim sentences that are greater than 1700 characters long
                    sentences = [sentence for sentence in sentences if len(nltk.tokenize.word_tokenize(sentence)) >= 10 and len(sentence) <= 2000]
                    # sentences = [sentence[:1700] for sentence in sentences]

                    print(f"Number of Sentences (after filtering): {len(sentences)}")

                    if len(sentences) > 1000:
                        print(f"Skipping article: {article_path}, too long")
                        continue
                    elif len(sentences) == 0:
                        print(f"Skipping article: {article_path}, no sentences")
                        continue

                    # Step 1: Filter sentences based on similarity
                    sentence_threshold = 0.55
                    relevant_sentence_indices, similarity_scores = extract_relevant_sentences(
                        sentences,
                        embeddings_model,
                        claim_embedding,
                        threshold=sentence_threshold
                    )
                    
                    print(f"Number of Relevant Sentences (threshold={sentence_threshold}): {len(relevant_sentence_indices)}\n")

                    if len(relevant_sentence_indices) == 0:
                        print("No relevant sentences found")
                        continue

                    # for idx in relevant_sentence_indices:
                    #     print(f"Sentence {idx} (Score: {similarity_scores[idx]:.2f}):\n{sentences[idx]}\n")

                    # TODO: Use other strategies for segment extraction
                    if use_segments:
                        # Step 2: Build segments with surrounding context
                        context_size = 1  # Number of sentences before and after
                        segment_threshold = 0.5
                        segments_with_context = build_segments(
                            sentences,
                            relevant_sentence_indices,
                            embeddings_model,
                            claim_embedding,
                            context_size=context_size,
                            similarity_threshold=segment_threshold
                        )
                        print(f"Number of Segments with Context (context size={context_size}): {len(segments_with_context)}\n")

                        # article_summary = ""
                        # for seg in segments_with_context:
                        #     print(f"Segment {seg['segment_start']}-{seg['segment_end'] - 1}:\n{seg['text']}\n")

                        filtered_articles[claim][search_key] = {
                            "scores": [similarity_scores[i] for i in relevant_sentence_indices],
                            "filtered_segments": [seg['text'] for seg in segments_with_context]                        
                        }
                    else:
                        # article_summary = '\n'.join(sentences[i] for i in relevant_sentence_indices)
                        filtered_articles[claim][search_key] = {
                            "scores": [similarity_scores[i] for i in relevant_sentence_indices],
                            "filtered_segments": [sentences[i] for i in relevant_sentence_indices]                        
                        }

        with open("outputs/filtered_results.json", "w") as f:
            json.dump(filtered_articles, f, indent=2)
    
    # Save the knowledge relationships to a JSON file
def main():
    search_results_path = "outputs/search_results.json"
    database_path = "outputs/index.db"
    prompt_claim_entities_path = "prompts/entity_extraction.txt"
    prompt_entity_relationship_path = "prompts/entity_relationship_extraction.txt"
    
    # Load search results
    with open(search_results_path, "r", encoding='utf-8') as fp:
        search_results = json.load(fp)
    
    process_claims(
        search_results=search_results,
        database_path=database_path,
        prompt_claim_entities_path=prompt_claim_entities_path,
        prompt_entity_relationship_path=prompt_entity_relationship_path
    )

if __name__ == "__main__":
    main()


# TODO: For each extracted high quality sentence, include its surrounding sentences to provide more context. 
# TODO: CHECK DIFFERENT EMBEDDING COMPARISION - Jaccard Similarity, Cosine Similarity, Euclidean Distance

