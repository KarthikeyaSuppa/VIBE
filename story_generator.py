from pymongo import MongoClient
from pinecone import Pinecone
import boto3
import os
from dotenv import load_dotenv
from groq import Groq
import torch
from transformers import AutoTokenizer, AutoModel
from bson import ObjectId
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import networkx as nx
import numpy as np
import io
from safetensors.torch import load
import tempfile
import json



# Load environment variables from .env file
load_dotenv()

# Access environment variables
mongodb_uri = os.getenv("MONGODB_URI")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
S3_MODEL_KEY = os.getenv("AWS_S3_MODEL_KEY")

# Groq setup
groq_client = Groq(api_key=groq_api_key)

# Global variables
colbert_model = None
tokenizer = None
s3_client = None
search_system = None

def get_s3_client():
    global s3_client
    if s3_client is None:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
            region_name=os.getenv("AWS_REGION")
        )
    return s3_client

def get_model_config():
    """Fetch model config from S3"""
    s3 = get_s3_client()
    response = s3.get_object(
        Bucket="vibe-01",
        Key="colbert-xm/config.json"
    )
    return json.loads(response['Body'].read())

def get_tokenizer_files():
    """Fetch tokenizer files from S3"""
    s3 = get_s3_client()
    tokenizer_files = {}
    for filename in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        response = s3.get_object(
            Bucket="vibe-01",
            Key=f"colbert-xm/{filename}"
        )
        tokenizer_files[filename] = response['Body'].read()
    return tokenizer_files

def init_tokenizer():
    """Initialize tokenizer directly from S3 files"""
    global tokenizer
    if tokenizer is None:
        try:
            # Create temporary directory for tokenizer files
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save tokenizer files temporarily
                tokenizer_files = get_tokenizer_files()
                for filename, content in tokenizer_files.items():
                    with open(f"{tmp_dir}/{filename}", 'wb') as f:
                        f.write(content)
                # Initialize tokenizer from temporary directory
                tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
                return tokenizer
        except Exception as e:
            print(f"Error initializing tokenizer: {e}")
            return None
    return tokenizer

def load_model_weights(query_text):
    """Load only necessary model weights for processing a specific query"""
    try:
        s3 = get_s3_client()
        
        # Get only the necessary part of model.safetensors
        # This is a simplified example - you'll need to implement proper partial loading
        response = s3.get_object(
            Bucket="vibe-01",
            Key="colbert-xm/model.safetensors",
            Range='bytes=0-1024000'  # Adjust range based on your needs
        )
        
        # Load partial weights
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(response['Body'].read())
            tmp.flush()
            partial_weights = load(tmp.name)
            
        return partial_weights
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None

def get_query_embedding(prompt: str):
    """Convert query to embedding using ColBERT-XM"""
    try:
        global colbert_model, tokenizer
        
        # Initialize if not done
        if tokenizer is None:
            tokenizer = init_tokenizer()
        if colbert_model is None:
            colbert_model = load_colbert_model()
            
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Generate embedding
        with torch.no_grad():
            outputs = colbert_model(**inputs)
            # Use mean pooling to get single vector
            embedding = outputs.last_hidden_state.mean(dim=1)
            
        return embedding
        
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None

def process_with_partial_weights(inputs, weights):
    """Process inputs with partial weights loading"""
    try:
        # Initialize model architecture
        model_config = get_model_config()
        model = AutoModel.from_pretrained(
            "antoinelouis/colbert-xm",
            config=model_config,
            state_dict=weights,
            trust_remote_code=True
        )
        
        # Set to evaluation mode
        model.eval()
        
        # Process inputs in smaller chunks
        with torch.no_grad():
            outputs = model(**inputs)
            # Get embeddings from the last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings
        
    except Exception as e:
        print(f"Error processing with partial weights: {e}")
        return None

def init_search_system():
    """Initialize the search system with tokenizer and model"""
    try:
        print("Initializing search system...")
        # Initialize tokenizer
        global tokenizer, colbert_model
        tokenizer = init_tokenizer()
        if tokenizer is None:
            raise ValueError("Failed to initialize tokenizer")
            
        # Initialize model config
        model_config = get_model_config()
        if model_config is None:
            raise ValueError("Failed to get model config")
            
        print("Search system initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing search system: {e}")
        return False

# Search for similar vectors
def search_vectors(query_embedding, index, k=5):
    try:
        if query_embedding is None:
            raise ValueError("Invalid query embedding")

        results = index.query(
            vector=query_embedding.tolist(),
            top_k=k*2,
            include_metadata=True
        )

        # Commented out debug print
        # print("Vector search results:", results.matches[:2])

        if not results.matches:
            return []

        # Determine collection based on metadata fields
        valid_matches = []
        for match in results.matches:
            metadata = getattr(match, 'metadata', {})

            # Determine collection based on metadata fields
            if 'characters' in metadata and 'plot_points' in metadata:
                metadata['collection'] = 'stories'
            elif 'name' in metadata and 'role' in metadata and 'traits' in metadata:
                metadata['collection'] = 'characters'
            elif 'type' in metadata and 'location' in metadata and 'participants' in metadata:
                metadata['collection'] = 'scenes'

            if metadata.get('collection'):
                valid_matches.append(match)
            # Commented out debug print
            # else:
            #     print(f"Skipping match due to unrecognized metadata format: {metadata}")

        if not valid_matches:
            return []

        return random.sample(valid_matches, min(k, len(valid_matches)))

    except Exception as e:
        print(f"Error searching vectors: {e}")
        return []

# Get documents from MongoDB
def get_documents(matches, db):
    documents = []
    try:
        for match in matches:
            # Commented out debug prints
            # print(f"Processing match: {match}")

            metadata = getattr(match, 'metadata', {})
            collection_name = metadata.get('collection')
            doc_id = metadata.get('_id')

            # Commented out debug print
            # print(f"Looking up document in collection '{collection_name}' with ID '{doc_id}'")

            if not collection_name or not doc_id:
                # print(f"Skipping match due to missing metadata: {metadata}")
                continue

            try:
                collection = db[collection_name]
                doc = collection.find_one({'_id': ObjectId(doc_id)})

                if doc:
                    doc['collection'] = collection_name
                    documents.append(doc)
                    # Commented out debug print
                    # print(f"Successfully found document in {collection_name}")
                # else:
                    # print(f"No document found for ID '{doc_id}' in collection '{collection_name}'")

            except Exception as doc_error:
                print(f"Error processing document: {str(doc_error)}")
                continue

        return documents
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []

# Format results
def format_results(documents):
    try:
        if not documents:
            return "No results found"

        output = ""
        for doc in documents:
            output += f"\n{'='*40}\n"
            output += f"Collection: {doc['collection']}\n\n"

            if doc['collection'] == 'characters':
                output += f"Name: {doc.get('name', 'Unknown')}\n"
                output += f"Role: {doc.get('role', 'Unknown')}\n"
                output += f"Traits: {', '.join(doc.get('traits', []))}\n"
                output += f"Backstory: {doc.get('backstory', 'N/A')[:512]}...\n"
                output += f"Relationships: {doc.get('relationships', 'N/A')[:512]}...\n"

            elif doc['collection'] == 'scenes':
                output += f"Type: {doc.get('type', 'Unknown')}\n"
                output += f"Location: {doc.get('location', 'Unknown')}\n"
                output += f"Description: {doc.get('description', 'N/A')[:512]}...\n"
                output += f"Participants: {', '.join(doc.get('participants', []))}\n"
                output += f"Actions: {doc.get('actions', 'N/A')[:512]}...\n"

            elif doc['collection'] == 'stories':
                output += f"Title: {doc.get('title', 'Untitled')}\n"
                output += f"Themes: {', '.join(doc.get('themes', []))}\n"
                output += f"Text: {doc.get('text', 'N/A')[:512]}...\n"
                output += f"Scenes: {doc.get('scenes', 'N/A')[:1554]}...\n"
                output += f"Plotpoints: {doc.get('plotpoints', 'N/A')[:512]}...\n"

        return output
    except Exception as e:
        print(f"Error formatting results: {e}")
        return "Error formatting results"

# Main search function
# Modify the existing search function to include feedback
def search(query_text, k=5):
    try:
        index, db = init_search_system()
        if None in (index, db):
            return "Failed to initialize search system"
        
        # Initialize feedback collection
        feedback_collection = init_feedback_collection(db)
        if feedback_collection is None:
            print("Warning: Feedback system not initialized")
        
        query_embedding = get_query_embedding(query_text)
        if query_embedding is None:
            return "Failed to create query embedding"
        
        matches = search_vectors(query_embedding, index, k)
        if not matches:
            return "No matching results found"
        
        documents = get_documents(matches, db)
        if not documents:
            return "Failed to retrieve documents"
        
        # Adjust results based on feedback
        adjusted_documents = adjust_search_results(documents, query_text, db, k)
        
        return format_results(adjusted_documents)
    except Exception as e:
        return f"Error during search: {str(e)}"
    

class CharacterMapper:
    def __init__(self, db):
        self.db = db
        self.character_graph = nx.Graph()
        self.character_cache = {}
        
    def build_relationship_graph(self) -> None:
        """Build a graph of character relationships from the database"""
        characters = self.db.characters.find({})
        
        for char in characters:
            char_id = str(char['_id'])
            self.character_cache[char['name']] = char
            self.character_graph.add_node(
                char['name'],
                traits=char.get('traits', []),
                role=char.get('role', 'unknown'),
                backstory=char.get('backstory', '')
            )
            
            # Parse relationships array which contains alternating names and relationship types
            relationships = char.get('relationships', [])
            for i in range(0, len(relationships), 2):
                if i + 1 < len(relationships):
                    related_char = relationships[i]
                    relation_type = relationships[i + 1]
                    self.character_graph.add_edge(
                        char['name'],
                        related_char,
                        relationship=relation_type
                    )
    
    def get_character_connections(self, character_name: str, depth: int = 2) -> Dict:
        """Get all connections for a character up to specified depth"""
        if not self.character_graph.has_node(character_name):
            return {}
            
        connections = defaultdict(list)
        for other_char in self.character_graph.nodes():
            if other_char == character_name:
                continue
                
            paths = nx.all_simple_paths(
                self.character_graph,
                character_name,
                other_char,
                cutoff=depth
            )
            
            for path in paths:
                relationship_chain = []
                for i in range(len(path) - 1):
                    edge_data = self.character_graph.get_edge_data(path[i], path[i+1])
                    relationship_chain.append(edge_data['relationship'])
                
                connections[other_char].append({
                    'path': path,
                    'relationships': relationship_chain
                })
                
        return connections
    
    def find_story_relevant_characters(self, story_doc: Dict) -> List[Dict]:
        """Find all relevant characters for a story including their relationships"""
        story_characters = story_doc.get('characters', [])
        relevant_chars = []
        
        for char_name in story_characters:
            if char_name in self.character_cache:
                char_data = self.character_cache[char_name].copy()
                char_data['connections'] = self.get_character_connections(char_name)
                relevant_chars.append(char_data)
                
        return relevant_chars
    
    def generate_relationship_context(self, story_doc: Dict) -> str:
        """Generate narrative context based on character relationships"""
        relevant_chars = self.find_story_relevant_characters(story_doc)
        context = []
        
        for char in relevant_chars:
            char_context = [f"{char['name']} is a {', '.join(char.get('traits', []))} character who {char.get('backstory', '')}"]
            
            # Add relationship context
            for other_char, connections in char.get('connections', {}).items():
                for connection in connections:
                    relationship_path = ' → '.join(connection['relationships'])
                    char_context.append(f"- Connected to {other_char} via: {relationship_path}")
            
            context.append('\n'.join(char_context))
            
        return '\n\n'.join(context)


def enhance_story_generation(db, story_doc: Dict) -> Dict:
    """Enhance story document with character relationship context"""
    mapper = CharacterMapper(db)
    mapper.build_relationship_graph()
    relationship_context = mapper.generate_relationship_context(story_doc)
    enhanced_doc = story_doc.copy()
    enhanced_doc['character_context'] = relationship_context
    return enhanced_doc



def generate_story_with_context(user_query, enhanced_results):
    """Generate story using the enhanced context while maintaining the original prompt structure"""
    try:
        # Initialize system
        _, db = init_search_system()
        
        # Detect genre and get parameters
        genre = detect_genre(user_query)
        genre_params = get_genre_parameters(genre)
        
        # Define the system prompt (keeping your original prompt)
        predefined_prompt = """Positive prompt:
You are a narrative story generative chatbot, whose goal is to take the query of the user and the provided summary it has a random story content that can include various elements. Feel free to be creative and add your own twist, while ensuring the story progresses in a manner
that reveals the conflict and then the context that resolves it, layer by layer like an onion.

In creating the story, keep in mind the following aspects of pre-writing:

1. Concept and Theme: Develop a core idea or theme that your story will explore.
2. Character Development: Create well-rounded characters with distinct personalities, motivations, flaws, and goals.
3. Setting: Establish a believable environment, considering time period, location, and atmosphere.
4. Plot Outline: Map out the major events of your story, including the inciting incident, rising action, climax, falling action, and resolution.

When writing the story, incorporate the following elements:

1. Hooking Introduction: Start with an engaging opening line or scene that captures the reader's attention and introduces the main conflict.
2. Dialogue: Use realistic dialogue to reveal character traits and advance the plot.
3. Show, Don't Tell: Describe details vividly using sensory language to immerse the reader in the story rather than simply stating facts.
4. Conflict and Tension: Build suspense by creating obstacles and challenges for your characters, raising the stakes as the story progresses.
5. Pacing: Balance action and exposition to maintain a steady rhythm throughout the story.
6. Point of View: Choose a consistent narrative perspective to guide the reader's experience.

Ensure your story includes the following key elements:

1. Character Arc: Show how your protagonist develops and changes throughout the story in response to the conflict.
2. Climax: The turning point of the story where the central conflict reaches its peak.
3. Resolution: The conclusion where the conflicts are resolved, leaving the reader with a sense of satisfaction.

Lastly, consider the following important aspects:

1. Consistency: Ensure your characters, setting, and plot remain consistent throughout the story.
2. Voice: Develop a unique writing style that reflects the tone and perspective of your story.
3. Revision and Editing: Once your first draft is complete, thoroughly revise and edit to polish your writing, correct errors, and improve clarity.

Additional Character Guidelines:
- Use the provided character relationships to create authentic interactions
- Maintain consistency with established character traits and backstories
- Develop conflicts that arise naturally from character relationships
- Show character growth through their interactions with related characters
- Use established relationships to create deeper emotional resonance

Negative Prompt: stuff saying here is your story, anything from positive prompt
Now Finally Generate story Dont Include anything in the negative prompt.
Give this as intro for user one line above the story - I'd be delighted to spin a yarn for you! Here's the story:
"""

        # Create additional context
        additional_context = f"""
Additional Context for Story Enhancement:
Genre: {genre}
Style Elements:
- Tone: {genre_params['tone']}
- Key elements to consider: {', '.join(genre_params['elements'])}
- Suggested pacing: {genre_params['pacing']}
- Character focus: {genre_params.get('character_focus', 'balanced')}
- Setting emphasis: {genre_params.get('setting_emphasis', 'moderate')}

Character Relationships and Background:
{enhanced_results}
"""
        
        # Combine everything for the final message
        user_message_content = f"Query: {user_query}\n\n{additional_context}"
        
        # Use the enhanced prompt for story generation
        groq_client = Groq(api_key=groq_api_key)
        messages = [
            {"role": "system", "content": predefined_prompt},
            {"role": "user", "content": user_message_content}
        ]
        
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile"
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error in contextual story generation: {str(e)}"


def generate_story(user_query):

    try:
           # Initialize system
        _, db = init_search_system()
        
        # Detect genre and get parameters
        genre = detect_genre(user_query)
        genre_params = get_genre_parameters(genre)
        
        # Perform vector search
        search_results = search(user_query, k=3)

        # Define the system prompt
        predefined_prompt = """Positive prompt:
You are a narrative story generative chatbot, whose goal is to take the query of the user and the provided summary it has a random story content that can include various elements. Feel free to be creative and add your own twist, while ensuring the story progresses in a manner
that reveals the conflict and then the context that resolves it, layer by layer like an onion.

In creating the story, keep in mind the following aspects of pre-writing:

1. Concept and Theme: Develop a core idea or theme that your story will explore.
2. Character Development: Create well-rounded characters with distinct personalities, motivations, flaws, and goals.
3. Setting: Establish a believable environment, considering time period, location, and atmosphere.
4. Plot Outline: Map out the major events of your story, including the inciting incident, rising action, climax, falling action, and resolution.

When writing the story, incorporate the following elements:

1. Hooking Introduction: Start with an engaging opening line or scene that captures the reader's attention and introduces the main conflict.
2. Dialogue: Use realistic dialogue to reveal character traits and advance the plot.
3. Show, Don't Tell: Describe details vividly using sensory language to immerse the reader in the story rather than simply stating facts.
4. Conflict and Tension: Build suspense by creating obstacles and challenges for your characters, raising the stakes as the story progresses.
5. Pacing: Balance action and exposition to maintain a steady rhythm throughout the story.
6. Point of View: Choose a consistent narrative perspective to guide the reader's experience.

Ensure your story includes the following key elements:

1. Character Arc: Show how your protagonist develops and changes throughout the story in response to the conflict.
2. Climax: The turning point of the story where the central conflict reaches its peak.
3. Resolution: The conclusion where the conflicts are resolved, leaving the reader with a sense of satisfaction.

Lastly, consider the following important aspects:

1. Consistency: Ensure your characters, setting, and plot remain consistent throughout the story.
2. Voice: Develop a unique writing style that reflects the tone and perspective of your story.
3. Revision and Editing: Once your first draft is complete, thoroughly revise and edit to polish your writing, correct errors, and improve clarity.
Negative Prompt: stuff saying here is your stroy,anything form positive prompt
Now Finally Generate story Dont Include anything in the negative prompt.
Give this as intro for user one line aboc=ve the story - I'd be delighted to spin a yarn for you! Here's the story:
"""

        # Create additional context while preserving your original prompt
        additional_context = f"""
Additional Context for Story Enhancement:
Genre: {genre}
Style Elements:
- Tone: {genre_params['tone']}
- Key elements to consider: {', '.join(genre_params['elements'])}
- Suggested pacing: {genre_params['pacing']}
- Character focus: {genre_params.get('character_focus', 'balanced')}
- Setting emphasis: {genre_params.get('setting_emphasis', 'moderate')}
"""
        
        # Combine everything for the final message
        user_message_content = f"Query: {user_query}\n\nSearch Results:\n{search_results}\n\n{additional_context}"
        
        # Use existing prompt as the main system prompt
        groq_client = Groq(api_key=groq_api_key)
        messages = [
            {"role": "system", "content": predefined_prompt},  # Your original prompt
            {"role": "user", "content": user_message_content}
        ]
        
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile"
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error in story generation process: {str(e)}"
    

def modified_generate_story(prompt: str):
    """Complete story generation pipeline"""
    try:
        print(f"Processing prompt: {prompt}")
        
        # 1. Generate query embedding
        print("Generating query embedding...")
        embedding = get_query_embedding(prompt)
        if embedding is None:
            raise ValueError("Failed to generate query embedding")
            
        # 2. Get relevant content
        print("Fetching relevant content...")
        relevant_content = get_relevant_content(embedding)
        if relevant_content is None:
            raise ValueError("Failed to fetch relevant content")
            
        # 3. Generate story using LLM
        print("Generating story with LLM...")
        story = generate_story_with_llm(prompt, relevant_content)
        if story is None:
            raise ValueError("Failed to generate story")
            
        return {
            "status": "success",
            "story": story,
            "context_used": relevant_content  # Optional: for debugging
        }
        
    except Exception as e:
        print(f"Error in story generation pipeline: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_relevant_content(embedding):
    """Get relevant content using embeddings"""
    try:
        # Connect to your vector store (Pinecone/MongoDB)
        # Search for relevant content using the embedding
        # This is a placeholder - implement your vector search logic
        
        relevant_content = [
            "Sample relevant content 1",
            "Sample relevant content 2"
        ]
        return relevant_content
        
    except Exception as e:
        print(f"Error getting relevant content: {e}")
        return None

def generate_story_with_llm(prompt: str, relevant_content: list):
    """Generate story using LLM with relevant content"""
    try:
        client = groq_client
        
        # Construct prompt with relevant content
        context = "\n".join(relevant_content)
        
        messages = [
            {
                "role": "system",
                "content": """You are a creative storyteller. Use the provided context 
                to create an engaging story. Incorporate elements from the context 
                while maintaining narrative coherence."""
            },
            {
                "role": "user",
                "content": f"""Context information:
                {context}
                
                Using this context, write a story based on this prompt: {prompt}"""
            }
        ]
        
        # Generate story
        completion = client.chat.completions.create(
            model="llama2-70b-4096",
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating story with LLM: {e}")
        return None

def init_feedback_collection(db):
    """Initialize feedback collection with necessary indexes"""
    try:
        feedback_collection = db['feedback']
        # Create indexes for efficient querying
        feedback_collection.create_index([('query_text', 1)])
        feedback_collection.create_index([('rating', 1)])
        feedback_collection.create_index([('timestamp', -1)])
        return feedback_collection
    except Exception as e:
        print(f"Error initializing feedback collection: {e}")
        return None

def store_feedback(query_text, story_text, rating, feedback_text=None):
    
    try:
        # Get database connection
        _, db = init_search_system()
        if db is None:
            print("Failed to connect to database")
            return False
            
        # Ensure feedback collection exists
        feedback_collection = db['feedback']
        
        # Create feedback document
        feedback_data = {
            'query_text': query_text,
            'story_text': story_text[:1000],  # Store first 1000 chars of story for reference
            'rating': rating,
            'feedback_text': feedback_text,
            'timestamp': datetime.utcnow()
        }
        
        # Insert feedback
        feedback_collection.insert_one(feedback_data)
        
        # Optional: Print confirmation
        print("Feedback successfully stored in database")
        return True
        
    except Exception as e:
        print(f"Error storing feedback: {e}")
        return False

def adjust_search_results(documents, query_text, db, k=5):
    """Adjust search results based on historical feedback"""
    try:
        feedback_collection = db['feedback']
        adjusted_documents = []
        
        for doc in documents:
            # Get average rating for this document
            ratings = feedback_collection.find({
                'document_id': str(doc['_id']),
                'rating': {'$exists': True}
            })
            
            total_rating = 0
            rating_count = 0
            
            for rating_doc in ratings:
                total_rating += rating_doc['rating']
                rating_count += 1
            
            # Calculate boost factor based on ratings
            boost_factor = 1.0
            if rating_count > 0:
                avg_rating = total_rating / rating_count
                boost_factor = avg_rating / 3.0  # Normalize around 1.0
            
            # Add boost factor to document
            doc['relevance_score'] = getattr(doc, 'score', 1.0) * boost_factor
            adjusted_documents.append(doc)
        
        # Sort by adjusted relevance score
        adjusted_documents.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return adjusted_documents[:k]
    except Exception as e:
        print(f"Error adjusting search results: {e}")
        return documents   
    
def detect_genre(query_text):
    """Detect the likely genre from the query"""
    genres = {
        'fantasy': ['magic', 'dragon', 'wizard', 'mythical', 'fairy', 'elf'],
        'scifi': ['space', 'robot', 'future', 'alien', 'technology', 'cyberpunk'],
        'romance': ['love', 'relationship', 'heart', 'romantic', 'date'],
        'mystery': ['detective', 'crime', 'solve', 'mystery', 'clue'],
        'horror': ['scary', 'ghost', 'horror', 'terrifying', 'spooky'],
        'adventure': ['quest', 'journey', 'explore', 'discover', 'adventure']
    }
    
    query_lower = query_text.lower()
    genre_scores = {}
    
    for genre, keywords in genres.items():
        score = sum(keyword in query_lower for keyword in keywords)
        if score > 0:
            genre_scores[genre] = score
    
    if genre_scores:
        return max(genre_scores.items(), key=lambda x: x[1])[0]
    return 'general'

def get_genre_parameters(genre):
    """Get genre-specific generation parameters"""
    genre_params = {
        'fantasy': {
            'elements': ['world-building', 'magic systems', 'mythical creatures'],
            'tone': 'epic and magical',
            'pacing': 'balanced between action and wonder',
            'character_focus': 'high',
            'setting_emphasis': 'very high'
        },
        'scifi': {
            'elements': ['technology', 'scientific concepts', 'future implications'],
            'tone': 'speculative and thought-provoking',
            'pacing': 'measured with moments of intensity',
            'character_focus': 'moderate',
            'setting_emphasis': 'high'
        },
        'romance': {
            'elements': ['character relationships', 'emotional development', 'personal growth'],
            'tone': 'emotionally resonant',
            'pacing': 'focused on relationship development',
            'character_focus': 'high',
            'setting_emphasis': 'moderate'
        },
        'mystery': {
            'elements': ['clues', 'suspense', 'revelation'],
            'tone': 'intriguing and suspenseful',
            'pacing': 'steady build-up of tension',
            'character_focus': 'moderate',
            'setting_emphasis': 'moderate'
        },
        'horror': {
            'elements': ['atmosphere', 'tension', 'fear'],
            'tone': 'dark and suspenseful',
            'pacing': 'building dread with moments of intensity',
            'character_focus': 'moderate',
            'setting_emphasis': 'high'
        },
        'adventure': {
            'elements': ['action', 'exploration', 'challenges'],
            'tone': 'exciting and dynamic',
            'pacing': 'fast-paced with moments of discovery',
            'character_focus': 'high',
            'setting_emphasis': 'high'
        },
        'general': {
            'elements': ['character development', 'plot progression', 'engaging narrative'],
            'tone': 'balanced and natural',
            'pacing': 'story-appropriate',
            'character_focus': 'balanced',
            'setting_emphasis': 'moderate'
        }
    }
    return genre_params.get(genre, genre_params['general'])