import asyncio
import json
import re
import aiohttp
import httpx
import openai
import fitz
import time
from io import BytesIO
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from lxml import html
from youtube_transcript_api import YouTubeTranscriptApi


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# System message for the chat model
SYSTEM_CONTENT = "Be a helpful assistant"

# Constants for web search
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
DEFAULT_RESULTS = 3
TIMEOUT = 10  # seconds
RETRY_LIMIT = 3
RATE_LIMIT = 0.5  # seconds


# Global variables to store API key, base URL, and models
api_key = None
base_url = None
openai_client = None
preloaded_models = []

# File to store settings
SETTINGS_FILE = 'settings.json'

# Function to load settings from file
def load_settings():
    global api_key, base_url, openai_client
    try:
        with open(SETTINGS_FILE, 'r') as file:
            settings = json.load(file)
            api_key = settings.get('api_key')
            base_url = settings.get('base_url')
            if api_key and base_url:
                openai_client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
    except (FileNotFoundError, json.JSONDecodeError):
        api_key = None
        base_url = None
        openai_client = None

# Function to save settings to file
def save_settings(api_key, base_url):
    settings = {
        'api_key': api_key,
        'base_url': base_url
    }
    with open(SETTINGS_FILE, 'w') as file:
        json.dump(settings, file)

# Function to fetch models from the API
def fetch_models():
    if not api_key or not base_url:
        return []
    
    models_url = f"{base_url}/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = httpx.get(models_url, headers=headers)
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                if isinstance(response_data, list):
                    models = response_data
                elif isinstance(response_data, dict):
                    models = response_data.get('data', [])
                else:
                    print("Unexpected response format")
                    return []
                
                # Extracting the 'id' field from each dictionary
                return [model['id'] for model in models if 'id' in model]
            except ValueError:
                print("Failed to parse JSON response")
                return []
        else:
            print(f"Failed to retrieve models. Status code: {response.status_code}")
            return []
    except httpx.RequestError as e:
        print(f"An error occurred while making the request: {e}")
        return []


# Function to preload models on app startup
def preload_models():
    global preloaded_models
    preloaded_models = fetch_models()

# Function to fetch search results from DuckDuckGo Lite
async def fetch_results(session, query, results=DEFAULT_RESULTS, retries=RETRY_LIMIT):
    url = 'https://lite.duckduckgo.com/lite/'
    data = {
        'q': query
    }
    headers = {
        'User-Agent': USER_AGENT
    }
    for attempt in range(retries + 1):
        try:
            async with session.post(url, data=data, headers=headers, timeout=TIMEOUT) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientError:
            if attempt < retries:
                await asyncio.sleep(RATE_LIMIT)
            else:
                return None

# Function to parse search results from HTML content
def parse_results(html_content, results=DEFAULT_RESULTS):
    if html_content is None:
        return []
    
    tree = html.fromstring(html_content)
    results_list = tree.xpath('//tr//td//a[@href]')
    if not results_list:
        return []
    
    links = []
    countLink = 0
    print()

    for a in results_list:
        href = a.get('href')
        if 'duckduckgo.com' not in href and 'reddit.com' not in href and 'youtube.com' not in href:
            countLink += 1
            print(f'Source URL number {countLink}: {href}')
            links.append(href)
            if len(links) == results:
                break
    print()
    countLink = 0
    return links

def clean_html_text(text):
    """Quick HTML text cleaning that preserves structure and readability"""
    try:
        tree = html.fromstring(text)
        
        # Remove script and style elements completely
        for element in tree.xpath('//script | //style'):
            element.getparent().remove(element)
        
        # Extract text content
        text_content = tree.text_content()
        
        # Quick and simple cleaning for speed
        # Normalize line breaks
        text_content = text_content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split and quickly process lines
        lines = text_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip and only add non-empty lines
            stripped = line.strip()
            if stripped:
                # Quick space normalization
                stripped = ' '.join(stripped.split())
                cleaned_lines.append(stripped)
        
        return '\n'.join(cleaned_lines)
    except Exception as e:
        # Return empty string if HTML parsing fails
        return ""

async def fetch_and_format_text(session, url, index, retries=RETRY_LIMIT):
    for attempt in range(retries + 1):
        try:
            async with session.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT) as response:
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    return ""
                html_content = await response.text()
                cleaned_text = clean_html_text(html_content)
                return f"Source text {index} from website {url}: \n \n {cleaned_text} \n \n"
        except (aiohttp.ClientError, Exception):
            if attempt < retries:
                await asyncio.sleep(RATE_LIMIT)
            else:
                return ""

# Function to get DuckDuckGo search results and texts
async def get_duckduckgo_results_and_texts(query, results=DEFAULT_RESULTS):
    async with aiohttp.ClientSession() as session:
        html_content = await fetch_results(session, query, results)
        links = parse_results(html_content, results)
        if not links:
            return [], []
        tasks = [fetch_and_format_text(session, link, i + 1) for i, link in enumerate(links)]
        formatted_texts = await asyncio.gather(*tasks)
        return links, formatted_texts

def filter_reasoning_content(conversation_history, start_tag='<think>', end_tag='</think>'):
    """
    Filter out reasoning content (between start and end tags) from conversation history
    only if the assistant message starts with start tag and has a corresponding end tag.
    Only removes the first reasoning section between the first start tag and its end tag.
    """
    filtered_history = []
    for message in conversation_history:
        if message['role'] == 'assistant' and 'content' in message:
            content = message['content']
            # Check if content starts with start_tag and contains end_tag
            if content.startswith(start_tag) and end_tag in content:
                # Find the first occurrence of end_tag
                first_end_pos = content.find(end_tag)
                if first_end_pos != -1:
                    # Keep content after the first end_tag
                    filtered_content = content[first_end_pos + len(end_tag):].strip()
                    if filtered_content:
                        filtered_history.append({'role': 'assistant', 'content': filtered_content})
                    else:
                        # If no content after end_tag, skip this message
                        continue
                else:
                    # If no end_tag found after start_tag, keep the entire message
                    filtered_history.append(message)
            else:
                # Keep the message as is if it doesn't start with start_tag or doesn't have end_tag
                filtered_history.append(message)
        else:
            # Keep non-assistant messages as is
            filtered_history.append(message)
    return filtered_history

# Function to handle web search command
def handle_search_command(user_content, results=DEFAULT_RESULTS):
    query = user_content
    if not query:
        return "Please provide a search query"
    
    try:
        links, formatted_texts = asyncio.run(get_duckduckgo_results_and_texts(query, results))
        if not links:
            return "No results found"
        
        return ''.join(formatted_texts)
    except Exception as e:
        return f"An error occurred: {e}"

# Function to handle YouTube command
def handle_youtube_command(user_content):
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',  # URLs
        r'^[a-zA-Z0-9_-]{11}$'  # Direct video ID
    ]
    
    video_id = None
    for pattern in patterns:
        match = re.search(pattern, user_content)
        if match:
            video_id = match.group(1) if len(match.groups()) > 0 else match.group(0)
            break
    
    if video_id:
        try:
            # Fetch the list of available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            language_available = []
            for transcript in transcript_list:
              transcript_language = transcript.language_code
              language_available.append(transcript_language)
            
            # Check if there is an English transcript available
            if 'en' in language_available:
                transcript = transcript_list.find_transcript(['en']).fetch()
            else:
                # If no English transcript, get the first available language
                transcript = transcript_list.find_transcript([language_available[0]]).fetch()
            
            # Join the transcript entries into a single string with no newlines
            transcript_text = ' '.join(snippet.text for snippet in transcript.snippets)
            return transcript_text
        except Exception as e:
            return f"Error getting transcript: {str(e)}"
    else:
        return "Please provide a valid YouTube URL or video ID"


# Function to handle webpage command
def handle_webpage_command(user_content):
    """Handle general webpage URLs, returning the extracted text."""
    pattern = r'https?://[^\s]+'
    match = re.search(pattern, user_content)
    
    if not match:
        return None
        
    url = match.group(0)
    try:
        response = httpx.get(url, follow_redirects=True)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            return ""
        
        cleaned_text = clean_html_text(response.text)
        
        return f"Source text from website {url}: \n \n {cleaned_text} \n \n"
    except httpx.HTTPStatusError as e:
        # Specifically handle HTTP errors like 404, 403, etc., after following redirects
        raise Exception(f"HTTP error {e.response.status_code} while fetching the webpage: {e}")
    except httpx.RequestError as e:
        # Handle network errors, timeouts, etc.
        raise Exception(f"Network error occurred while fetching the webpage: {e}")
    except Exception as e:
        # Handle any other unexpected errors during fetching/parsing
        raise Exception(f"An unexpected error occurred while fetching the webpage: {e}")

def handle_arxiv_command(user_content):
    """Handle arXiv PDF and abstract URLs, returning the extracted text."""
    arxiv_pattern = r'https?://arxiv\.org/(abs|pdf)/\d+\.\d+(v\d+)?'
    arxiv_match = re.search(arxiv_pattern, user_content)
    
    if not arxiv_match:
        return None
        
    arxiv_link = arxiv_match.group(0)
    arxiv_type = arxiv_match.group(1)  # 'abs' or 'pdf'
    
    try:
        response = httpx.get(arxiv_link)
        response.raise_for_status()
        
        if arxiv_type == 'abs':
            # Extract abstract from HTML
            text = response.text
            start_marker = "Abstract:</span>"
            end_marker = "Comments:"
            start_index = text.find(start_marker) + len(start_marker)
            end_index = text.find(end_marker, start_index)
            
            if start_index == -1 or end_index == -1:
                raise Exception("Abstract not found in the response.")
            
            return text[start_index:end_index].strip()
        else:
            # Handle PDF
            pdf_file = BytesIO(response.content)
            pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
            return " ".join(page.get_text() for page in pdf_document)
            
    except Exception as e:
        raise Exception(f"Failed to process arXiv {arxiv_type}: {str(e)}")

# Route to render the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to fetch models
@app.route('/fetch-models', methods=['GET'])
def fetch_models_route():
    return jsonify(preloaded_models)

# Route to handle saving settings
@app.route('/save-settings', methods=['POST'])
def save_settings_route():
    global api_key, base_url, openai_client, preloaded_models
    api_key = request.json.get('apiKey')
    base_url = request.json.get('baseUrl')
    openai_client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    save_settings(api_key, base_url)
    preloaded_models = fetch_models()
    return jsonify({"status": "success"})

# Route to handle chat requests
@app.route('/chat', methods=['POST'])
def chat():
    user_content = request.json.get('message')
    conversation_history = request.json.get('conversation', [])
    selected_model = request.json.get('model')
    system_content = request.json.get('systemContent', SYSTEM_CONTENT)
    parameters = request.json.get('parameters', {})
    is_deep_query_mode = request.json.get('isDeepQueryMode', False)
    start_tag = request.json.get('startTag', '<think>')

    # Convert string values to appropriate types for numeric parameters
    if parameters:
        for key, value in parameters.items():
            if isinstance(value, str):
                # Try int first
                try:
                    parameters[key] = int(value)
                    continue
                except ValueError:
                    pass
                # Then try float
                try:
                    parameters[key] = float(value)
                except ValueError:
                    # leave non-numeric strings untouched
                    pass
            # non-string values are left as-is

    additional_text = ""
    # Only process search commands if user_content is a string (not an image message)
    if isinstance(user_content, str):
        if user_content.lower().startswith("@s") and (len(user_content) == 2 or user_content[2].isspace()):
            user_content = user_content[2:].strip()

            # Check for YouTube link
            if re.search(r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/.+', user_content):
                additional_text = handle_youtube_command(user_content)
                user_content = re.sub(r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/[^ ]+', '', user_content).strip()
                if user_content:
                    system_content = "You are an assistant specialized in Question & Answer. Please provide a clear and concise response to the user query based on the video transcript. Query: {}".format(user_content)
                    user_content = f"{user_content} \n\n "
                else:
                    system_content = "You are an assistant specialized in summarizing videos. Please provide a clear, concise and well-formatted summary of the video content."

            # Check for arXiv link
            elif re.search(r'https?://arxiv\.org/(abs|pdf)/\d+\.\d+(v\d+)?', user_content):
                additional_text = handle_arxiv_command(user_content)
                if additional_text is None:
                    return "Invalid arXiv URL"
                # Extract any user query after the arXiv link
                user_content = re.sub(r'https?://arxiv\.org/(abs|pdf)/\d+\.\d+(v\d+)?[^ ]*', '', user_content).strip()
                if user_content:
                    system_content = system_content = "You are an assistant specialized in Question & Answer. Please provide a clear and concise response to the user query based on the arXiv paper. Query: {}".format(user_content)
                    user_content = f"{user_content} \n\n "
                else:
                    system_content = "You are an assistant specialized in summarizing arXiv papers. Please provide a clear, concise and well-formatted summary of the paper's content."

            # Check for general link
            elif re.search(r'https?://[^\s]+', user_content):
                additional_text = handle_webpage_command(user_content)
                if additional_text is None:
                    return "Please provide a valid URL"
                user_content = re.sub(r'https?://[^\s]+[^ ]*', '', user_content).strip()
                if user_content:
                    system_content = "You are an assistant specialized in Question & Answer. Please provide a clear and concise response to the user query based on the webpage content. Query: {}".format(user_content)
                    user_content = f"{user_content} \n\n "
                else:
                    system_content = "You are an assistant specialized in summarizing webpages. Please provide a clear, concise and well-formatted summary of the webpage content."

            # No link, treat as general search
            else:
                additional_text = handle_search_command(user_content)
                user_content = f"SEARCH QUERY: {user_content} \n\n "
                system_content = f"""CURRENT_SYSTEM_TIME = f"{time.strftime("%Y-%m-%d %H:%M:%S")}" \n \n 
                                You are a knowledgeable search assistant. Analyze the following search query and use latest information from the provided source texts to create a comprehensive response: \n \n 

                                SEARCH QUERY: {user_content} \n \n 

                                Instructions:
                                - Focus ONLY on directly answering the query using the provided sources
                                - NO general background or context unless specifically requested
                                - Provide accurate, detailed information using an unbiased, journalistic tone
                                - Use markdown formatting for better readability:
                                • Lists and bullet points for multiple items
                                • Code blocks with language specification
                                • Tables for structured data
                                - Focus on factual information without subjective statements
                                - Organize information logically with clear paragraph breaks
                                - Match the query's language and tone

                                For specialized topics:
                                - Academic: Provide detailed analysis with proper sections
                                - News: Summarize key points with bullet points.
                                - Technical: Include code blocks with language specification
                                - Scientific: Use LaTeX for formulas (\\(inline\\) or \\[block\\])
                                - Biographical: Focus on key facts and achievements
                                - Products: Group options by category (max 5 recommendations)
                                """

    # Filter reasoning content from conversation history
    filtered_history = filter_reasoning_content(conversation_history, start_tag, end_tag='</think>')
    
    # Handle messages with images
    if isinstance(user_content, list):
        # The message contains both text and image
        messages = [{"role": "system", "content": system_content}] if system_content else []
        messages.extend(filtered_history)
        messages.append({"role": "user", "content": user_content})
    else:
        # Regular text message
        if system_content:
            messages = [{"role": "system", "content": system_content}] + filtered_history + [{"role": "user", "content": user_content + additional_text}]
        else:
            messages = filtered_history + [{"role": "user", "content": user_content + additional_text}]

    # Add deep query mode message if enabled
    if is_deep_query_mode:
        messages.append({"role": "assistant", "content": f"{start_tag}"})

    def generate():
        if openai_client is None:
            yield "Please set your API key and base URL in the settings."
            return

        # Track the current state: None, 'reasoning', or 'content'
        current_mode = None

        try:
            # Create the stream (keeping your existing parameter logic)
            if parameters:
                stream = openai_client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    stream=True,
                    **parameters
                )
            else:
                stream = openai_client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    stream=True
                )

            for chunk in stream:
                if not chunk.choices or not chunk.choices[0].delta:
                    continue
                
                delta = chunk.choices[0].delta
                
                # 1. Handle Reasoning Content
                # We check truthiness (val) to ignore empty strings often sent as keep-alives
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    if current_mode != 'reasoning':
                        # We are entering reasoning mode
                        yield "<think>"
                        current_mode = 'reasoning'
                    yield delta.reasoning_content
                
                # 2. Handle Regular Content
                # Use elif because a delta usually contains one or the other
                elif delta.content:
                    if current_mode == 'reasoning':
                        # We are leaving reasoning mode
                        yield "</think>"
                        current_mode = 'content'
                    yield delta.content
            
        except Exception as e:
            # If an error occurs, ensure we close the tag if we are inside reasoning
            if current_mode == 'reasoning':
                yield "</think>"
                current_mode = None # Update state so finally doesn't double-close
            yield f"An error occurred: {str(e)}"
        
        finally:
            # FINAL SAFETY NET: Ensure the tag is closed even if the stream 
            # ends abruptly without sending a final content chunk.
            if current_mode == 'reasoning':
                yield "end"

    return Response(generate(), mimetype='text/event-stream')

# Route to handle chat requests
@app.route('/continue_generation', methods=['POST'])
def continue_generation():
    conversation_history = request.json.get('conversation', [])
    selected_model = request.json.get('model', "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    system_content = request.json.get('systemContent', SYSTEM_CONTENT)
    parameters = request.json.get('parameters', {})

    # Convert string values to appropriate types for numeric parameters
    if parameters:
        for key, value in parameters.items():
            if isinstance(value, str):
                # Try int first
                try:
                    parameters[key] = int(value)
                    continue
                except ValueError:
                    pass
                # Then try float
                try:
                    parameters[key] = float(value)
                except ValueError:
                    # leave non-numeric strings untouched
                    pass
            # non-string values are left as-is

    # Filter reasoning content from conversation history
    filtered_history = filter_reasoning_content(conversation_history, start_tag='<think>', end_tag='</think>')
    
    if system_content == '':
        messages = filtered_history
    else:
        messages = [{"role": "system", "content": system_content}] + filtered_history

    def generate():
        if openai_client is None:
            yield "Please set your API key and base URL in the settings."
            return

        # Track the current state: None, 'reasoning', or 'content'
        current_mode = None

        try:
            # Create the stream (keeping your existing parameter logic)
            if parameters:
                stream = openai_client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    stream=True,
                    **parameters
                )
            else:
                stream = openai_client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    stream=True
                )

            for chunk in stream:
                if not chunk.choices or not chunk.choices[0].delta:
                    continue
                
                delta = chunk.choices[0].delta
                
                # 1. Handle Reasoning Content
                # We check truthiness (val) to ignore empty strings often sent as keep-alives
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    if current_mode != 'reasoning':
                        # We are entering reasoning mode
                        yield "<think>"
                        current_mode = 'reasoning'
                    yield delta.reasoning_content
                
                # 2. Handle Regular Content
                # Use elif because a delta usually contains one or the other
                elif delta.content:
                    if current_mode == 'reasoning':
                        # We are leaving reasoning mode
                        yield "</think>"
                        current_mode = 'content'
                    yield delta.content
            
        except Exception as e:
            # If an error occurs, ensure we close the tag if we are inside reasoning
            if current_mode == 'reasoning':
                yield "</think>"
                current_mode = None # Update state so finally doesn't double-close
            yield f"An error occurred: {str(e)}"
        
        finally:
            # FINAL SAFETY NET: Ensure the tag is closed even if the stream 
            # ends abruptly without sending a final content chunk.
            if current_mode == 'reasoning':
                yield "end"

    return Response(generate(), mimetype='text/event-stream')

# Route to generate a title for the conversation
@app.route('/generate-title', methods=['POST'])
def generate_title():
    message = request.json.get('message')
    selected_model = request.json.get('model')
    assistant_response = request.json.get('assistantResponse', '')
    
    try:
        messages = [
            {
                "role": "system",
                "content": "Generate a 5-word max title for this conversation. Focus on the main topic. Respond ONLY with the title without any quotation."
            },
            {
                "role": "user",
                "content": f"User message: {message} \n \n Assistant response: {assistant_response}"
            }
        ]
        
        response = openai_client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=0
        )
        
        title = response.choices[0].message.content.strip()
        return jsonify({"title": title})
    except Exception as e:
        print(f"Error generating title: {e}")
        return jsonify({"title": None})

# Load settings and preload models when the app starts
print("Starting Chat WebUI")
load_settings()
preload_models()

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
