from flask import Flask, render_template, request, jsonify, Response
import lmstudio as lms
import json
import os
from werkzeug.utils import secure_filename
from werkzeug.exceptions import ClientDisconnected
import PyPDF2  # For PDF text extraction
import docx  # For Word document text extraction

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'md', 'csv', 'json', 'xml'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load system context from file ===
CONTEXT_FILE = 'context.txt'
if not os.path.exists(CONTEXT_FILE):
    # Create a default context file if missing
    with open(CONTEXT_FILE, 'w', encoding='utf-8') as f:
        f.write("You are a helpful and concise AI assistant.")

with open(CONTEXT_FILE, 'r', encoding='utf-8') as f:
    SYSTEM_CONTEXT = f.read().strip()

# Initialize the LM Studio model
model = lms.llm("falcon3-3b-instruct")

# Store conversation history in memory
conversation_history = []

def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath, filename):
    """Extract text from various file formats."""
    ext = filename.rsplit('.', 1)[1].lower()
    
    try:
        if ext == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif ext == 'pdf':
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        
        elif ext in ['doc', 'docx']:
            doc = docx.Document(filepath)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        elif ext == 'md':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif ext == 'csv':
            import csv
            with open(filepath, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                text = ""
                for row in csv_reader:
                    text += ", ".join(row) + "\n"
                return text
        
        elif ext == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        
        elif ext == 'xml':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            return None
    
    except Exception as e:
        print(f"Error extracting text from {filename}: {str(e)}")
        return None

def build_prompt(history, current_message, system_context):
    """Construct a prompt with system context and chat history."""
    prompt = f"{system_context}\n\n"
    for turn in history:
        role = "User" if turn['role'] == 'user' else "Assistant"
        prompt += f"{role}: {turn['content']}\n"
    prompt += f"User: {current_message}\nAssistant: "
    return prompt

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and text extraction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from the file
        extracted_text = extract_text_from_file(filepath, filename)
        
        if extracted_text:
            # Clean up the uploaded file after extraction
            os.remove(filepath)
            
            # Add file content to conversation history
            file_message = f"[Uploaded file: {filename}]\n\nFile content:\n{extracted_text[:2000]}"  # Limit content length
            
            # Add to conversation history
            conversation_history.append({'role': 'user', 'content': file_message})
            
            return jsonify({
                'success': True,
                'filename': filename,
                'preview': extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                'message': f"File '{filename}' uploaded successfully. You can now ask questions about its content."
            })
        else:
            # Clean up if extraction failed
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Could not extract text from the file'}), 400
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/remove_file', methods=['POST'])
def remove_file():
    """Remove uploaded file content from conversation history."""
    global conversation_history
    
    # Remove any messages that contain uploaded file content
    initial_length = len(conversation_history)
    conversation_history = [
        msg for msg in conversation_history 
        if not (msg['role'] == 'user' and (
            msg['content'].startswith('[Uploaded file:') or 
            msg['content'].startswith('[Bestand:')
        ))
    ]
    
    removed_count = initial_length - len(conversation_history)
    
    return jsonify({
        'success': True, 
        'message': f'Removed {removed_count} file entries from chat history',
        'removed_count': removed_count
    })

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Add user message to history
    conversation_history.append({'role': 'user', 'content': user_message})

    # Optional: limit history to avoid excessive token usage
    MAX_TURNS = 10  # 10 exchanges = 20 messages
    if len(conversation_history) > MAX_TURNS * 2:
        conversation_history = conversation_history[-(MAX_TURNS * 2):]

    # Build full prompt with context + history
    prompt = build_prompt(conversation_history[:-1], user_message, SYSTEM_CONTEXT)

    def generate():
        full_response = ""
        try:
            for fragment in model.respond_stream(prompt):
                content = fragment.content
                full_response += content
                try:
                    yield f"data: {json.dumps({'content': content})}\n\n"
                except ClientDisconnected:
                    # Client disconnected, so we stop generating
                    break

            # Only save to history if generation was not interrupted
            if len(full_response) > 0:
                conversation_history.append({'role': 'model', 'content': full_response})
                yield f"data: {json.dumps({'done': True, 'full_response': full_response})}\n\n"
        except Exception as e:
            # Check if the exception was due to client disconnect
            if "client" in str(e).lower() or "disconnect" in str(e).lower():
                # Client disconnected, stop generating
                pass
            else:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/reset', methods=['POST'])
def reset_chat():
    global conversation_history
    conversation_history = []
    return jsonify({'message': 'Chat history reset successfully'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)