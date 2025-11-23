from flask import Flask, render_template, request, jsonify, session, Response
from werkzeug.utils import secure_filename
import os
import asyncio
import json
import uuid
from datetime import datetime
import threading
import time
from queue import Queue
import logging
from functools import wraps
import base64
import tempfile
import subprocess
import shutil
import re
import requests
from PIL import Image
import io
import numpy as np

# Import your existing processors
import sys
sys.path.append('models')
sys.path.append('.')

# Import the Agent Orchestrator
try:
    from agent_orchestrator import AgentOrchestratorAPI
    AGENT_ORCHESTRATOR_AVAILABLE = True
    print("✅ Agent Orchestrator loaded successfully")
except ImportError as e:
    AGENT_ORCHESTRATOR_AVAILABLE = False
    print(f"⚠️ Agent Orchestrator not available: {e}")

# Fallback imports for individual processors
try:
    from voice_processor import ModernVoiceProcessor, VoiceProcessorFactory
    VOICE_PROCESSOR_AVAILABLE = True
    print("✅ Voice processor loaded successfully")
except ImportError as e:
    VOICE_PROCESSOR_AVAILABLE = False
    print(f"⚠️ Voice processor not available: {e}")

try:
    from text_processor import EnhancedFreeChatBot
    TEXT_PROCESSOR_AVAILABLE = True
    print("✅ EnhancedFreeChatBot imported successfully")
except ImportError as e:
    TEXT_PROCESSOR_AVAILABLE = False
    print(f"⚠️ Text processor import failed: {e}")

try:
    from image_processor import OptimizedImageProcessor
    IMAGE_PROCESSOR_AVAILABLE = True
    print("✅ Image processor loaded successfully")
except ImportError as e:
    IMAGE_PROCESSOR_AVAILABLE = False
    print(f"⚠️ Image processor not available: {e}")

app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage for processing jobs
processing_jobs = {}
processing_queue = Queue()

# Chat sessions storage
chat_sessions = {}

# Initialize Agent Orchestrator or fallback processors
agent_api = None
text_processor = None
voice_processor = None
image_processor = None

def initialize_agents():
    """Initialize Agent Orchestrator AND individual processors"""
    global agent_api, text_processor, voice_processor, image_processor
    
    # Initialize Agent Orchestrator
    if AGENT_ORCHESTRATOR_AVAILABLE:
        try:
            agent_api = AgentOrchestratorAPI()
            print("🤖 Agent Orchestrator created")
        except Exception as e:
            print(f"❌ Failed to create Agent Orchestrator: {e}")
            agent_api = None
    
    print("🔄 Initializing individual processors...")
    
    # Text processor
    if TEXT_PROCESSOR_AVAILABLE:
        try:
            text_processor = EnhancedFreeChatBot()
            print("✅ Text processor initialized")
        except Exception as e:
            print(f"❌ Text processor failed: {e}")
    
    # Voice processor
    if VOICE_PROCESSOR_AVAILABLE:
        try:
            voice_processor = VoiceProcessorFactory.create_production_processor()
            print("✅ Voice processor initialized")
        except Exception as e:
            print(f"❌ Voice processor failed: {e}")
    
    # Image processor
    if IMAGE_PROCESSOR_AVAILABLE:
        try:
            image_processor = OptimizedImageProcessor()
            print("✅ Image processor initialized")
        except Exception as e:
            print(f"❌ Image processor failed: {e}")

# Initialize on startup
initialize_agents()

class ProcessingJob:
    def __init__(self, job_id, job_type, content, user_session, language='en'):
        self.job_id = job_id
        self.job_type = job_type
        self.content = content
        self.user_session = user_session
        self.language = language
        self.status = 'queued'
        self.progress = 0
        self.result = None
        self.error_message = None
        self.created_at = datetime.now()
        self.completed_at = None

class ChatSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.messages = []
        self.created_at = datetime.now()
        self.title = "New Chat"
        self.updated_at = datetime.now()

def convert_webm_to_wav(webm_path, wav_path):
    """Convert WebM audio to WAV using ffmpeg"""
    try:
        if not shutil.which('ffmpeg'):
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(webm_path, format="webm")
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(wav_path, format="wav")
                return True
            except ImportError:
                logger.error("Neither ffmpeg nor pydub available")
                return False
        
        cmd = [
            'ffmpeg', '-y', '-i', webm_path,
            '-ar', '16000', '-ac', '1', '-f', 'wav',
            wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
            
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return False

def download_and_convert_image(image_url):
    """Download image from URL and convert to base64"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(image_url, timeout=15, headers=headers)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            
            if image.width > 800 or image.height > 600:
                image.thumbnail((800, 600), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=90)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        logger.error(f"Image download error: {e}")
    return None

def format_code_blocks(text):
    """Format code blocks with proper syntax highlighting"""
    code_pattern = r'```(\w+)?\n?(.*?)```'
    
    def replace_code_block(match):
        language = match.group(1) or 'text'
        code = match.group(2).strip()
        
        lines = code.split('\n')
        if lines:
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]
        
        formatted_code = '\n'.join(lines)
        
        return f'<div class="code-block" data-language="{language}">' \
               f'<div class="code-header">' \
               f'<span class="code-language">{language.upper()}</span>' \
               f'<div class="code-actions">' \
               f'<button class="view-toggle active" data-view="rendered" onclick="toggleCodeView(this)">👁️</button>' \
               f'<button class="view-toggle" data-view="code" onclick="toggleCodeView(this)">&lt;/&gt;</button>' \
               f'<button class="copy-btn" onclick="copyCode(this)">Copy</button>' \
               f'</div>' \
               f'</div>' \
               f'<div class="code-content">' \
               f'<div class="code-rendered">{formatted_code}</div>' \
               f'<pre class="code-raw" style="display: none;"><code class="language-{language}">{formatted_code}</code></pre>' \
               f'</div>' \
               f'</div>'
    
    return re.sub(code_pattern, replace_code_block, text, flags=re.DOTALL)

def format_response_text(text):
    """Format text for better presentation"""
    text = format_code_blocks(text)
    
    image_pattern = r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp|svg)'
    def replace_image(match):
        image_url = match.group()
        base64_image = download_and_convert_image(image_url)
        if base64_image:
            return f'<div class="image-container"><img src="{base64_image}" class="generated-image" alt="Generated Image" onclick="app.showImageFullscreen(\'{base64_image}\')"></div>'
        return f'<div class="image-container"><img src="{image_url}" class="generated-image" alt="Generated Image" onclick="app.showImageFullscreen(\'{image_url}\')"></div>'
    
    text = re.sub(image_pattern, replace_image, text)
    
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append('<br>')
            continue
            
        if line.startswith('###'):
            line = f'<h4 class="response-header h4">{line[3:].strip()}</h4>'
        elif line.startswith('##'):
            line = f'<h3 class="response-header h3">{line[2:].strip()}</h3>'
        elif line.startswith('#'):
            line = f'<h2 class="response-header h2">{line[1:].strip()}</h2>'
        elif re.match(r'^\d+\.', line):
            line = f'<div class="numbered-item">{line}</div>'
        elif line.startswith('- ') or line.startswith('• '):
            line = f'<div class="bullet-item">{line}</div>'
        else:
            if line and not line.startswith('<'):
                line = f'<p class="response-paragraph">{line}</p>'
        
        line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
        line = re.sub(r'\*(.*?)\*', r'<em>\1</em>', line)
        line = re.sub(r'`([^`]+)`', r'<code class="inline-code">\1</code>', line)
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def format_image_analysis_result(analysis_result):
    """Format image analysis result"""
    try:
        if isinstance(analysis_result, dict):
            objects = analysis_result.get('objects', [])
            scene_type = analysis_result.get('scene_type', 'unknown')
            dominant_colors = analysis_result.get('dominant_colors', [])
            emotions = analysis_result.get('emotions', [])
            text_content = analysis_result.get('text_content', '')
            confidence_score = analysis_result.get('confidence_score', 0.0)
            suggestions = analysis_result.get('suggestions', [])
            metadata = analysis_result.get('metadata', {})
            processing_time = analysis_result.get('processing_time', 0.0)
            enhanced_image_base64 = analysis_result.get('enhanced_image_base64', None)
        else:
            objects = getattr(analysis_result, 'objects', [])
            scene_type = getattr(analysis_result, 'scene_type', 'unknown')
            dominant_colors = getattr(analysis_result, 'dominant_colors', [])
            emotions = getattr(analysis_result, 'emotions', [])
            text_content = getattr(analysis_result, 'text_content', '')
            confidence_score = getattr(analysis_result, 'confidence_score', 0.0)
            suggestions = getattr(analysis_result, 'suggestions', [])
            metadata = getattr(analysis_result, 'metadata', {})
            processing_time = getattr(analysis_result, 'processing_time', 0.0)
            enhanced_image_base64 = getattr(analysis_result, 'enhanced_image_base64', None)
        
        enhanced_image_html = ""
        if enhanced_image_base64:
            enhanced_image_html = f"""
            <div class="enhanced-image-section">
                <h4><i class="fas fa-magic"></i> AI-Enhanced Image</h4>
                <div class="enhanced-image-container">
                    <img src="{enhanced_image_base64}" class="enhanced-image-display" alt="AI Enhanced Image" 
                         onclick="app.showImageFullscreen('{enhanced_image_base64}')">
                </div>
            </div>
            """
        
        html = f"""
        <div class="image-analysis-result">
            <div class="analysis-header">
                <h3><i class="fas fa-eye"></i> Image Analysis Results</h3>
                <div class="confidence-badge">
                    <span class="confidence-score">{confidence_score:.1%} Confidence</span>
                </div>
            </div>
            
            {enhanced_image_html}
            
            <div class="analysis-grid">
                <div class="analysis-section">
                    <h4><i class="fas fa-cube"></i> Objects Detected</h4>
                    <div class="tag-container">
        """
        
        for obj in objects[:10]:
            html += f'<span class="analysis-tag object-tag">{str(obj).replace("_", " ").title()}</span>'
        
        html += f"""
                    </div>
                </div>
                
                <div class="analysis-section">
                    <h4><i class="fas fa-map"></i> Scene Type</h4>
                    <div class="scene-type">
                        <span class="scene-badge">{str(scene_type).replace("_", " ").title()}</span>
                    </div>
                </div>
                
                <div class="analysis-section">
                    <h4><i class="fas fa-palette"></i> Dominant Colors</h4>
                    <div class="tag-container">
        """
        
        for color in dominant_colors[:6]:
            html += f'<span class="analysis-tag color-tag">{str(color).replace("_", " ").title()}</span>'
        
        html += f"""
                    </div>
                </div>
                
                <div class="analysis-section">
                    <h4><i class="fas fa-smile"></i> Emotions Detected</h4>
                    <div class="tag-container">
        """
        
        for emotion in emotions[:5]:
            html += f'<span class="analysis-tag emotion-tag">{str(emotion).replace("_", " ").title()}</span>'
        
        html += f"""
                    </div>
                </div>
            </div>
        """
        
        if text_content and len(str(text_content).strip()) > 0:
            text_lines = str(text_content).split('\n')
            formatted_text = '<br>'.join([line.strip() for line in text_lines if line.strip()])
            html += f"""
            <div class="analysis-section text-extraction-section">
                <h4><i class="fas fa-font"></i> Extracted Text</h4>
                <div class="extracted-text-box">
                    <pre class="extracted-text-content">{formatted_text}</pre>
                </div>
            </div>
            """
        
        if metadata and 'image_quality' in metadata:
            quality = metadata['image_quality']
            html += f"""
            <div class="analysis-section quality-metrics-section">
                <h4><i class="fas fa-chart-bar"></i> Quality Metrics</h4>
                <div class="quality-metrics-grid">
                    <div class="quality-metric">
                        <span class="metric-label">Sharpness:</span>
                        <span class="metric-value">{quality.get('sharpness', 0):.1f}</span>
                    </div>
                    <div class="quality-metric">
                        <span class="metric-label">Brightness:</span>
                        <span class="metric-value">{quality.get('brightness', 0):.1f}</span>
                    </div>
                    <div class="quality-metric">
                        <span class="metric-label">Contrast:</span>
                        <span class="metric-value">{quality.get('contrast', 0):.1f}</span>
                    </div>
                    <div class="quality-metric">
                        <span class="metric-label">Quality Score:</span>
                        <span class="metric-value">{quality.get('quality_score', 0):.3f}</span>
                    </div>
                </div>
            </div>
            """
        
        if suggestions:
            html += f"""
            <div class="analysis-section suggestions-section">
                <h4><i class="fas fa-lightbulb"></i> AI Suggestions</h4>
                <ul class="suggestions-list">
            """
            
            for suggestion in suggestions:
                html += f'<li class="suggestion-item">{str(suggestion)}</li>'
            
            html += """
                </ul>
            </div>
            """
        
        pipeline_name = metadata.get('selected_pipeline', 'Standard') if isinstance(metadata, dict) else 'Standard'
        html += f"""
            <div class="analysis-footer">
                <div class="processing-info">
                    <span class="processing-time">⏱️ Processed in {processing_time:.2f}s</span>
                    <span class="pipeline-used">🔧 Pipeline: {pipeline_name}</span>
                </div>
            </div>
        </div>
        """
        
        return html
        
    except Exception as e:
        logger.error(f"Error formatting image analysis: {e}")
        return f"""
        <div class="image-analysis-result error">
            <p>Error formatting results: {str(e)}</p>
        </div>
        """

@app.route('/')
def index():
    """Main chatbot interface"""
    return render_template('index.html')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all chat sessions"""
    sessions_list = []
    for session_id, session in chat_sessions.items():
        sessions_list.append({
            'id': session_id,
            'title': session.title,
            'created_at': session.created_at.isoformat(),
            'updated_at': session.updated_at.isoformat(),
            'message_count': len(session.messages)
        })
    
    sessions_list.sort(key=lambda x: x['updated_at'], reverse=True)
    return jsonify({'sessions': sessions_list})

@app.route('/api/sessions', methods=['POST'])
def create_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = ChatSession(session_id)
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'title': 'New Chat'
    })

@app.route('/api/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    """Get messages for a specific session"""
    if session_id not in chat_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = chat_sessions[session_id]
    return jsonify({
        'session_id': session_id,
        'title': session.title,
        'messages': session.messages
    })

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return jsonify({'success': True})
    return jsonify({'error': 'Session not found'}), 404

def create_fresh_prompt(user_message):
    """Create a fresh prompt"""
    clean_message = user_message.strip()
    fresh_prompt = f"""Please provide a direct response to: {clean_message}

Focus only on answering the current question."""
    
    return fresh_prompt

@app.route('/api/process/text', methods=['POST'])
def process_text():
    """Process text input - ALWAYS USE STREAMING"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        if session_id and session_id in chat_sessions:
            chat_sessions[session_id].messages.append({
                'type': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat()
            })
            chat_sessions[session_id].updated_at = datetime.now()
            
            if len(chat_sessions[session_id].messages) == 1:
                title = user_message[:50] + "..." if len(user_message) > 50 else user_message
                chat_sessions[session_id].title = title
        
        logger.info(f"STREAMING MODE for: {user_message[:50]}")
        return Response(stream_text_response(user_message, session_id), mimetype='text/event-stream')
        
    except Exception as e:
        logger.error(f"Text processing error: {str(e)}")
        return jsonify({'error': 'Failed to process text', 'details': str(e)}), 500

def stream_text_response(user_message, session_id=None):
    """Stream text response WORD BY WORD"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response_text = ""
            fresh_prompt = create_fresh_prompt(user_message)
            
            if agent_api:
                try:
                    response = loop.run_until_complete(
                        agent_api.chat(fresh_prompt, None)
                    )
                    
                    if response.get('success', False):
                        response_text = response['content']
                except Exception as e:
                    logger.warning(f"Agent failed: {e}")
                    response_text = ""
            
            if not response_text and text_processor:
                if hasattr(text_processor, 'conversation_history'):
                    original_history = text_processor.conversation_history[:]
                    text_processor.conversation_history = []
                
                response_text = loop.run_until_complete(text_processor.process_message(fresh_prompt))
                
                if hasattr(text_processor, 'conversation_history'):
                    text_processor.conversation_history = original_history[-2:] if len(original_history) > 2 else original_history
            
            if not response_text:
                yield f"data: {json.dumps({'error': 'AI service unavailable'})}\n\n"
                return
            
            formatted_response = format_response_text(str(response_text))
            
            if session_id and session_id in chat_sessions:
                chat_sessions[session_id].messages.append({
                    'type': 'ai',
                    'content': formatted_response,
                    'timestamp': datetime.now().isoformat()
                })
            
            words = formatted_response.split(' ')
            
            for i, word in enumerate(words):
                if i == 0:
                    yield f"data: {json.dumps({'word': word, 'complete': False})}\n\n"
                else:
                    yield f"data: {json.dumps({'word': ' ' + word, 'complete': False})}\n\n"
                
                time.sleep(0.03)
            
            yield f"data: {json.dumps({'word': '', 'complete': True})}\n\n"
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"STREAMING ERROR: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.route('/api/process/image', methods=['POST'])
def process_image():
    """Process uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        if not ('.' in image_file.filename and 
                image_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid format'}), 400
        
        image_file.seek(0, os.SEEK_END)
        file_size = image_file.tell()
        image_file.seek(0)
        
        if file_size > 20 * 1024 * 1024:
            return jsonify({'error': 'File too large'}), 400
        
        job_id = str(uuid.uuid4())
        session_id = request.form.get('session_id')
        
        filename = secure_filename(image_file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        temp_filename = f"{job_id}.{file_extension}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        image_file.save(temp_path)
        
        job = ProcessingJob(job_id, 'image_analysis', temp_path, session_id)
        processing_jobs[job_id] = job
        
        processing_queue.put(job_id)
        
        if not hasattr(app, 'processor_thread') or not app.processor_thread.is_alive():
            app.processor_thread = threading.Thread(target=process_jobs_queue, daemon=True)
            app.processor_thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Image analysis started'
        })
        
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return jsonify({'error': 'Failed to process image', 'details': str(e)}), 500

@app.route('/api/process/image-with-text', methods=['POST'])
def process_image_with_text():
    """Process image with text"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400
        
        image_file = request.files['image']
        user_question = request.form.get('question', '').strip()
        session_id = request.form.get('session_id')
        
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not user_question:
            return jsonify({'error': 'Question required'}), 400
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        if not ('.' in image_file.filename and 
                image_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid format'}), 400
        
        image_file.seek(0, os.SEEK_END)
        file_size = image_file.tell()
        image_file.seek(0)
        
        if file_size > 20 * 1024 * 1024:
            return jsonify({'error': 'File too large'}), 400
        
        job_id = str(uuid.uuid4())
        
        filename = secure_filename(image_file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        temp_filename = f"{job_id}.{file_extension}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        image_file.save(temp_path)
        
        job_content = {
            'image_path': temp_path,
            'question': user_question
        }
        job = ProcessingJob(job_id, 'image_text_analysis', job_content, session_id)
        processing_jobs[job_id] = job
        
        processing_queue.put(job_id)
        
        if not hasattr(app, 'processor_thread') or not app.processor_thread.is_alive():
            app.processor_thread = threading.Thread(target=process_jobs_queue, daemon=True)
            app.processor_thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Analysis started'
        })
        
    except Exception as e:
        logger.error(f"Image-text error: {str(e)}")
        return jsonify({'error': 'Failed to process', 'details': str(e)}), 500

@app.route('/api/status/<job_id>')
def get_job_status(job_id):
    """Get job status"""
    job = processing_jobs.get(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    response = {
        'job_id': job_id,
        'status': job.status,
        'progress': job.progress,
        'type': job.job_type,
        'created_at': job.created_at.isoformat(),
    }
    
    if job.completed_at:
        response['completed_at'] = job.completed_at.isoformat()
    
    if job.error_message:
        response['error'] = job.error_message
    
    if job.result and job.status == 'completed':
        response['result'] = job.result
    
    return jsonify(response)

@app.route('/api/process/voice', methods=['POST'])
def process_voice():
    """Process voice input (Voice Processing mode - transcription only)"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'en')
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        job_id = str(uuid.uuid4())
        session_id = request.form.get('session_id')
        
        filename = secure_filename(audio_file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'webm'
        temp_filename = f"{job_id}.{file_extension}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        audio_file.save(temp_path)
        
        job_content = {
            'audio_path': temp_path,
            'language': language
        }
        job = ProcessingJob(job_id, 'voice_to_text', job_content, session_id, language)
        processing_jobs[job_id] = job
        
        processing_queue.put(job_id)
        
        if not hasattr(app, 'processor_thread') or not app.processor_thread.is_alive():
            app.processor_thread = threading.Thread(target=process_jobs_queue, daemon=True)
            app.processor_thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Voice processing started'
        })
        
    except Exception as e:
        logger.error(f"Voice processing error: {str(e)}")
        return jsonify({'error': 'Failed to process voice', 'details': str(e)}), 500

@app.route('/api/process/voice-conversation', methods=['POST'])
def process_voice_conversation():
    """FIXED: Process voice conversation - Voice → AI → Voice + Text (14 languages)"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'en')
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        logger.info(f"🎙️ Voice conversation, language: {language}")
        
        job_id = str(uuid.uuid4())
        session_id = request.form.get('session_id')
        
        filename = secure_filename(audio_file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'webm'
        temp_filename = f"{job_id}.{file_extension}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        audio_file.save(temp_path)
        
        job_content = {
            'audio_path': temp_path,
            'language': language
        }
        job = ProcessingJob(job_id, 'voice_conversation', job_content, session_id, language)
        processing_jobs[job_id] = job
        
        processing_queue.put(job_id)
        
        if not hasattr(app, 'processor_thread') or not app.processor_thread.is_alive():
            app.processor_thread = threading.Thread(target=process_jobs_queue, daemon=True)
            app.processor_thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Voice conversation started'
        })
        
    except Exception as e:
        logger.error(f"Voice conversation error: {str(e)}")
        return jsonify({'error': 'Failed to process', 'details': str(e)}), 500

@app.route('/api/process/text-to-speech', methods=['POST'])
def text_to_speech():
    """FIXED: Convert text to speech (14 languages)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        text = data.get('text', '').strip()
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({'error': 'Text required'}), 400
        
        logger.info(f"🔊 TTS: language={language}")
        
        job_id = str(uuid.uuid4())
        session_id = data.get('session_id')
        
        job_content = {
            'text': text,
            'language': language
        }
        job = ProcessingJob(job_id, 'text_to_speech', job_content, session_id, language)
        processing_jobs[job_id] = job
        
        processing_queue.put(job_id)
        
        if not hasattr(app, 'processor_thread') or not app.processor_thread.is_alive():
            app.processor_thread = threading.Thread(target=process_jobs_queue, daemon=True)
            app.processor_thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'TTS started'
        })
        
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        return jsonify({'error': 'Failed to process', 'details': str(e)}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_chat_history():
    """Clear chat history"""
    try:
        data = request.get_json()
        session_id = data.get('session_id') if data else None
        
        if session_id and session_id in chat_sessions:
            chat_sessions[session_id].messages = []
        else:
            if text_processor and hasattr(text_processor, 'conversation_history'):
                text_processor.conversation_history = []
        
        return jsonify({
            'success': True,
            'message': 'Chat history cleared'
        })
        
    except Exception as e:
        logger.error(f"Clear history error: {str(e)}")
        return jsonify({'error': 'Failed to clear history'}), 500

def process_jobs_queue():
    """Background thread to process jobs"""
    while True:
        try:
            if not processing_queue.empty():
                job_id = processing_queue.get()
                process_single_job(job_id)
            else:
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Queue error: {str(e)}")
            time.sleep(2)

def process_single_job(job_id):
    """Process a single job"""
    job = processing_jobs.get(job_id)
    if not job:
        return
    
    try:
        job.status = 'processing'
        job.progress = 10
        
        if job.job_type == 'image_analysis':
            process_image_analysis_job_direct(job)
        elif job.job_type == 'image_text_analysis':
            process_image_text_analysis_job_direct(job)
        elif job.job_type == 'voice_to_text':
            process_voice_to_text_job(job)
        elif job.job_type == 'voice_conversation':
            process_voice_conversation_job(job)
        elif job.job_type == 'text_to_speech':
            process_text_to_speech_job(job)
            
    except Exception as e:
        logger.error(f"Job error {job_id}: {str(e)}")
        job.status = 'error'
        job.error_message = str(e)
        job.completed_at = datetime.now()

def process_image_analysis_job_direct(job):
    """Process image analysis"""
    try:
        job.progress = 25
        image_path = job.content
        
        if not os.path.exists(image_path):
            raise Exception(f"Image not found")
        
        job.progress = 50
        
        if not image_processor:
            raise Exception("Image processor not available")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            analysis_result = loop.run_until_complete(
                image_processor.analyze_image_comprehensive(image_path)
            )
        finally:
            loop.close()
        
        job.progress = 85
        formatted_result = format_image_analysis_result(analysis_result)
        
        enhanced_image_base64 = None
        if isinstance(analysis_result, dict):
            enhanced_image_base64 = analysis_result.get('enhanced_image_base64')
        elif hasattr(analysis_result, 'enhanced_image_base64'):
            enhanced_image_base64 = analysis_result.enhanced_image_base64
        
        if isinstance(analysis_result, dict):
            job.result = {
                'analysis': analysis_result,
                'formatted_html': formatted_result,
                'enhanced_image_base64': enhanced_image_base64,
                'type': 'image_analysis'
            }
        else:
            job.result = {
                'analysis': {
                    'objects': getattr(analysis_result, 'objects', []),
                    'scene_type': getattr(analysis_result, 'scene_type', 'unknown'),
                    'dominant_colors': getattr(analysis_result, 'dominant_colors', []),
                    'emotions': getattr(analysis_result, 'emotions', []),
                    'text_content': getattr(analysis_result, 'text_content', ''),
                    'confidence_score': getattr(analysis_result, 'confidence_score', 0.0),
                    'suggestions': getattr(analysis_result, 'suggestions', []),
                    'metadata': getattr(analysis_result, 'metadata', {}),
                    'processing_time': getattr(analysis_result, 'processing_time', 0.0)
                },
                'formatted_html': formatted_result,
                'enhanced_image_base64': enhanced_image_base64,
                'type': 'image_analysis'
            }
        
        job.progress = 100
        job.status = 'completed'
        job.completed_at = datetime.now()
        
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        job.status = 'error'
        job.error_message = str(e)
        job.completed_at = datetime.now()
    finally:
        try:
            if os.path.exists(job.content):
                os.unlink(job.content)
        except:
            pass

def process_image_text_analysis_job_direct(job):
    """Process image with text"""
    try:
        job.progress = 20
        content = job.content
        image_path = content['image_path']
        user_question = content['question']
        
        if not os.path.exists(image_path):
            raise Exception(f"Image not found")
        
        job.progress = 40
        
        if not image_processor:
            raise Exception("Image processor not available")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            analysis_result = loop.run_until_complete(
                image_processor.analyze_image_comprehensive(image_path)
            )
            
            job.progress = 65
            
            if text_processor:
                clean_prompt = f"""Based on this image, answer: {user_question}

Image contains:
- Objects: {', '.join(getattr(analysis_result, 'objects', [])[:5])}
- Scene: {getattr(analysis_result, 'scene_type', 'unknown')}

Answer directly."""
                
                ai_response = loop.run_until_complete(
                    text_processor._general_chat(clean_prompt)
                )
            else:
                ai_response = f"Based on the image: {', '.join(getattr(analysis_result, 'objects', [])[:5])}"
            
            job.progress = 85
            
            formatted_analysis = format_image_analysis_result(analysis_result)
            formatted_ai_response = format_response_text(ai_response)
            
            enhanced_image_base64 = None
            if hasattr(analysis_result, 'enhanced_image_base64'):
                enhanced_image_base64 = analysis_result.enhanced_image_base64
            
            combined_html = f"""
            <div class="combined-analysis">
                {formatted_analysis}
                <div class="ai-response-section">
                    <h3><i class="fas fa-robot"></i> AI Response</h3>
                    <div class="user-question-display">
                        <strong>Question:</strong> {user_question}
                    </div>
                    <div class="ai-response-content">
                        {formatted_ai_response}
                    </div>
                </div>
            </div>
            """
            
            job.result = {
                'image_analysis': {
                    'objects': getattr(analysis_result, 'objects', []),
                    'scene_type': getattr(analysis_result, 'scene_type', 'unknown'),
                    'dominant_colors': getattr(analysis_result, 'dominant_colors', []),
                    'emotions': getattr(analysis_result, 'emotions', []),
                    'text_content': getattr(analysis_result, 'text_content', ''),
                    'confidence_score': getattr(analysis_result, 'confidence_score', 0.0),
                    'suggestions': getattr(analysis_result, 'suggestions', []),
                    'metadata': getattr(analysis_result, 'metadata', {}),
                    'processing_time': getattr(analysis_result, 'processing_time', 0.0)
                },
                'user_question': user_question,
                'ai_response': ai_response,
                'formatted_html': combined_html,
                'enhanced_image_base64': enhanced_image_base64,
                'agent_response': False,
                'type': 'image_text_analysis'
            }
            
        finally:
            loop.close()
        
        job.progress = 100
        job.status = 'completed'
        job.completed_at = datetime.now()
        
    except Exception as e:
        logger.error(f"Image-text error: {str(e)}")
        job.status = 'error'
        job.error_message = str(e)
        job.completed_at = datetime.now()
    finally:
        try:
            if os.path.exists(content['image_path']):
                os.unlink(content['image_path'])
        except:
            pass

def process_voice_to_text_job(job):
    """Process voice to text"""
    try:
        job.progress = 20
        
        content = job.content
        audio_path = content['audio_path']
        language = content.get('language', 'en')
        
        if not os.path.exists(audio_path):
            raise Exception(f"Audio not found")
        
        job.progress = 40
        
        wav_path = audio_path
        if not audio_path.endswith('.wav'):
            wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
            success = convert_webm_to_wav(audio_path, wav_path)
            if not success:
                raise Exception("Audio conversion failed")
        
        job.progress = 60
        
        if not voice_processor:
            raise Exception("Voice processor not available")
        
        import soundfile as sf
        audio_data, sample_rate = sf.read(wav_path, dtype='float32')
        
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        
        job.progress = 80
        
        voice_processor.config.language = language
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            transcription = loop.run_until_complete(
                voice_processor.speech_to_text_advanced(audio_data)
            )
            
            if transcription.get('error'):
                error_msg = transcription['error']
                if 'too short' in error_msg.lower():
                    job.result = {
                        'transcription': '',
                        'language': language,
                        'confidence': 0.0,
                        'type': 'voice_to_text',
                        'error': 'Audio too short',
                        'user_message': '🎤 Recording too short'
                    }
                elif 'too quiet' in error_msg.lower():
                    job.result = {
                        'transcription': '',
                        'language': language,
                        'confidence': 0.0,
                        'type': 'voice_to_text',
                        'error': 'Audio too quiet',
                        'user_message': '🔇 Speak louder'
                    }
                else:
                    job.result = {
                        'transcription': '',
                        'language': language,
                        'confidence': 0.0,
                        'type': 'voice_to_text',
                        'error': error_msg,
                        'user_message': f'⚠️ {error_msg}'
                    }
            elif not transcription.get('text') or not transcription['text'].strip():
                job.result = {
                    'transcription': '',
                    'language': language,
                    'confidence': 0.0,
                    'type': 'voice_to_text',
                    'error': 'No speech detected',
                    'user_message': '🎤 No speech detected'
                }
            else:
                job.result = {
                    'transcription': transcription.get('text', ''),
                    'language': language,
                    'confidence': transcription.get('confidence', 0.0),
                    'emotion': transcription.get('emotion', {}),
                    'processing_time': transcription.get('processing_time', 0.0),
                    'type': 'voice_to_text'
                }
            
        finally:
            loop.close()
        
        job.progress = 100
        job.status = 'completed'
        job.completed_at = datetime.now()
        
    except Exception as e:
        logger.error(f"Voice-to-text error: {str(e)}")
        job.status = 'error'
        job.error_message = str(e)
        job.completed_at = datetime.now()
    finally:
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            if 'wav_path' in locals() and wav_path != audio_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except:
            pass

def process_voice_conversation_job(job):
    """FIXED: Process voice conversation - Voice → AI → Voice (14 languages) + Text"""
    try:
        job.progress = 10
        
        content = job.content
        audio_path = content['audio_path']
        language = content.get('language', 'en')
        
        if not os.path.exists(audio_path):
            raise Exception(f"Audio not found")
        
        logger.info(f"🎙️ Voice conversation: {language}")
        
        # Step 1: Convert to WAV
        job.progress = 20
        wav_path = audio_path
        if not audio_path.endswith('.wav'):
            wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
            success = convert_webm_to_wav(audio_path, wav_path)
            if not success:
                raise Exception("Audio conversion failed")
        
        # Step 2: Transcribe
        job.progress = 35
        
        if not voice_processor:
            raise Exception("Voice processor not available")
        
        import soundfile as sf
        audio_data, sample_rate = sf.read(wav_path, dtype='float32')
        
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        
        voice_processor.config.language = language
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            transcription = loop.run_until_complete(
                voice_processor.speech_to_text_advanced(audio_data)
            )
            
            if transcription.get('error') or not transcription.get('text'):
                raise Exception(transcription.get('error', 'No speech detected'))
            
            user_text = transcription.get('text', '').strip()
            if not user_text:
                raise Exception("Empty transcription")
            
            logger.info(f"✅ Transcribed: {user_text}")
            
            # Step 3: Get AI response
            job.progress = 55
            
            ai_response_text = ""
            fresh_prompt = create_fresh_prompt(user_text)
            
            if agent_api:
                try:
                    response = loop.run_until_complete(
                        agent_api.chat(fresh_prompt, None)
                    )
                    if response.get('success', False):
                        ai_response_text = response['content']
                except Exception as e:
                    logger.warning(f"Agent failed: {e}")
            
            if not ai_response_text and text_processor:
                ai_response_text = loop.run_until_complete(
                    text_processor._general_chat(fresh_prompt)
                )
            
            if not ai_response_text:
                ai_response_text = "I'm sorry, I couldn't process your request."
            
            logger.info(f"🤖 AI Response: {ai_response_text[:100]}...")
            
            # Step 4: Generate TTS in selected language
            job.progress = 75
            
            language_map = {
                'en-US': 'en', 'en-GB': 'en', 'en': 'en',
                'es-ES': 'es', 'es': 'es',
                'fr-FR': 'fr', 'fr': 'fr',
                'de-DE': 'de', 'de': 'de',
                'it-IT': 'it', 'it': 'it',
                'pt-PT': 'pt', 'pt-BR': 'pt', 'pt': 'pt',
                'ru-RU': 'ru', 'ru': 'ru',
                'ja-JP': 'ja', 'ja': 'ja',
                'ko-KR': 'ko', 'ko': 'ko',
                'zh-CN': 'zh-CN', 'zh-TW': 'zh-TW', 'zh': 'zh-CN',
                'ar-SA': 'ar', 'ar': 'ar',
                'hi-IN': 'hi', 'hi': 'hi',
                'ur-PK': 'ur', 'ur': 'ur',
                'th-TH': 'th', 'th': 'th'
            }
            
            gtts_language = language_map.get(language, 'en')
            logger.info(f"🔊 TTS language: {gtts_language}")
            
            from gtts import gTTS
            import tempfile
            
            temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_mp3_path = temp_mp3.name
            temp_mp3.close()
            
            try:
                clean_text = ai_response_text.replace('<', '').replace('>', '').replace('*', '')
                
                tts = gTTS(text=clean_text, lang=gtts_language, slow=False)
                tts.save(temp_mp3_path)
                
                with open(temp_mp3_path, 'rb') as f:
                    audio_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                job.progress = 95
                
                formatted_response = format_response_text(ai_response_text)
                
                job.result = {
                    'transcription': user_text,
                    'ai_response': formatted_response,
                    'audio_base64': audio_base64,
                    'audio_format': 'mp3',
                    'language': gtts_language,
                    'original_language': language,
                    'type': 'voice_conversation',
                    'success': True
                }
                
                logger.info(f"✅ Voice conversation completed")
                
            finally:
                if os.path.exists(temp_mp3_path):
                    os.unlink(temp_mp3_path)
        
        finally:
            loop.close()
        
        job.progress = 100
        job.status = 'completed'
        job.completed_at = datetime.now()
        
    except Exception as e:
        logger.error(f"❌ Voice conversation error: {str(e)}")
        job.status = 'error'
        job.error_message = str(e)
        job.completed_at = datetime.now()
    finally:
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            if 'wav_path' in locals() and wav_path != audio_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except:
            pass

def process_text_to_speech_job(job):
    """FIXED: Process TTS with 14 languages"""
    try:
        job.progress = 20
        
        content = job.content
        text = content['text']
        language = content.get('language', 'en')
        
        logger.info(f"🔊 TTS: language={language}")
        
        job.progress = 60
        
        language_map = {
            'en-US': 'en', 'en-GB': 'en', 'en': 'en',
            'es-ES': 'es', 'es': 'es',
            'fr-FR': 'fr', 'fr': 'fr',
            'de-DE': 'de', 'de': 'de',
            'it-IT': 'it', 'it': 'it',
            'pt-PT': 'pt', 'pt-BR': 'pt', 'pt': 'pt',
            'ru-RU': 'ru', 'ru': 'ru',
            'ja-JP': 'ja', 'ja': 'ja',
            'ko-KR': 'ko', 'ko': 'ko',
            'zh-CN': 'zh-CN', 'zh-TW': 'zh-TW', 'zh': 'zh-CN',
            'ar-SA': 'ar', 'ar': 'ar',
            'hi-IN': 'hi', 'hi': 'hi',
            'ur-PK': 'ur', 'ur': 'ur',
            'th-TH': 'th', 'th': 'th'
        }
        
        gtts_language = language_map.get(language, 'en')
        
        from gtts import gTTS
        import tempfile
        
        temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_mp3_path = temp_mp3.name
        temp_mp3.close()
        
        try:
            clean_text = text.replace('<', '').replace('>', '').replace('*', '')
            
            tts = gTTS(text=clean_text, lang=gtts_language, slow=False)
            tts.save(temp_mp3_path)
            
            with open(temp_mp3_path, 'rb') as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            job.progress = 90
            
            job.result = {
                'audio_base64': audio_base64,
                'audio_format': 'mp3',
                'language': gtts_language,
                'original_language': language,
                'text_length': len(text),
                'type': 'text_to_speech',
                'success': True
            }
            
            job.progress = 100
            job.status = 'completed'
            job.completed_at = datetime.now()
            
            logger.info(f"✅ TTS completed: {gtts_language}")
            
        finally:
            if os.path.exists(temp_mp3_path):
                os.unlink(temp_mp3_path)
        
    except Exception as e:
        logger.error(f"❌ TTS error: {str(e)}")
        job.status = 'error'
        job.error_message = str(e)
        job.completed_at = datetime.now()

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    print("🚀 FIXED Enhanced Multimodal AI Chatbot")
    print("📋 Features:")
    print("   ✅ Streaming: Word-by-word")
    print("   ✅ Image Analysis: 20MB support")
    print("   ✅ Voice-to-Text: Clear errors")
    print("   ✅ Text-to-Speech: 14 languages")
    print("   ✅ Voice Conversation: Voice → AI → Voice (14 languages) + Text")
    print("   ✅ Intelligent Chat TTS: 14 languages")
    
    if agent_api:
        print("   🤖 Agent Orchestrator: ACTIVE")
    else:
        print("   ⚠️  Agent Orchestrator: FALLBACK")
    
    if image_processor:
        print("   🖼️  Image Processor: ACTIVE")
    else:
        print("   ⚠️  Image Processor: NOT AVAILABLE")
    
    if voice_processor:
        print("   🎤 Voice Processor: ACTIVE")
    else:
        print("   ⚠️  Voice Processor: NOT AVAILABLE")
    
    print("\n🌐 http://localhost:5000")
    print("✅ ALL FEATURES FIXED!")
    print("   🎙️  Natural Conversation: Voice → AI → Voice (14 languages) + Text")
    print("   🔊 Text-to-Speech: EN, ES, FR, DE, IT, PT, RU, JA, KO, ZH, AR, HI, UR, TH")
    print("   💬 Intelligent Chat TTS: Speaks in selected language")
    
    app.run(debug=True, host='0.0.0.0', port=5000)