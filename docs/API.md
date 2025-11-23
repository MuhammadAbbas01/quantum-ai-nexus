# 📡 API Documentation - Quantum AI Nexus

## Table of Contents
1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Endpoints Reference](#endpoints-reference)
4. [Request/Response Formats](#request-response-formats)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Code Examples](#code-examples)

---

## 🌐 API Overview

**Base URL**: `https://api.quantum-ai-nexus.com/v1`  
**Protocol**: HTTPS only  
**Format**: JSON  
**Versioning**: URL-based (`/v1`, `/v2`)

### Quick Start

```bash
# Health Check
curl https://api.quantum-ai-nexus.com/v1/health

# Basic Request
curl -X POST https://api.quantum-ai-nexus.com/v1/process/text \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

---

## 🔐 Authentication

### Session-Based Authentication

```javascript
// Create a new session
fetch('/api/sessions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' }
})
.then(res => res.json())
.then(data => {
  const sessionId = data.session_id;
  localStorage.setItem('session_id', sessionId);
});

// Use session in subsequent requests
const sessionId = localStorage.getItem('session_id');
fetch('/api/process/text', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "Your message",
    session_id: sessionId
  })
});
```

### API Key Authentication (Optional)

```bash
curl -X POST https://api.quantum-ai-nexus.com/v1/process/text \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

---

## 📚 Endpoints Reference

### Session Management

#### `POST /api/sessions`
Create a new chat session.

**Request**
```json
{}
```

**Response**
```json
{
  "success": true,
  "session_id": "a1b2c3d4",
  "title": "New Chat",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Status Codes**
- `200 OK` - Session created successfully
- `500 Internal Server Error` - Server error

---

#### `GET /api/sessions`
Get all chat sessions for the current user.

**Response**
```json
{
  "sessions": [
    {
      "id": "a1b2c3d4",
      "title": "Weather Discussion",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:45:00Z",
      "message_count": 12
    },
    {
      "id": "e5f6g7h8",
      "title": "Image Analysis Chat",
      "created_at": "2024-01-14T14:20:00Z",
      "updated_at": "2024-01-14T14:35:00Z",
      "message_count": 8
    }
  ]
}
```

**Status Codes**
- `200 OK` - Sessions retrieved
- `500 Internal Server Error` - Server error

---

#### `GET /api/sessions/{session_id}/messages`
Get all messages for a specific session.

**Path Parameters**
- `session_id` (string) - The session identifier

**Response**
```json
{
  "session_id": "a1b2c3d4",
  "title": "Weather Discussion",
  "messages": [
    {
      "type": "user",
      "content": "What's the weather in New York?",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "type": "ai",
      "content": "The current weather in New York is...",
      "timestamp": "2024-01-15T10:30:05Z"
    }
  ]
}
```

**Status Codes**
- `200 OK` - Messages retrieved
- `404 Not Found` - Session not found
- `500 Internal Server Error` - Server error

---

#### `DELETE /api/sessions/{session_id}`
Delete a specific session.

**Path Parameters**
- `session_id` (string) - The session identifier

**Response**
```json
{
  "success": true,
  "message": "Session deleted successfully"
}
```

**Status Codes**
- `200 OK` - Session deleted
- `404 Not Found` - Session not found
- `500 Internal Server Error` - Server error

---

### Text Processing

#### `POST /api/process/text`
Process text input and get AI response.

**Request**
```json
{
  "message": "What's the weather in London?",
  "session_id": "a1b2c3d4"
}
```

**Parameters**
- `message` (string, required) - User message (1-5000 characters)
- `session_id` (string, optional) - Session identifier for context

**Response (Streaming)**

The endpoint returns Server-Sent Events (SSE) for word-by-word streaming:

```
data: {"word": "The", "complete": false}

data: {"word": " current", "complete": false}

data: {"word": " weather", "complete": false}

data: {"word": "", "complete": true}
```

**Status Codes**
- `200 OK` - Processing successful
- `400 Bad Request` - Invalid input
- `500 Internal Server Error` - Processing error

**Example: Weather Query**
```json
// Request
{
  "message": "What's the weather in Paris?",
  "session_id": "abc123"
}

// Response (formatted)
{
  "response": "🌤️ Current Weather in Paris:\n\nTemperature: 18°C\nCondition: Partly Cloudy\nHumidity: 65%\nWind Speed: 12 km/h"
}
```

**Example: Financial Query**
```json
// Request
{
  "message": "What's the stock price of AAPL?",
  "session_id": "abc123"
}

// Response
{
  "response": "📊 Financial Analysis\n\n📈 AAPL: $182.45 (+1.23%)\nApple Inc.\n\n*Updated: 15:30:00*"
}
```

---

### Image Processing

#### `POST /api/process/image`
Analyze an uploaded image.

**Request**
```
Content-Type: multipart/form-data

image: [file] (required) - Image file (PNG, JPG, JPEG, GIF, BMP, WEBP)
session_id: [string] (optional) - Session identifier
```

**Response**
```json
{
  "job_id": "img_123abc",
  "status": "queued",
  "message": "Image analysis started"
}
```

**Status Codes**
- `200 OK` - Job created
- `400 Bad Request` - Invalid file or missing image
- `413 Payload Too Large` - File exceeds 20MB limit
- `500 Internal Server Error` - Server error

---

#### `POST /api/process/image-with-text`
Analyze an image with a specific question.

**Request**
```
Content-Type: multipart/form-data

image: [file] (required) - Image file
question: [string] (required) - Question about the image
session_id: [string] (optional) - Session identifier
```

**Example**
```bash
curl -X POST http://localhost:5000/api/process/image-with-text \
  -F "image=@photo.jpg" \
  -F "question=What objects are in this image?" \
  -F "session_id=abc123"
```

**Response**
```json
{
  "job_id": "img_456def",
  "status": "queued",
  "message": "Analysis started"
}
```

**Status Codes**
- `200 OK` - Job created
- `400 Bad Request` - Invalid input
- `413 Payload Too Large` - File too large
- `500 Internal Server Error` - Server error

---

### Voice Processing

#### `POST /api/process/voice`
Transcribe voice input to text (Voice Processing mode).

**Request**
```
Content-Type: multipart/form-data

audio: [file] (required) - Audio file (WebM, WAV, MP3)
language: [string] (optional) - Language code (default: 'en')
session_id: [string] (optional) - Session identifier
```

**Supported Languages**
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese
- `ar` - Arabic
- `hi` - Hindi
- `ur` - Urdu
- `th` - Thai

**Response**
```json
{
  "job_id": "voice_789ghi",
  "status": "queued",
  "message": "Voice processing started"
}
```

**Status Codes**
- `200 OK` - Job created
- `400 Bad Request` - Invalid file
- `500 Internal Server Error` - Server error

---

#### `POST /api/process/voice-conversation`
Process voice input and get AI voice response (Voice Conversation mode).

**Request**
```
Content-Type: multipart/form-data

audio: [file] (required) - Audio file
language: [string] (optional) - Language code (default: 'en')
session_id: [string] (optional) - Session identifier
```

**Response**
```json
{
  "job_id": "conv_abc123",
  "status": "queued",
  "message": "Voice conversation started"
}
```

**Complete Flow**
1. User speaks in selected language
2. System transcribes speech to text
3. AI processes the text and generates response
4. System converts AI response to speech in same language
5. User receives both text and audio response

**Status Codes**
- `200 OK` - Job created
- `400 Bad Request` - Invalid input
- `500 Internal Server Error` - Server error

---

#### `POST /api/process/text-to-speech`
Convert text to speech.

**Request**
```json
{
  "text": "Hello, this is a test message",
  "language": "en",
  "session_id": "abc123"
}
```

**Parameters**
- `text` (string, required) - Text to convert (1-5000 characters)
- `language` (string, optional) - Language code (default: 'en')
- `session_id` (string, optional) - Session identifier

**Response**
```json
{
  "job_id": "tts_xyz789",
  "status": "queued",
  "message": "TTS started"
}
```

**Status Codes**
- `200 OK` - Job created
- `400 Bad Request` - Invalid input
- `500 Internal Server Error` - Server error

---

### Job Status

#### `GET /api/status/{job_id}`
Get the status and results of a processing job.

**Path Parameters**
- `job_id` (string) - The job identifier

**Response (Pending)**
```json
{
  "job_id": "img_123abc",
  "status": "processing",
  "progress": 45,
  "type": "image_analysis",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Response (Completed - Image Analysis)**
```json
{
  "job_id": "img_123abc",
  "status": "completed",
  "progress": 100,
  "type": "image_analysis",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:30:03Z",
  "result": {
    "analysis": {
      "objects": ["person", "car", "building"],
      "scene_type": "urban",
      "dominant_colors": ["blue", "gray", "white"],
      "emotions": ["neutral"],
      "text_content": "STOP",
      "confidence_score": 0.95,
      "suggestions": ["High quality image", "Good lighting"],
      "processing_time": 2.34
    },
    "formatted_html": "<div class='image-analysis-result'>...</div>",
    "enhanced_image_base64": "data:image/jpeg;base64,...",
    "type": "image_analysis"
  }
}
```

**Response (Completed - Voice Conversation)**
```json
{
  "job_id": "conv_abc123",
  "status": "completed",
  "progress": 100,
  "type": "voice_conversation",
  "result": {
    "transcription": "What is the weather today?",
    "ai_response": "<p>The weather today is sunny with...</p>",
    "audio_base64": "//NExAAAAAANIAAAAAExBTUUzLjEw...",
    "audio_format": "mp3",
    "language": "en",
    "type": "voice_conversation",
    "success": true
  }
}
```

**Response (Error)**
```json
{
  "job_id": "img_123abc",
  "status": "error",
  "progress": 0,
  "type": "image_analysis",
  "error": "Failed to process image: Invalid format"
}
```

**Status Codes**
- `200 OK` - Job status retrieved
- `404 Not Found` - Job not found
- `500 Internal Server Error` - Server error

---

### Utility Endpoints

#### `POST /api/clear-history`
Clear chat history for a session.

**Request**
```json
{
  "session_id": "a1b2c3d4"
}
```

**Response**
```json
{
  "success": true,
  "message": "Chat history cleared"
}
```

**Status Codes**
- `200 OK` - History cleared
- `500 Internal Server Error` - Server error

---

## 📝 Request/Response Formats

### Standard Response Format

All successful API responses follow this structure:

```json
{
  "success": true,
  "data": { ... },
  "message": "Operation successful",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Response Format

```json
{
  "success": false,
  "error": "Error message",
  "error_code": "INVALID_INPUT",
  "details": {
    "field": "message",
    "reason": "Message cannot be empty"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## ⚠️ Error Handling

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_INPUT` | 400 | Request validation failed |
| `MISSING_FIELD` | 400 | Required field missing |
| `FILE_TOO_LARGE` | 413 | File exceeds size limit |
| `INVALID_FORMAT` | 400 | Unsupported file format |
| `SESSION_NOT_FOUND` | 404 | Session ID not found |
| `JOB_NOT_FOUND` | 404 | Job ID not found |
| `PROCESSING_ERROR` | 500 | Error during processing |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |

### Example Error Responses

**Invalid Input**
```json
{
  "success": false,
  "error": "Message is required",
  "error_code": "MISSING_FIELD",
  "details": {
    "field": "message"
  }
}
```

**File Too Large**
```json
{
  "success": false,
  "error": "File size exceeds 20MB limit",
  "error_code": "FILE_TOO_LARGE",
  "details": {
    "max_size": "20MB",
    "received_size": "25.3MB"
  }
}
```

**Rate Limit Exceeded**
```json
{
  "success": false,
  "error": "Rate limit exceeded",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "details": {
    "limit": "100 requests per hour",
    "retry_after": 3600
  }
}
```

---

## 🚦 Rate Limiting

### Default Limits

| Endpoint | Rate Limit | Window |
|----------|-----------|--------|
| `/api/process/text` | 10 requests | per minute |
| `/api/process/image` | 5 requests | per minute |
| `/api/process/voice` | 5 requests | per minute |
| `/api/sessions` | 20 requests | per minute |
| Global | 100 requests | per hour |

### Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

### Handling Rate Limits

```javascript
fetch('/api/process/text', options)
  .then(response => {
    if (response.status === 429) {
      const retryAfter = response.headers.get('Retry-After');
      console.log(`Rate limited. Retry after ${retryAfter} seconds`);
    }
    return response.json();
  });
```

---

## 💻 Code Examples

### JavaScript/TypeScript

```javascript
// Create session and send message
class QuantumAIClient {
  constructor(baseUrl = 'http://localhost:5000') {
    this.baseUrl = baseUrl;
    this.sessionId = null;
  }

  async createSession() {
    const response = await fetch(`${this.baseUrl}/api/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const data = await response.json();
    this.sessionId = data.session_id;
    return this.sessionId;
  }

  async sendMessage(message) {
    const response = await fetch(`${this.baseUrl}/api/process/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        session_id: this.sessionId
      })
    });

    // Handle streaming response
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullResponse = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (data.word) {
            fullResponse += data.word;
            console.log('Streaming:', data.word);
          }
          if (data.complete) {
            return fullResponse;
          }
        }
      }
    }
  }

  async analyzeImage(file, question = null) {
    const formData = new FormData();
    formData.append('image', file);
    if (question) formData.append('question', question);
    if (this.sessionId) formData.append('session_id', this.sessionId);

    const response = await fetch(`${this.baseUrl}/api/process/image${question ? '-with-text' : ''}`, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    return await this.pollJobStatus(data.job_id);
  }

  async pollJobStatus(jobId, maxAttempts = 60) {
    for (let i = 0; i < maxAttempts; i++) {
      const response = await fetch(`${this.baseUrl}/api/status/${jobId}`);
      const data = await response.json();

      if (data.status === 'completed') {
        return data.result;
      } else if (data.status === 'error') {
        throw new Error(data.error);
      }

      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    throw new Error('Job timeout');
  }
}

// Usage
const client = new QuantumAIClient();
await client.createSession();
const response = await client.sendMessage('What is AI?');
console.log(response);
```

### Python

```python
import requests
import time
import json

class QuantumAIClient:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        self.session_id = None

    def create_session(self):
        response = requests.post(f'{self.base_url}/api/sessions')
        data = response.json()
        self.session_id = data['session_id']
        return self.session_id

    def send_message(self, message):
        response = requests.post(
            f'{self.base_url}/api/process/text',
            json={'message': message, 'session_id': self.session_id},
            stream=True
        )

        full_response = ''
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if 'word' in data:
                        full_response += data['word']
                    if data.get('complete'):
                        return full_response

    def analyze_image(self, image_path, question=None):
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'session_id': self.session_id}
            if question:
                data['question'] = question
                endpoint = '/api/process/image-with-text'
            else:
                endpoint = '/api/process/image'

            response = requests.post(
                f'{self.base_url}{endpoint}',
                files=files,
                data=data
            )

        job_data = response.json()
        return self.poll_job_status(job_data['job_id'])

    def poll_job_status(self, job_id, max_attempts=60):
        for _ in range(max_attempts):
            response = requests.get(f'{self.base_url}/api/status/{job_id}')
            data = response.json()

            if data['status'] == 'completed':
                return data['result']
            elif data['status'] == 'error':
                raise Exception(data['error'])

            time.sleep(1)

        raise Exception('Job timeout')

# Usage
client = QuantumAIClient()
client.create_session()
response = client.send_message('What is machine learning?')
print(response)
```

### cURL Examples

```bash
# Create session
curl -X POST http://localhost:5000/api/sessions

# Send text message
curl -X POST http://localhost:5000/api/process/text \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello AI", "session_id": "abc123"}'

# Upload and analyze image
curl -X POST http://localhost:5000/api/process/image \
  -F "image=@photo.jpg" \
  -F "session_id=abc123"

# Image with question
curl -X POST http://localhost:5000/api/process/image-with-text \
  -F "image=@photo.jpg" \
  -F "question=What is in this image?" \
  -F "session_id=abc123"

# Check job status
curl http://localhost:5000/api/status/img_123abc

# Voice processing
curl -X POST http://localhost:5000/api/process/voice \
  -F "audio=@recording.webm" \
  -F "language=en" \
  -F "session_id=abc123"

# Text-to-speech
curl -X POST http://localhost:5000/api/process/text-to-speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "en", "session_id": "abc123"}'
```

---

## 🔗 Additional Resources

- [Architecture Documentation](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [GitHub Repository](https://github.com/MuhammadAbbas01/quantum-ai-nexus)

---

**Last Updated**: January 2024  
**API Version**: v1.0.0  
**Support**: abbaskhan0011ehe@gmail.com
