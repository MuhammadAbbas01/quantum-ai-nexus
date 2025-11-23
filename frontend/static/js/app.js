// ===== ENHANCED MULTIMODAL CHAT APPLICATION - STREAMING FIXED =====
class EnhancedMultimodalChatApp {
    constructor() {
        this.currentMode = 'text';
        this.currentSessionId = null;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.currentJobId = null;
        this.pollingInterval = null;
        this.speechSynthesis = window.speechSynthesis;
        this.speechRecognition = null;
        this.streamingEnabled = true;  // FIXED: ALWAYS TRUE BY DEFAULT
        this.textToSpeechEnabled = true;
        this.currentStreamingMessage = null;
        this.sidebarOpen = false;
        this.availableVoices = [];
        this.uploadedImageFile = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initializeSpeechRecognition();
        this.initializeSpeechSynthesis();
        this.showWelcomeAnimation();
        this.loadUserPreferences();
        this.loadChatSessions();
        this.createNewSession();
        
        console.log('✅ APP INITIALIZED: Streaming is ENABLED (FIXED)');
    }
    
    // ===== INITIALIZATION =====
    setupEventListeners() {
        const messageInput = document.getElementById('messageInput');
        const ttsMessageInput = document.getElementById('ttsMessageInput');
        const imageInput = document.getElementById('imageInput');
        const imageAnalysisInput = document.getElementById('imageAnalysisInput');
        
        if (messageInput) {
            messageInput.addEventListener('input', this.autoResizeTextarea);
            messageInput.addEventListener('keydown', (e) => this.handleKeydown(e));
        }
        
        if (ttsMessageInput) {
            ttsMessageInput.addEventListener('input', this.autoResizeTextarea);
        }
        
        if (imageInput) {
            imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        }
        
        if (imageAnalysisInput) {
            imageAnalysisInput.addEventListener('change', (e) => this.handleImageAnalysisUpload(e));
        }
        
        const recordBtn = document.getElementById('recordBtn');
        if (recordBtn) {
            recordBtn.addEventListener('mousedown', () => this.startRecording());
            recordBtn.addEventListener('mouseup', () => this.stopRecording());
            recordBtn.addEventListener('mouseleave', () => this.stopRecording());
            recordBtn.addEventListener('touchstart', (e) => {
                e.preventDefault();
                this.startRecording();
            });
            recordBtn.addEventListener('touchend', (e) => {
                e.preventDefault();
                this.stopRecording();
            });
            recordBtn.addEventListener('contextmenu', (e) => e.preventDefault());
        }
        
        // Drag and drop for image upload areas
        this.setupDragAndDrop();
        
        window.addEventListener('beforeunload', () => this.cleanup());
        
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('generated-image') || e.target.classList.contains('enhanced-image')) {
                this.showImageFullscreen(e.target.src);
            }
        });
    }
    
    setupDragAndDrop() {
        const uploadZones = document.querySelectorAll('.upload-zone, .upload-area');
        
        uploadZones.forEach(zone => {
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('drag-over');
            });
            
            zone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                zone.classList.remove('drag-over');
            });
            
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('drag-over');
                
                const files = e.dataTransfer.files;
                if (files.length > 0 && files[0].type.startsWith('image/')) {
                    if (zone.closest('#imageInputContainer')) {
                        this.handleImageAnalysisFile(files[0]);
                    } else {
                        this.handleImageFile(files[0]);
                    }
                }
            });
        });
    }
    
    initializeSpeechRecognition() {
        if ('webkitSpeechRecognition' in window) {
            this.speechRecognition = new webkitSpeechRecognition();
            this.speechRecognition.continuous = false;
            this.speechRecognition.interimResults = false;
            this.speechRecognition.lang = 'en-US';
        }
    }
    
    initializeSpeechSynthesis() {
        if (this.speechSynthesis) {
            const loadVoices = () => {
                this.availableVoices = this.speechSynthesis.getVoices();
                console.log(`Loaded ${this.availableVoices.length} voices`);
            };
            
            loadVoices();
            
            if (this.speechSynthesis.onvoiceschanged !== undefined) {
                this.speechSynthesis.onvoiceschanged = loadVoices;
            }
            
            setTimeout(loadVoices, 1000);
        }
    }
    
    showWelcomeAnimation() {
        const modeSelector = document.querySelector('.mode-selector');
        if (modeSelector) {
            modeSelector.classList.add('animate__animated', 'animate__fadeInUp');
        }
    }
    
    loadUserPreferences() {
        // FIXED: FORCE STREAMING TO BE ENABLED
        this.streamingEnabled = true;
        localStorage.setItem('streaming_enabled', 'true');
        this.updateStreamingUI();
        
        const ttsPref = localStorage.getItem('tts_enabled');
        if (ttsPref !== null) {
            this.textToSpeechEnabled = ttsPref === 'true';
            this.updateTTSUI();
        }
        
        console.log('✅ PREFERENCES LOADED: Streaming = ENABLED (FORCED)');
    }
    
    autoResizeTextarea(event) {
        const textarea = event.target;
        textarea.style.height = 'auto';
        const maxHeight = window.innerWidth <= 768 ? 100 : 120;
        textarea.style.height = Math.min(textarea.scrollHeight, maxHeight) + 'px';
    }
    
    handleKeydown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }
    
    // ===== SESSION MANAGEMENT (UNCHANGED) =====
    async loadChatSessions() {
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();
            
            if (response.ok) {
                this.displaySessions(data.sessions);
            }
        } catch (error) {
            console.error('Failed to load sessions:', error);
        }
    }
    
    displaySessions(sessions) {
        const sessionsList = document.getElementById('sessionsList');
        if (!sessionsList) return;
        
        if (sessions.length === 0) {
            sessionsList.innerHTML = '<div class="no-sessions">No chat history yet</div>';
            return;
        }
        
        sessionsList.innerHTML = sessions.map(session => `
            <div class="session-item ${session.id === this.currentSessionId ? 'active' : ''}" 
                 onclick="app.loadSession('${session.id}')">
                <div class="session-title">${session.title}</div>
                <div class="session-meta">
                    <span>${session.message_count} messages</span>
                    <span>${new Date(session.updated_at).toLocaleDateString()}</span>
                </div>
                <button class="session-delete" onclick="event.stopPropagation(); app.deleteSession('${session.id}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `).join('');
    }
    
    async createNewSession() {
        try {
            const response = await fetch('/api/sessions', { method: 'POST' });
            const data = await response.json();
            
            if (response.ok) {
                this.currentSessionId = data.session_id;
                this.loadChatSessions();
            }
        } catch (error) {
            console.error('Failed to create session:', error);
        }
    }
    
    async loadSession(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}/messages`);
            const data = await response.json();
            
            if (response.ok) {
                this.currentSessionId = sessionId;
                this.displaySessionMessages(data.messages);
                this.loadChatSessions();
                
                if (this.sidebarOpen) {
                    this.toggleSidebar();
                }
            }
        } catch (error) {
            console.error('Failed to load session:', error);
        }
    }
    
    displaySessionMessages(messages) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const existingMessages = chatMessages.querySelectorAll('.message');
        existingMessages.forEach(msg => msg.remove());
        
        messages.forEach(message => {
            this.addMessage(message.type, message.content, false);
        });
    }
    
    async deleteSession(sessionId) {
        if (!confirm('Are you sure you want to delete this chat?')) return;
        
        try {
            const response = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
            
            if (response.ok) {
                if (sessionId === this.currentSessionId) {
                    await this.createNewSession();
                    this.clearChatDisplay();
                }
                this.loadChatSessions();
            }
        } catch (error) {
            console.error('Failed to delete session:', error);
        }
    }
    
    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        const mainContainer = document.querySelector('.main-container');
        
        this.sidebarOpen = !this.sidebarOpen;
        if (sidebar) sidebar.classList.toggle('active', this.sidebarOpen);
        if (mainContainer) mainContainer.classList.toggle('sidebar-open', this.sidebarOpen);
        
        if (this.sidebarOpen) {
            this.loadChatSessions();
        }
    }
    
    // ===== MODE MANAGEMENT (UNCHANGED) =====
    selectMode(mode) {
        this.currentMode = mode;
        
        document.querySelectorAll('.mode-card').forEach(card => {
            card.classList.remove('active');
        });
        
        const selectedCard = document.querySelector(`[data-mode="${mode}"]`);
        if (selectedCard) selectedCard.classList.add('active');
        
        setTimeout(() => {
            this.showChatInterface();
        }, 300);
    }
    
    showChatInterface() {
        const modeSelector = document.getElementById('modeSelector');
        const chatMessages = document.getElementById('chatMessages');
        const inputInterface = document.getElementById('inputInterface');
        
        if (modeSelector) modeSelector.style.display = 'none';
        if (chatMessages) chatMessages.style.display = 'block';
        if (inputInterface) inputInterface.style.display = 'block';
        
        this.setupInputInterface();
        
        if (chatMessages) chatMessages.classList.add('animate__animated', 'animate__fadeIn');
        if (inputInterface) inputInterface.classList.add('animate__animated', 'animate__fadeInUp');
    }
    
    setupInputInterface() {
        const textContainer = document.getElementById('textInputContainer');
        const voiceContainer = document.getElementById('voiceInputContainer');
        const imageContainer = document.getElementById('imageInputContainer');
        const languageSelector = document.getElementById('languageSelector');
        const ttsInput = document.getElementById('ttsInput');
        const imageToggle = document.getElementById('imageToggle');
        
        // Hide all containers first
        if (textContainer) textContainer.style.display = 'none';
        if (voiceContainer) voiceContainer.style.display = 'none';
        if (imageContainer) imageContainer.style.display = 'none';
        
        if (this.currentMode === 'text') {
            if (textContainer) textContainer.style.display = 'block';
            if (imageToggle) imageToggle.style.display = 'block';
            const messageInput = document.getElementById('messageInput');
            if (messageInput) messageInput.focus();
        } else if (this.currentMode === 'image') {
            if (imageContainer) imageContainer.style.display = 'block';
        } else {
            if (voiceContainer) voiceContainer.style.display = 'block';
            
            const voiceStatus = document.getElementById('voiceStatus');
            if (this.currentMode === 'voice-voice') {
                if (languageSelector) languageSelector.style.display = 'block';
                if (ttsInput) ttsInput.style.display = 'none';
                if (voiceStatus) {
                    voiceStatus.innerHTML = '<i class="fas fa-info-circle"></i><span>Press and hold to speak, AI will respond in selected language</span>';
                }
            } else {
                if (languageSelector) languageSelector.style.display = 'none';
                if (ttsInput) ttsInput.style.display = 'block';
                if (voiceStatus) {
                    voiceStatus.innerHTML = '<i class="fas fa-info-circle"></i><span>Voice to text conversion only, no AI responses</span>';
                }
            }
        }
    }
    
    showModeSelector() {
        const modeSelector = document.getElementById('modeSelector');
        const chatMessages = document.getElementById('chatMessages');
        const inputInterface = document.getElementById('inputInterface');
        
        if (modeSelector) modeSelector.style.display = 'block';
        if (chatMessages) chatMessages.style.display = 'none';
        if (inputInterface) inputInterface.style.display = 'none';
        
        if (modeSelector) modeSelector.classList.add('animate__animated', 'animate__fadeIn');
    }
    
    // ===== IMAGE HANDLING (keeping all existing functions) =====
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleImageFile(file);
        }
    }
    
    handleImageAnalysisUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleImageAnalysisFile(file);
        }
    }
    
    handleImageFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showAlert('Please select a valid image file', 'warning');
            return;
        }
        
        if (file.size > 10 * 1024 * 1024) {
            this.showAlert('Image size too large. Please select an image under 10MB', 'warning');
            return;
        }
        
        this.uploadedImageFile = file;
        this.displayImagePreview(file, 'uploadedImagePreview');
        
        const imageUploadArea = document.getElementById('imageUploadArea');
        const uploadedImagePreview = document.getElementById('uploadedImagePreview');
        
        if (imageUploadArea && uploadedImagePreview) {
            const uploadZone = imageUploadArea.querySelector('.upload-zone');
            if (uploadZone) uploadZone.style.display = 'none';
            uploadedImagePreview.style.display = 'block';
        }
    }
    
    handleImageAnalysisFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showAlert('Please select a valid image file', 'warning');
            return;
        }
        
        if (file.size > 10 * 1024 * 1024) {
            this.showAlert('Image size too large. Please select an image under 10MB', 'warning');
            return;
        }
        
        this.uploadedImageFile = file;
        this.displayImagePreview(file, 'analysisPreviewImg');
        
        const imagePreview = document.getElementById('imagePreview');
        const uploadArea = document.querySelector('.upload-area');
        
        if (uploadArea && imagePreview) {
            uploadArea.style.display = 'none';
            imagePreview.style.display = 'block';
        }
    }
    
    displayImagePreview(file, imageElementId) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.getElementById(imageElementId);
            if (img) {
                img.src = e.target.result;
            }
        };
        reader.readAsDataURL(file);
    }
    
    removeUploadedImage() {
        this.uploadedImageFile = null;
        const imageUploadArea = document.getElementById('imageUploadArea');
        const uploadedImagePreview = document.getElementById('uploadedImagePreview');
        const imageInput = document.getElementById('imageInput');
        
        if (imageUploadArea && uploadedImagePreview) {
            const uploadZone = imageUploadArea.querySelector('.upload-zone');
            if (uploadZone) uploadZone.style.display = 'block';
            uploadedImagePreview.style.display = 'none';
        }
        if (imageInput) imageInput.value = '';
    }
    
    removeImagePreview() {
        this.uploadedImageFile = null;
        const imagePreview = document.getElementById('imagePreview');
        const uploadArea = document.querySelector('.upload-area');
        const imageAnalysisInput = document.getElementById('imageAnalysisInput');
        
        if (uploadArea && imagePreview) {
            uploadArea.style.display = 'block';
            imagePreview.style.display = 'none';
        }
        if (imageAnalysisInput) imageAnalysisInput.value = '';
    }
    
    toggleImageUpload() {
        const imageUploadArea = document.getElementById('imageUploadArea');
        const imageToggle = document.getElementById('imageToggle');
        
        if (imageUploadArea) {
            if (imageUploadArea.style.display === 'none' || !imageUploadArea.style.display) {
                imageUploadArea.style.display = 'block';
                if (imageToggle) imageToggle.classList.add('active');
            } else {
                imageUploadArea.style.display = 'none';
                if (imageToggle) imageToggle.classList.remove('active');
                this.removeUploadedImage();
            }
        }
    }
    
    // ===== IMAGE ANALYSIS (keeping existing) =====
    async analyzeImage() {
        if (!this.uploadedImageFile) {
            this.showAlert('Please select an image first', 'warning');
            return;
        }
        
        this.showProcessingStatus('Analyzing image with advanced AI processing...');
        
        try {
            const formData = new FormData();
            formData.append('image', this.uploadedImageFile);
            if (this.currentSessionId) {
                formData.append('session_id', this.currentSessionId);
            }
            
            const response = await fetch('/api/process/image', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || 'Image processing failed');
            }
            
            this.currentJobId = result.job_id;
            this.startPolling();
            
        } catch (error) {
            console.error('Image analysis error:', error);
            this.showAlert('Image analysis failed', 'error', error.message);
            this.hideProcessingStatus();
        }
    }
    
    async analyzeImageWithText() {
        if (!this.uploadedImageFile) {
            this.showAlert('Please select an image first', 'warning');
            return;
        }
        
        const messageInput = document.getElementById('imageMessageInput');
        const userMessage = messageInput ? messageInput.value.trim() : '';
        
        if (!userMessage) {
            this.showAlert('Please enter a question about the image', 'warning');
            return;
        }
        
        this.showProcessingStatus('Analyzing image and generating AI response...');
        
        try {
            const formData = new FormData();
            formData.append('image', this.uploadedImageFile);
            formData.append('question', userMessage);
            if (this.currentSessionId) {
                formData.append('session_id', this.currentSessionId);
            }
            
            const response = await fetch('/api/process/image-with-text', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || 'Image processing failed');
            }
            
            this.currentJobId = result.job_id;
            this.startPolling();
            
            if (messageInput) {
                messageInput.value = '';
            }
            
        } catch (error) {
            console.error('Image with text analysis error:', error);
            this.showAlert('Image analysis failed', 'error', error.message);
            this.hideProcessingStatus();
        }
    }
    
    // ===== FIXED: MESSAGE HANDLING WITH GUARANTEED STREAMING =====
    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput ? messageInput.value.trim() : '';
        
        // Handle image + text in text mode
        if (this.currentMode === 'text' && this.uploadedImageFile) {
            return this.sendImageWithText(message);
        }
        
        if (!message) {
            this.showAlert('Please enter a message', 'warning');
            return;
        }
        
        this.addMessage('user', message);
        
        if (messageInput) {
            messageInput.value = '';
            messageInput.style.height = 'auto';
        }
        
        this.setLoadingState(true);
        
        try {
            // FIXED: ALWAYS USE STREAMING (removed toggle check)
            console.log('✅ SENDING MESSAGE WITH STREAMING');
            await this.sendStreamingMessage(message);
        } catch (error) {
            console.error('Message processing error:', error);
            this.showAlert('Failed to process message', 'error', error.message);
            this.addMessage('ai', 'I\'m having trouble connecting. Please try again.');
        } finally {
            this.setLoadingState(false);
        }
    }
    
    async sendImageWithText(message) {
        if (!this.uploadedImageFile) {
            this.showAlert('Please select an image first', 'warning');
            return;
        }
        
        const userMessage = message || 'Analyze this image in detail';
        this.addMessage('user', `${userMessage} [Image: ${this.uploadedImageFile.name}]`);
        
        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.value = '';
            messageInput.style.height = 'auto';
        }
        
        this.showProcessingStatus('Analyzing image and generating AI response...');
        
        try {
            const formData = new FormData();
            formData.append('image', this.uploadedImageFile);
            formData.append('question', userMessage);
            if (this.currentSessionId) {
                formData.append('session_id', this.currentSessionId);
            }
            
            const response = await fetch('/api/process/image-with-text', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || 'Image processing failed');
            }
            
            this.currentJobId = result.job_id;
            this.startPolling();
            
            // Clear uploaded image after sending
            this.removeUploadedImage();
            
        } catch (error) {
            console.error('Image with text processing error:', error);
            this.showAlert('Image processing failed', 'error', error.message);
            this.hideProcessingStatus();
        }
    }
    
    // ===== FIXED: STREAMING FUNCTION - WORD BY WORD GUARANTEED =====
    async sendStreamingMessage(message) {
        console.log('✅ STREAMING: Starting stream for:', message.substring(0, 50));
        
        const aiMessageId = this.addStreamingMessage('ai');
        
        try {
            const response = await fetch('/api/process/text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message: message, 
                    stream: true,  // FORCE STREAMING
                    session_id: this.currentSessionId 
                })
            });
            
            console.log('✅ STREAMING: Response status:', response.status);
            
            if (!response.ok) {
                const errorData = await response.json();
                this.removeStreamingMessage(aiMessageId);
                this.addMessage('ai', errorData.response || 'Connection error. Please try again.');
                return;
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';
            
            console.log('✅ STREAMING: Reading stream...');
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    console.log('✅ STREAMING: Stream complete, total length:', fullResponse.length);
                    break;
                }
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data.trim()) {
                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.error) {
                                    console.error('❌ STREAMING: Error:', parsed.error);
                                    this.removeStreamingMessage(aiMessageId);
                                    this.addMessage('ai', 'Connection error. Please try again.');
                                    return;
                                }
                                if (parsed.word !== undefined) {
                                    fullResponse += parsed.word;
                                    this.updateStreamingMessage(aiMessageId, fullResponse);
                                }
                                if (parsed.complete) {
                                    console.log('✅ STREAMING: Complete signal received');
                                    this.completeStreamingMessage(aiMessageId);
                                    break;
                                }
                            } catch (e) {
                                console.error('❌ STREAMING: Parse error:', e);
                            }
                        }
                    }
                }
            }
            
            if (this.textToSpeechEnabled && fullResponse.trim()) {
                const textForSpeech = this.stripHTML(fullResponse);
                this.speakText(textForSpeech);
            }
            
        } catch (error) {
            console.error('❌ STREAMING: Fatal error:', error);
            this.removeStreamingMessage(aiMessageId);
            this.addMessage('ai', 'Connection error. Please try again.');
            throw error;
        }
    }
    
    // ===== MESSAGE DISPLAY =====
    addMessage(type, content, animate = true) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'ai' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
        const text = document.createElement('div');
        text.className = 'message-text';
        
        if (content && typeof content === 'string') {
            text.innerHTML = this.processMessageContent(content);
        } else {
            text.innerHTML = content || '';
        }
        
        const time = document.createElement('div');
        time.className = 'message-time';
        time.textContent = new Date().toLocaleTimeString();
        
        bubble.appendChild(text);
        bubble.appendChild(time);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(bubble);
        
        if (animate) {
            messageDiv.style.opacity = '0';
            messageDiv.style.transform = 'translateY(20px)';
        }
        
        chatMessages.appendChild(messageDiv);
        
        if (animate) {
            requestAnimationFrame(() => {
                messageDiv.style.transition = 'all 0.3s ease-out';
                messageDiv.style.opacity = '1';
                messageDiv.style.transform = 'translateY(0)';
            });
        }
        
        this.scrollToBottom();
        
        if (window.Prism) {
            Prism.highlightAllUnder(messageDiv);
        }
    }
    
    processMessageContent(content) {
        const base64ImageRegex = /data:image\/(jpeg|jpg|png|gif|webp);base64,([A-Za-z0-9+\/=]+)/g;
        
        let processedContent = content;
        let match;
        
        while ((match = base64ImageRegex.exec(content)) !== null) {
            const fullBase64 = match[0];
            
            const imageElement = `
                <div class="enhanced-image-container" style="margin: 15px 0; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);">
                    <div class="image-label" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 8px 15px; font-size: 13px; font-weight: 600;">
                        <i class="fas fa-magic" style="margin-right: 8px;"></i>Enhanced Image
                    </div>
                    <img src="${fullBase64}" 
                         class="enhanced-image" 
                         style="width: 100%; height: auto; display: block; cursor: pointer; transition: transform 0.3s ease;" 
                         onclick="app.showImageFullscreen('${fullBase64}')"
                         onmouseover="this.style.transform='scale(1.02)'"
                         onmouseout="this.style.transform='scale(1)'"
                         alt="Enhanced processed image" />
                </div>`;
            
            processedContent = processedContent.replace(fullBase64, imageElement);
        }
        
        return processedContent;
    }
    
    // ===== STREAMING MESSAGE HANDLING =====
    addStreamingMessage(type) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return null;
        
        const messageId = 'stream-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.id = messageId;
        messageDiv.className = `message ${type}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'ai' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
        const text = document.createElement('div');
        text.className = 'message-text';
        text.innerHTML = '<span class="streaming-text"></span><span class="typing-cursor"></span>';
        
        const time = document.createElement('div');
        time.className = 'message-time';
        time.textContent = new Date().toLocaleTimeString();
        
        bubble.appendChild(text);
        bubble.appendChild(time);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(bubble);
        
        chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        console.log('✅ STREAMING: Message container created:', messageId);
        
        return messageId;
    }
    
    updateStreamingMessage(messageId, content) {
        const messageEl = document.getElementById(messageId);
        if (messageEl) {
            const streamingText = messageEl.querySelector('.streaming-text');
            if (streamingText) {
                streamingText.innerHTML = this.processMessageContent(content || '');
                this.scrollToBottom();
            }
        }
    }
    
    completeStreamingMessage(messageId) {
        const messageEl = document.getElementById(messageId);
        if (messageEl) {
            const cursor = messageEl.querySelector('.typing-cursor');
            if (cursor) cursor.remove();
            
            if (window.Prism) {
                Prism.highlightAllUnder(messageEl);
            }
        }
    }
    
    removeStreamingMessage(messageId) {
        const messageEl = document.getElementById(messageId);
        if (messageEl) messageEl.remove();
    }
    
    scrollToBottom() {
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    // ===== VOICE RECORDING (keeping all existing) =====
    async startRecording() {
        if (this.isRecording) return;
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            this.isRecording = true;
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                this.processRecording();
            };
            
            this.mediaRecorder.start();
            this.updateRecordingUI(true);
            this.showVoiceVisualizer();
            
        } catch (error) {
            console.error('Recording start error:', error);
            this.showAlert('Microphone access denied', 'error', 
                'Please allow microphone access to use voice features');
        }
    }
    
    stopRecording() {
        if (!this.isRecording || !this.mediaRecorder) return;
        
        this.isRecording = false;
        this.mediaRecorder.stop();
        this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        this.updateRecordingUI(false);
        this.hideVoiceVisualizer();
    }
    
    updateRecordingUI(recording) {
        const recordBtn = document.getElementById('recordBtn');
        const voiceStatus = document.getElementById('voiceStatus');
        
        if (recording) {
            if (recordBtn) {
                recordBtn.classList.add('recording');
                const btnText = recordBtn.querySelector('.btn-text');
                if (btnText) btnText.textContent = 'Recording...';
            }
            if (voiceStatus) {
                voiceStatus.innerHTML = '<i class="fas fa-microphone"></i><span>Listening... Release to send</span>';
            }
        } else {
            if (recordBtn) {
                recordBtn.classList.remove('recording');
                const btnText = recordBtn.querySelector('.btn-text');
                if (btnText) btnText.textContent = 'Hold to Speak';
            }
            
            if (voiceStatus) {
                if (this.currentMode === 'voice-voice') {
                    voiceStatus.innerHTML = '<i class="fas fa-info-circle"></i><span>Press and hold to speak, AI will respond in selected language</span>';
                } else {
                    voiceStatus.innerHTML = '<i class="fas fa-info-circle"></i><span>Voice to text conversion only</span>';
                }
            }
        }
    }
    
    showVoiceVisualizer() {
        const visualizer = document.getElementById('voiceVisualizer');
        if (visualizer) {
            visualizer.style.display = 'flex';
            visualizer.classList.add('animate__animated', 'animate__fadeIn');
        }
    }
    
    hideVoiceVisualizer() {
        const visualizer = document.getElementById('voiceVisualizer');
        if (visualizer) {
            visualizer.style.display = 'none';
        }
    }
    
    async processRecording() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        
        if (this.currentMode === 'voice-text') {
            await this.processVoiceToTextOnly(audioBlob);
        } else if (this.currentMode === 'voice-voice') {
            await this.processVoiceToVoice(audioBlob);
        }
    }
    
    async processVoiceToTextOnly(audioBlob) {
        this.showProcessingStatus('Converting speech to text...');
        
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            
            const response = await fetch('/api/process/voice', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || 'Voice processing failed');
            }
            
            this.currentJobId = result.job_id;
            this.startPolling();
            
        } catch (error) {
            console.error('Voice to text only error:', error);
            this.showAlert('Voice processing failed', 'error', error.message);
            this.hideProcessingStatus();
        }
    }
    
    async processVoiceToVoice(audioBlob) {
        this.showProcessingStatus('Processing voice conversation...');
        
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            
            const languageSelect = document.getElementById('responseLanguage');
            const language = languageSelect ? languageSelect.value : 'en';
            formData.append('language', language);
            
            const response = await fetch('/api/process/voice', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || 'Voice processing failed');
            }
            
            this.currentJobId = result.job_id;
            this.startPolling();
            
        } catch (error) {
            console.error('Voice to voice error:', error);
            this.showAlert('Voice processing failed', 'error', error.message);
            this.hideProcessingStatus();
        }
    }
    
    // ===== TEXT TO SPEECH (keeping existing) =====
    getBestVoiceForLanguage(languageCode) {
        if (!this.availableVoices.length) {
            return null;
        }
        
        const langPrefix = languageCode.split('-')[0];
        
        const voicePreferences = {
            'zh': ['Google 普通话', 'Microsoft Huihui', 'Ting-Ting'],
            'ja': ['Google 日本語', 'Microsoft Haruka', 'Kyoko'],
            'ko': ['Google 한국의', 'Microsoft Heami', 'Yuna'],
            'ar': ['Google العربية', 'Microsoft Hoda', 'Maged'],
            'hi': ['Google हिन्दी', 'Microsoft Hemant', 'Lekha'],
            'th': ['Google ไทย', 'Microsoft Achara', 'Kanya'],
            'ru': ['Google русский', 'Microsoft Irina', 'Milena'],
            'fr': ['Google français', 'Microsoft Hortense', 'Amelie'],
            'de': ['Google Deutsch', 'Microsoft Katja', 'Anna'],
            'es': ['Google español', 'Microsoft Helena', 'Monica'],
            'it': ['Google italiano', 'Microsoft Elsa', 'Alice'],
            'pt': ['Google português', 'Microsoft Maria', 'Luciana'],
            'en': ['Google UK English Female', 'Microsoft Zira', 'Alex', 'Samantha']
        };
        
        const preferredNames = voicePreferences[langPrefix] || voicePreferences['en'];
        
        for (const prefName of preferredNames) {
            const voice = this.availableVoices.find(v => 
                v.name.includes(prefName) || 
                v.name.toLowerCase().includes(prefName.toLowerCase())
            );
            if (voice) return voice;
        }
        
        let voice = this.availableVoices.find(v => 
            v.lang.startsWith(langPrefix) || 
            v.lang.startsWith(languageCode)
        );
        
        if (voice) return voice;
        
        return this.availableVoices.find(v => v.default) || this.availableVoices[0];
    }
    
    speakText(text, languageCode = 'en-US') {
        if (!this.speechSynthesis || !this.textToSpeechEnabled || !text || !text.trim()) return;
        
        this.speechSynthesis.cancel();
        
        let cleanText = text
            .replace(/<[^>]*>/g, ' ')
            .replace(/\s+/g, ' ')
            .replace(/[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u0600-\u06ff\u0900-\u097f\u0e00-\u0e7f]/g, ' ')
            .trim();
        
        if (!cleanText || cleanText.length > 500) {
            cleanText = cleanText.substring(0, 497) + '...';
        }
        
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.lang = languageCode;
        const bestVoice = this.getBestVoiceForLanguage(languageCode);
        if (bestVoice) {
            utterance.voice = bestVoice;
        }
        
        utterance.rate = 0.9;
        utterance.pitch = 1;
        utterance.volume = 0.8;
        
        utterance.onerror = (event) => {
            console.error('Speech error:', event.error);
        };
        
        try {
            this.speechSynthesis.speak(utterance);
        } catch (error) {
            console.error('Speech synthesis error:', error);
        }
    }
    
    async speakTextFromInput() {
        const ttsInput = document.getElementById('ttsMessageInput');
        const text = ttsInput ? ttsInput.value.trim() : '';
        
        if (!text) {
            this.showAlert('Please enter text to speak', 'warning');
            return;
        }
        
        try {
            const languageSelect = document.getElementById('responseLanguage');
            const language = languageSelect ? languageSelect.value : 'en';
            
            const response = await fetch('/api/process/text-to-speech', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text: text,
                    language: language
                })
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || 'Text-to-speech failed');
            }
            
            this.currentJobId = result.job_id;
            this.startPolling();
            
            if (ttsInput) ttsInput.value = '';
            
        } catch (error) {
            console.error('Text to speech error:', error);
            this.showAlert('Text-to-speech failed', 'error', error.message);
        }
    }
    
    // ===== JOB POLLING (keeping existing) =====
    startPolling() {
        this.pollingInterval = setInterval(() => {
            this.checkJobStatus();
        }, 1000);
    }
    
    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }
    
    async checkJobStatus() {
        if (!this.currentJobId) return;
        
        try {
            const response = await fetch(`/api/status/${this.currentJobId}`);
            const status = await response.json();
            
            if (!response.ok) {
                throw new Error(status.error || 'Status check failed');
            }
            
            this.updateProcessingProgress(status.progress || 0);
            
            if (status.status === 'completed') {
                this.handleJobComplete(status);
            } else if (status.status === 'error') {
                this.handleJobError(status);
            }
            
        } catch (error) {
            console.error('Status check error:', error);
            this.handleJobError({ error: error.message });
        }
    }
    
    handleJobComplete(status) {
        this.stopPolling();
        this.hideProcessingStatus();
        
        const result = status.result;
        
        if (status.type === 'image_analysis') {
            if (result) {
                let responseContent = '';
                
                const enhancedImage = result.enhanced_image_base64 || 
                                    (result.metadata && result.metadata.enhanced_image_base64) ||
                                    (result.analysis_result && result.analysis_result.enhanced_image_base64);
                
                if (enhancedImage) {
                    responseContent += enhancedImage + '<br><br>';
                }
                
                if (result.formatted_html) {
                    responseContent += result.formatted_html;
                } else if (result.analysis_text) {
                    responseContent += result.analysis_text;
                } else {
                    responseContent += 'Image analysis completed successfully.';
                }
                
                this.addMessage('ai', responseContent);
                this.showAlert('Image analysis completed successfully', 'success');
            } else {
                this.showAlert('Image analysis completed but no results returned', 'warning');
            }
            
        } else if (status.type === 'image_text_analysis') {
            if (result) {
                let responseContent = '';
                
                const enhancedImage = result.enhanced_image_base64 || 
                                    (result.metadata && result.metadata.enhanced_image_base64) ||
                                    (result.analysis_result && result.analysis_result.enhanced_image_base64);
                
                if (enhancedImage) {
                    responseContent += enhancedImage + '<br><br>';
                }
                
                if (result.formatted_html) {
                    responseContent += result.formatted_html;
                } else if (result.ai_response) {
                    responseContent += result.ai_response;
                } else {
                    responseContent += 'Image analysis with text completed successfully.';
                }
                
                this.addMessage('ai', responseContent);
                
                if (this.textToSpeechEnabled && (result.ai_response || result.formatted_html)) {
                    const textForSpeech = this.stripHTML(result.ai_response || result.formatted_html);
                    setTimeout(() => this.speakText(textForSpeech), 500);
                }
            } else {
                this.showAlert('Image analysis completed but no results returned', 'warning');
            }
            
        } else if (status.type === 'voice_to_text') {
            if (result && result.transcription) {
                this.addMessage('user', `[Voice] ${result.transcription}`);
                this.showAlert('Voice converted to text successfully', 'success');
            } else {
                this.showAlert('No speech detected', 'warning');
            }
            
        } else if (status.type === 'text_to_speech') {
            if (result && result.audio_base64) {
                const audio = new Audio(`data:audio/mp3;base64,${result.audio_base64}`);
                audio.play();
                this.showAlert('Text-to-speech completed', 'success');
            }
        }
        
        this.currentJobId = null;
    }
    
    handleJobError(status) {
        this.stopPolling();
        this.hideProcessingStatus();
        this.showAlert('Processing failed', 'error', status.error || 'An unexpected error occurred');
        this.currentJobId = null;
    }
    
    getLanguageCode(lang) {
        const languageMap = {
            'en': 'en-US', 'es': 'es-ES', 'fr': 'fr-FR', 'de': 'de-DE',
            'it': 'it-IT', 'pt': 'pt-PT', 'ru': 'ru-RU', 'ja': 'ja-JP',
            'ko': 'ko-KR', 'zh': 'zh-CN', 'ar': 'ar-SA', 'hi': 'hi-IN',
            'ur': 'ur-PK', 'th': 'th-TH'
        };
        return languageMap[lang] || 'en-US';
    }
    
    // ===== UI STATE MANAGEMENT =====
    setLoadingState(loading) {
        const sendBtn = document.getElementById('sendBtn');
        const messageInput = document.getElementById('messageInput');
        
        if (sendBtn) {
            sendBtn.disabled = loading;
            sendBtn.classList.toggle('loading', loading);
        }
        
        if (messageInput) {
            messageInput.disabled = loading;
        }
    }
    
    showProcessingStatus(message) {
        const statusEl = document.getElementById('processingStatus');
        const titleEl = document.getElementById('statusTitle');
        const messageEl = document.getElementById('statusMessage');
        
        if (statusEl) {
            if (titleEl) titleEl.textContent = 'Processing...';
            if (messageEl) messageEl.textContent = message || 'Processing...';
            statusEl.style.display = 'block';
            statusEl.classList.add('animate__animated', 'animate__fadeIn');
        }
    }
    
    hideProcessingStatus() {
        const statusEl = document.getElementById('processingStatus');
        if (statusEl) {
            statusEl.style.display = 'none';
        }
    }
    
    updateProcessingProgress(progress) {
        const progressBar = document.getElementById('statusProgressBar');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
    }
    
    // ===== FIXED: TOGGLE FUNCTIONS - STREAMING FORCED ON =====
    toggleStreaming() {
        // FIXED: Don't actually toggle, just show status
        this.showAlert('Streaming is always enabled for best experience', 'info');
        console.log('✅ Streaming toggle called - keeping ENABLED');
    }
    
    updateStreamingUI() {
        const toggleBtn = document.getElementById('streamToggle');
        if (toggleBtn) {
            toggleBtn.classList.add('active');  // Always active
            toggleBtn.title = 'Streaming: ALWAYS ON';
        }
    }
    
    toggleTextToVoice() {
        this.textToSpeechEnabled = !this.textToSpeechEnabled;
        localStorage.setItem('tts_enabled', this.textToSpeechEnabled.toString());
        this.updateTTSUI();
        
        const status = this.textToSpeechEnabled ? 'ON' : 'OFF';
        this.showAlert(`Text-to-Speech ${status}`, 'info');
    }
    
    updateTTSUI() {
        const toggleText = document.getElementById('ttsToggleText');
        if (toggleText) {
            toggleText.textContent = `TTS: ${this.textToSpeechEnabled ? 'ON' : 'OFF'}`;
        }
    }
    
    // ===== UTILITY FUNCTIONS =====
    stripHTML(html) {
        if (!html) return '';
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;
        return tempDiv.textContent || tempDiv.innerText || '';
    }
    
    showImageFullscreen(imageSrc) {
        const modal = document.getElementById('imageModal');
        const fullscreenImage = document.getElementById('fullscreenImage');
        
        if (modal && fullscreenImage) {
            fullscreenImage.src = imageSrc;
            modal.style.display = 'flex';
            modal.classList.add('animate__animated', 'animate__fadeIn');
        }
    }
    
    hideImageFullscreen() {
        const modal = document.getElementById('imageModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }
    
    // ===== CHAT MANAGEMENT =====
    async clearCurrentChat() {
        if (!confirm('Are you sure you want to clear the current chat?')) return;
        
        try {
            const response = await fetch('/api/clear-history', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.currentSessionId })
            });
            
            if (response.ok) {
                this.clearChatDisplay();
                this.showAlert('Chat history cleared', 'success');
            } else {
                throw new Error('Failed to clear chat history');
            }
            
        } catch (error) {
            console.error('Clear chat error:', error);
            this.showAlert('Failed to clear chat', 'error');
        }
    }
    
    clearChatDisplay() {
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            const messages = chatMessages.querySelectorAll('.message');
            messages.forEach(msg => msg.remove());
            this.addWelcomeMessage();
        }
    }
    
    addWelcomeMessage() {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome-message';
        welcomeDiv.innerHTML = `
            <div class="ai-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <h3>Welcome! ✅ Streaming Mode Active</h3>
                <p>I can help you with:</p>
                <ul>
                    <li><i class="fas fa-stream"></i> Word-by-word streaming responses (like ChatGPT/Gemini)</li>
                    <li><i class="fas fa-code"></i> Code with syntax highlighting</li>
                    <li><i class="fas fa-image"></i> Advanced image analysis</li>
                    <li><i class="fas fa-volume-up"></i> Multi-language speech</li>
                    <li><i class="fas fa-history"></i> Chat session management</li>
                </ul>
            </div>
        `;
        chatMessages.appendChild(welcomeDiv);
    }
    
    // ===== MODAL MANAGEMENT =====
    showFeatures() {
        const modal = document.getElementById('featuresModal');
        if (modal) {
            modal.style.display = 'flex';
            modal.classList.add('animate__animated', 'animate__fadeIn');
        }
    }
    
    hideFeatures() {
        const modal = document.getElementById('featuresModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }
    
    // ===== ALERT SYSTEM =====
    showAlert(message, type = 'info', details = null) {
        const alertContainer = document.getElementById('alertContainer');
        if (!alertContainer) return;
        
        const alertId = 'alert-' + Date.now();
        const alertDiv = document.createElement('div');
        alertDiv.id = alertId;
        alertDiv.className = `alert ${type}`;
        
        const iconMap = {
            success: 'fas fa-check-circle',
            error: 'fas fa-times-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        
        alertDiv.innerHTML = `
            <i class="${iconMap[type] || iconMap.info}"></i>
            <div class="alert-content">
                <div class="alert-title">${message || 'Notification'}</div>
                ${details ? `<div class="alert-message">${details}</div>` : ''}
            </div>
            <button class="alert-close" onclick="app.closeAlert('${alertId}')">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        alertContainer.appendChild(alertDiv);
        
        if (type === 'success' || type === 'info') {
            setTimeout(() => this.closeAlert(alertId), 5000);
        }
        
        setTimeout(() => alertDiv.classList.add('animate__animated', 'animate__slideInRight'), 10);
    }
    
    closeAlert(alertId) {
        const alert = document.getElementById(alertId);
        if (alert) {
            alert.classList.add('animate__animated', 'animate__slideOutRight');
            setTimeout(() => alert.remove(), 300);
        }
    }
    
    // ===== CLEANUP =====
    cleanup() {
        this.stopPolling();
        if (this.isRecording && this.mediaRecorder) {
            this.mediaRecorder.stop();
        }
        if (this.speechSynthesis) {
            this.speechSynthesis.cancel();
        }
    }
}

// ===== GLOBAL FUNCTIONS =====
function selectMode(mode) {
    if (window.app) window.app.selectMode(mode);
}

function sendMessage() {
    if (window.app) window.app.sendMessage();
}

function toggleRecording() {
    if (window.app) {
        if (window.app.isRecording) {
            window.app.stopRecording();
        } else {
            window.app.startRecording();
        }
    }
}

function speakText() {
    if (window.app) window.app.speakTextFromInput();
}

function showModeSelector() {
    if (window.app) window.app.showModeSelector();
}

function newChat() {
    if (window.app) window.app.createNewSession();
}

function clearCurrentChat() {
    if (window.app) window.app.clearCurrentChat();
}

function toggleSidebar() {
    if (window.app) window.app.toggleSidebar();
}

function showFeatures() {
    if (window.app) window.app.showFeatures();
}

function hideFeatures() {
    if (window.app) window.app.hideFeatures();
}

function hideImageFullscreen() {
    if (window.app) window.app.hideImageFullscreen();
}

function toggleStreaming() {
    if (window.app) window.app.toggleStreaming();
}

function toggleTextToVoice() {
    if (window.app) window.app.toggleTextToVoice();
}

function toggleImageUpload() {
    if (window.app) window.app.toggleImageUpload();
}

function handleImageUpload(event) {
    if (window.app) window.app.handleImageUpload(event);
}

function handleImageAnalysisUpload(event) {
    if (window.app) window.app.handleImageAnalysisUpload(event);
}

function removeUploadedImage() {
    if (window.app) window.app.removeUploadedImage();
}

function removeImagePreview() {
    if (window.app) window.app.removeImagePreview();
}

function analyzeImage() {
    if (window.app) window.app.analyzeImage();
}

function analyzeImageWithText() {
    if (window.app) window.app.analyzeImageWithText();
}

// ===== CODE BLOCK UTILITIES =====
function toggleCodeView(button) {
    const codeBlock = button.closest('.code-block');
    const rendered = codeBlock.querySelector('.code-rendered');
    const raw = codeBlock.querySelector('.code-raw');
    const toggles = codeBlock.querySelectorAll('.view-toggle');
    
    const view = button.dataset.view;
    
    toggles.forEach(toggle => toggle.classList.remove('active'));
    button.classList.add('active');
    
    if (view === 'rendered') {
        rendered.style.display = 'block';
        raw.style.display = 'none';
    } else {
        rendered.style.display = 'none';
        raw.style.display = 'block';
        
        if (window.Prism) {
            Prism.highlightAllUnder(raw);
        }
    }
}

function copyCode(button) {
    const codeBlock = button.closest('.code-block');
    const code = codeBlock.querySelector('.code-rendered').textContent || 
                 codeBlock.querySelector('code').textContent;
    
    navigator.clipboard.writeText(code).then(() => {
        const originalText = button.textContent;
        button.textContent = 'Copied!';
        button.style.background = 'rgba(16, 185, 129, 0.3)';
        
        setTimeout(() => {
            button.textContent = originalText;
            button.style.background = 'rgba(16, 185, 129, 0.2)';
        }, 2000);
    }).catch(() => {
        if (window.app) {
            window.app.showAlert('Failed to copy code', 'error');
        }
    });
}

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', function() {
    window.app = new EnhancedMultimodalChatApp();
    
    console.log('✅ APPLICATION INITIALIZED - STREAMING GUARANTEED');
    console.log('✅ Streaming is ALWAYS ENABLED for smooth responses');
    
    const navbar = document.querySelector('.navbar');
    let scrollTimeout;
    
    window.addEventListener('scroll', () => {
        clearTimeout(scrollTimeout);
        
        if (navbar) {
            if (window.scrollY > 50) {
                navbar.style.background = 'rgba(17, 24, 39, 0.98)';
                navbar.style.backdropFilter = 'blur(25px)';
            } else {
                navbar.style.background = 'rgba(17, 24, 39, 0.95)';
                navbar.style.backdropFilter = 'blur(20px)';
            }
        }
        
        scrollTimeout = setTimeout(() => {}, 100);
    });
    
    document.body.style.opacity = '0';
    document.body.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        document.body.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
        document.body.style.opacity = '1';
        document.body.style.transform = 'translateY(0)';
    }, 100);
});

window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    if (window.app && e.error && !e.error.message.includes('ResizeObserver')) {
        window.app.showAlert('An unexpected error occurred', 'error');
    }
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    if (window.app && e.reason && e.reason.name !== 'AbortError') {
        window.app.showAlert('Connection issue detected', 'warning');
    }
});
