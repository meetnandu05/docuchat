// Configuration
const API_URL = window.location.origin;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const documentsList = document.getElementById('documentsList');
const messagesContainer = document.getElementById('messagesContainer');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const inputHint = document.getElementById('inputHint');
const docCount = document.getElementById('docCount');
const chunkCount = document.getElementById('chunkCount');

// State
let hasDocuments = false;

// ============================================
// INITIALIZATION
// ============================================

async function init() {
    try {
        const response = await fetch(`${API_URL}/stats`);
        if (!response.ok) throw new Error();
    } catch (error) {
        showToast('Cannot connect to backend. Make sure the server is running.', 'error');
        inputHint.textContent = '⚠️ Backend not reachable';
        return;
    }

    await refreshDocuments();
    await refreshStats();
}

// ============================================
// FILE UPLOAD
// ============================================

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
        uploadFile(file);
    } else {
        showToast('Please upload a PDF file', 'error');
    }
});

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) uploadFile(file);
});

async function uploadFile(file) {
    const MAX_SIZE_MB = 20;
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        if (!confirm(`This file is ${(file.size / 1024 / 1024).toFixed(1)}MB. Large files may take a while to process. Continue?`)) {
            return;
        }
    }

    uploadProgress.hidden = false;
    progressFill.style.width = '0%';
    progressText.textContent = `Uploading ${file.name}...`;

    const formData = new FormData();
    formData.append('file', file);

    try {
        progressFill.style.width = '30%';
        progressText.textContent = 'Processing document...';

        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        progressFill.style.width = '70%';

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        progressFill.style.width = '100%';
        progressText.textContent = 'Done!';
        showToast(`"${file.name}" processed successfully!`, 'success');

        await refreshDocuments();
        await refreshStats();
        enableChat();

    } catch (error) {
        showToast(`Upload failed: ${error.message}`, 'error');
    } finally {
        setTimeout(() => {
            uploadProgress.hidden = true;
            progressFill.style.width = '0%';
        }, 2000);
        fileInput.value = '';
    }
}

// ============================================
// DOCUMENT MANAGEMENT
// ============================================

async function refreshDocuments() {
    try {
        const response = await fetch(`${API_URL}/documents`);
        const data = await response.json();

        if (data.documents && data.documents.length > 0) {
            hasDocuments = true;
            documentsList.innerHTML = data.documents.map(doc => `
                <div class="document-item">
                    <span class="doc-name">📄 ${doc}</span>
                    <button class="delete-btn" onclick="deleteDocument('${doc}')">✕</button>
                </div>
            `).join('');
            enableChat();
        } else {
            hasDocuments = false;
            documentsList.innerHTML = '<p class="empty-state">No documents uploaded yet</p>';
            disableChat();
        }
    } catch (error) {
        console.error('Failed to fetch documents:', error);
    }
}

async function deleteDocument(name) {
    if (!confirm(`Delete "${name}"? This cannot be undone.`)) return;

    try {
        const response = await fetch(`${API_URL}/documents/${name}`, { method: 'DELETE' });
        if (response.ok) {
            showToast(`"${name}" deleted`, 'success');
            await refreshDocuments();
            await refreshStats();
        }
    } catch (error) {
        showToast('Failed to delete document', 'error');
    }
}

async function refreshStats() {
    try {
        const response = await fetch(`${API_URL}/stats`);
        const stats = await response.json();
        docCount.textContent = stats.documents || 0;
        chunkCount.textContent = stats.total_chunks || 0;
    } catch (error) {
        console.error('Failed to fetch stats:', error);
    }
}

// ============================================
// CHAT
// ============================================

function enableChat() {
    chatInput.disabled = false;
    sendBtn.disabled = false;
    inputHint.textContent = 'Ask anything about your documents';
    chatInput.placeholder = 'Ask a question about your documents...';
}

function disableChat() {
    chatInput.disabled = true;
    sendBtn.disabled = true;
    inputHint.textContent = 'Upload a document to start chatting';
}

chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = chatInput.value.trim();
    if (!question) return;

    chatInput.value = '';
    chatInput.style.height = 'auto';

    const welcome = messagesContainer.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    addMessage('user', question);
    const typingId = showTyping();

    chatInput.disabled = true;
    sendBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, n_results: 3 })
        });

        if (!response.ok) throw new Error('Failed to get response');

        const result = await response.json();
        removeTyping(typingId);
        addMessage('assistant', result.answer, result.sources);

    } catch (error) {
        removeTyping(typingId);
        addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
        showToast('Failed to get response', 'error');
    } finally {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
});

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.requestSubmit();
    }
});

function askSuggestion(question) {
    if (!hasDocuments) {
        showToast('Upload a document first', 'error');
        return;
    }
    chatInput.disabled = false;
    sendBtn.disabled = false;
    chatInput.value = question;
    chatForm.requestSubmit();
}

// ============================================
// MESSAGE RENDERING
// ============================================

function addMessage(role, content, sources = null) {
    const avatar = role === 'user' ? '👤' : '🤖';

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="sources">
                <div class="sources-title">📚 Sources</div>
                ${sources.map(s => `
                    <div class="source-item">
                        <span class="relevance">${Math.round(s.relevance * 100)}%</span>
                        <strong>${s.document}</strong> — ${s.text}
                    </div>
                `).join('')}
            </div>
        `;
    }

    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;
    messageEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            ${formatText(content)}
            ${sourcesHtml}
        </div>
    `;

    messagesContainer.appendChild(messageEl);
    scrollToBottom();
}

function formatText(text) {
    return text
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/`(.*?)`/g, '<code>$1</code>');
}

function showTyping() {
    const id = 'typing-' + Date.now();
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.id = id;
    el.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    messagesContainer.appendChild(el);
    scrollToBottom();
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// ============================================
// UTILITIES
// ============================================

function showToast(message, type = 'error') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ============================================
// STARTUP
// ============================================

init();
