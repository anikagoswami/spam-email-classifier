/**
 * MAILLY - Live Spam Email Classification System
 * Frontend JavaScript for email classification interface
 */

// Configuration
const API_BASE_URL = 'http://localhost:5000';
const API_ENDPOINTS = {
    PREDICT: '/predict',
    MODELS: '/models',
    HEALTH: '/health'
};

// State management
const state = {
    currentSection: 'inbox',
    selectedEmail: null,
    classificationHistory: [],
    modelInfo: null
};

// DOM Elements
const elements = {
    // Navigation
    navItems: document.querySelectorAll('.nav-item'),
    sections: document.querySelectorAll('.section'),
    
    // Inbox
    emailList: document.querySelector('.email-list'),
    emailPanel: document.getElementById('email-panel'),
    closePanelBtn: document.getElementById('close-panel'),
    
    // Classification
    emailInput: document.getElementById('email-input'),
    classifyBtn: document.getElementById('classify-btn'),
    clearBtn: document.getElementById('clear-btn'),
    sampleBtn: document.getElementById('sample-btn'),
    
    // Results
    resultStatus: document.getElementById('result-status'),
    predictionValue: document.getElementById('prediction-value'),
    confidenceValue: document.getElementById('confidence-value'),
    modelValue: document.getElementById('model-value'),
    timestampValue: document.getElementById('timestamp-value'),
    confidenceBar: document.getElementById('confidence-bar'),
    barPercentage: document.getElementById('bar-percentage'),
    feedbackBtn: document.getElementById('feedback-btn'),
    copyBtn: document.getElementById('copy-btn'),
    
    // Spam folder
    spamList: document.getElementById('spam-list'),
    spamCount: document.getElementById('spam-count'),
    
    // Analytics
    totalClassifications: document.getElementById('total-classifications'),
    spamDetected: document.getElementById('spam-detected'),
    hamDetected: document.getElementById('ham-detected'),
    avgConfidence: document.getElementById('avg-confidence'),
    modelDetails: document.getElementById('model-details')
};

// Sample emails for testing
const sampleEmails = [
    {
        subject: "Congratulations! You've won $1000!",
        content: "Dear User, Congratulations! You have been selected as a winner in our monthly lottery. You have won $1000 in cash! To claim your prize, please click on the link below and provide your banking information. This is a limited time offer and expires in 24 hours. Don't miss out on this amazing opportunity!",
        isSpam: true
    },
    {
        subject: "Meeting scheduled for tomorrow",
        content: "Hi John, I hope this email finds you well. I wanted to let you know that I've scheduled our meeting for tomorrow at 10 AM in the conference room. Please let me know if this time works for you or if you need to reschedule. Looking forward to our discussion about the project.",
        isSpam: false
    },
    {
        subject: "URGENT: Your account has been compromised",
        content: "SECURITY ALERT: Your account has been compromised due to suspicious activity. For your security, we have temporarily suspended your account. To restore access immediately, please verify your identity by clicking the link below and entering your personal information. Failure to act within 24 hours will result in permanent account closure.",
        isSpam: true
    },
    {
        subject: "Dinner this weekend?",
        content: "Hey sweetie, I was thinking we could have dinner together this weekend. How about Saturday night? There's a new Italian restaurant that just opened downtown that I've been wanting to try. Let me know what you think and what time works best for you. Love you!",
        isSpam: false
    },
    {
        subject: "50% OFF Everything - Today Only!",
        content: "Huge savings alert! For today only, get 50% off everything in our store. Use code SAVE50 at checkout to redeem your discount. This is our biggest sale of the year, so don't miss out! Hurry, this offer expires at midnight. Shop now and save big on all your favorite products.",
        isSpam: true
    }
];

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Set up event listeners
    setupEventListeners();
    
    // Load model information
    loadModelInfo();
    
    // Load classification history from localStorage
    loadClassificationHistory();
    
    // Load spam folder
    updateSpamFolder();
    
    // Update analytics
    updateAnalytics();
    
    console.log('MAILLY application initialized');
}

/**
 * Set up all event listeners
 */
function setupEventListeners() {
    // Navigation
    elements.navItems.forEach(item => {
        item.addEventListener('click', handleNavigation);
    });
    
    // Email interactions
    if (elements.emailList) {
        elements.emailList.addEventListener('click', handleEmailClick);
    }
    
    elements.closePanelBtn.addEventListener('click', closeEmailPanel);
    
    // Classification
    elements.classifyBtn.addEventListener('click', handleClassification);
    elements.clearBtn.addEventListener('click', clearClassification);
    elements.sampleBtn.addEventListener('click', loadSampleEmail);
    elements.feedbackBtn.addEventListener('click', handleFeedback);
    elements.copyBtn.addEventListener('click', handleCopyResult);
    
    // Analytics
    document.getElementById('analytics-section').addEventListener('click', loadModelInfo);
}

/**
 * Handle navigation between sections
 */
function handleNavigation(e) {
    const section = e.currentTarget.getAttribute('data-section');
    
    // Update active navigation item
    elements.navItems.forEach(item => {
        item.classList.remove('active');
    });
    e.currentTarget.classList.add('active');
    
    // Show/hide sections
    elements.sections.forEach(sectionEl => {
        sectionEl.classList.remove('active');
    });
    
    const targetSection = document.getElementById(`${section}-section`);
    if (targetSection) {
        targetSection.classList.add('active');
        state.currentSection = section;
    }
}

/**
 * Handle email clicks in inbox
 */
function handleEmailClick(e) {
    const emailItem = e.target.closest('.email-item');
    if (!emailItem) return;
    
    // Update active email
    document.querySelectorAll('.email-item').forEach(item => {
        item.classList.remove('active');
    });
    emailItem.classList.add('active');
    
    // Show email panel
    showEmailPanel(emailItem);
}

/**
 * Show email reading panel
 */
function showEmailPanel(emailItem) {
    const emailId = emailItem.getAttribute('data-email-id');
    const sender = emailItem.querySelector('.email-sender').textContent;
    const subject = emailItem.querySelector('.email-subject').textContent;
    const preview = emailItem.querySelector('.email-preview').textContent;
    const time = emailItem.querySelector('.email-time').textContent;
    
    // Update panel content
    document.getElementById('email-from').textContent = sender;
    document.getElementById('email-subject').textContent = subject;
    document.getElementById('email-date').textContent = time;
    document.getElementById('email-body-content').textContent = preview;
    
    // Show panel
    elements.emailPanel.classList.add('active');
}

/**
 * Close email panel
 */
function closeEmailPanel() {
    elements.emailPanel.classList.remove('active');
}

/**
 * Handle email classification
 */
async function handleClassification() {
    const emailText = elements.emailInput.value.trim();
    const modelSelect = document.getElementById('model-select');
    const selectedModel = modelSelect ? modelSelect.value : '';
    
    if (!emailText) {
        showMessage('Please enter email text to classify', 'error');
        return;
    }
    
    // Show loading state
    setClassificationLoading(true);
    
    try {
        const result = await classifyEmail(emailText, selectedModel);
        displayClassificationResult(result);
        updateClassificationHistory(result);
        updateAnalytics();
    } catch (error) {
        console.error('Classification error:', error);
        showMessage('Error during classification. Please try again.', 'error');
    } finally {
        setClassificationLoading(false);
    }
}

/**
 * Classify email using API
 */
async function classifyEmail(emailText, model = '') {
    const payload = {
        email: emailText
    };
    
    if (model) {
        payload.model = model;
    }
    
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.PREDICT}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    
    if (result.error) {
        throw new Error(result.error);
    }
    
    return result;
}

/**
 * Display classification result
 */
function displayClassificationResult(result) {
    // Update result values
    elements.predictionValue.textContent = result.prediction;
    elements.confidenceValue.textContent = `${(result.confidence * 100).toFixed(2)}%`;
    elements.modelValue.textContent = result.model_used;
    elements.timestampValue.textContent = new Date(result.timestamp).toLocaleString();
    
    // Update status indicator
    const isSpam = result.prediction === 'spam';
    elements.resultStatus.textContent = isSpam ? 'SPAM DETECTED' : 'NOT SPAM';
    elements.resultStatus.className = `result-status ${isSpam ? 'spam' : 'normal'}`;
    
    // Update confidence bar
    const confidencePercent = result.confidence * 100;
    elements.confidenceBar.style.width = `${confidencePercent}%`;
    elements.confidenceBar.className = `bar-fill ${isSpam ? 'spam' : 'normal'}`;
    elements.barPercentage.textContent = `${confidencePercent.toFixed(1)}%`;
    elements.barPercentage.style.color = isSpam ? '#990000' : '#13545A';
    
    // Enable buttons
    elements.feedbackBtn.disabled = false;
    elements.copyBtn.disabled = false;
    
    // Add spam email to spam folder if detected
    if (isSpam) {
        addSpamEmail(result);
    }
    
    // Add success message
    showMessage(`Email classified as ${result.prediction} with ${(result.confidence * 100).toFixed(1)}% confidence`, 'success');
}

/**
 * Set classification loading state
 */
function setClassificationLoading(loading) {
    if (loading) {
        elements.classifyBtn.disabled = true;
        elements.classifyBtn.innerHTML = '<span class="btn-icon">⏳</span> Classifying...';
        elements.resultStatus.textContent = 'Classifying...';
        elements.resultStatus.className = 'result-status loading';
    } else {
        elements.classifyBtn.disabled = false;
        elements.classifyBtn.innerHTML = '<span class="btn-icon">🔍</span> Classify Email';
    }
}

/**
 * Clear classification results
 */
function clearClassification() {
    elements.emailInput.value = '';
    elements.predictionValue.textContent = '-';
    elements.confidenceValue.textContent = '-';
    elements.modelValue.textContent = '-';
    elements.timestampValue.textContent = '-';
    elements.resultStatus.textContent = 'No prediction yet';
    elements.resultStatus.className = 'result-status';
    elements.confidenceBar.style.width = '0%';
    elements.barPercentage.textContent = '0%';
    elements.feedbackBtn.disabled = true;
    elements.copyBtn.disabled = true;
}

/**
 * Load sample email
 */
function loadSampleEmail() {
    const randomEmail = sampleEmails[Math.floor(Math.random() * sampleEmails.length)];
    elements.emailInput.value = randomEmail.content;
    showMessage(`Loaded sample email: ${randomEmail.subject}`, 'success');
}

/**
 * Handle feedback/report incorrect button
 */
function handleFeedback() {
    const prediction = elements.predictionValue.textContent;
    const confidence = elements.confidenceValue.textContent;
    const model = elements.modelValue.textContent;
    const emailText = elements.emailInput.value;
    
    if (!prediction || prediction === '-') {
        showMessage('No prediction to report. Please classify an email first.', 'error');
        return;
    }
    
    // Create feedback object
    const feedback = {
        email: emailText,
        predicted_as: prediction,
        confidence: confidence,
        model_used: model,
        timestamp: new Date().toISOString(),
        user_correction: prediction === 'spam' ? 'not spam' : 'spam'
    };
    
    // Store feedback in localStorage
    let feedbackHistory = JSON.parse(localStorage.getItem('feedbackHistory') || '[]');
    feedbackHistory.push(feedback);
    localStorage.setItem('feedbackHistory', JSON.stringify(feedbackHistory));
    
    showMessage(`Feedback recorded: Email marked as incorrect (was ${prediction}, should be ${feedback.user_correction})`, 'success');
    console.log('Feedback stored:', feedback);
}

/**
 * Handle copy result button
 */
function handleCopyResult() {
    const prediction = elements.predictionValue.textContent;
    const confidence = elements.confidenceValue.textContent;
    const model = elements.modelValue.textContent;
    const timestamp = elements.timestampValue.textContent;
    
    const resultText = `MAILLY Classification Result\n` +
                      `Prediction: ${prediction}\n` +
                      `Confidence: ${confidence}\n` +
                      `Model: ${model}\n` +
                      `Time: ${timestamp}`;
    
    navigator.clipboard.writeText(resultText).then(() => {
        showMessage('Classification result copied to clipboard!', 'success');
    }).catch(() => {
        showMessage('Failed to copy to clipboard. Please try again.', 'error');
    });
}

/**
 * Add spam email to spam folder
 */
function addSpamEmail(result) {
    // Extract email subject from the input (assume first line is subject or extract key info)
    const emailText = elements.emailInput.value;
    const lines = emailText.split('\n');
    const subject = lines[0].substring(0, 60) || 'Unknown Spam Email';
    const preview = emailText.substring(0, 80).replace(/\n/g, ' ') + '...';
    
    // Create spam email item
    const spamEmailId = 'spam-' + Date.now();
    const spamEmail = {
        id: spamEmailId,
        subject: subject,
        preview: preview,
        confidence: result.confidence,
        model: result.model_used,
        timestamp: new Date(result.timestamp)
    };
    
    // Store in localStorage
    let spamEmails = JSON.parse(localStorage.getItem('spamEmails') || '[]');
    spamEmails.unshift(spamEmail);
    localStorage.setItem('spamEmails', JSON.stringify(spamEmails));
    
    // Update UI
    updateSpamFolder();
}

/**
 * Update spam folder display
 */
function updateSpamFolder() {
    const spamEmails = JSON.parse(localStorage.getItem('spamEmails') || '[]');
    
    if (!elements.spamList) return;
    
    // Clear spam list
    elements.spamList.innerHTML = '';
    
    if (spamEmails.length === 0) {
        elements.spamList.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📭</div>
                <p>No spam emails yet. Classified emails will appear here.</p>
            </div>
        `;
        if (elements.spamCount) elements.spamCount.textContent = '0';
        return;
    }
    
    // Add spam emails to list
    spamEmails.forEach(email => {
        const confidencePercent = (email.confidence * 100).toFixed(1);
        const emailItem = document.createElement('div');
        emailItem.className = 'email-item spam';
        emailItem.innerHTML = `
            <div class="email-subject">${email.subject}</div>
            <div class="email-preview">${email.preview}</div>
            <div class="email-spam-info">
                <span class="confidence-tag">${confidencePercent}% confidence • ${email.model}</span>
                <span class="email-time">${new Date(email.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="email-label spam-label">SPAM</div>
        `;
        elements.spamList.appendChild(emailItem);
    });
    
    // Update count
    if (elements.spamCount) elements.spamCount.textContent = spamEmails.length;
}

/**
 * Update classification history
 */
function updateClassificationHistory(result) {
    const historyEntry = {
        email: elements.emailInput.value.substring(0, 100) + '...',
        prediction: result.prediction,
        confidence: result.confidence,
        model: result.model_used,
        timestamp: result.timestamp
    };
    
    state.classificationHistory.unshift(historyEntry);
    
    // Keep only last 50 entries
    if (state.classificationHistory.length > 50) {
        state.classificationHistory = state.classificationHistory.slice(0, 50);
    }
    
    // Save to localStorage
    localStorage.setItem('mailly_classification_history', JSON.stringify(state.classificationHistory));
}

/**
 * Load classification history from localStorage
 */
function loadClassificationHistory() {
    const savedHistory = localStorage.getItem('mailly_classification_history');
    if (savedHistory) {
        try {
            state.classificationHistory = JSON.parse(savedHistory);
        } catch (e) {
            console.error('Error loading classification history:', e);
            state.classificationHistory = [];
        }
    }
}

/**
 * Update analytics dashboard
 */
function updateAnalytics() {
    const total = state.classificationHistory.length;
    const spamCount = state.classificationHistory.filter(h => h.prediction === 'spam').length;
    const hamCount = total - spamCount;
    const avgConfidence = total > 0 
        ? (state.classificationHistory.reduce((sum, h) => sum + h.confidence, 0) / total * 100).toFixed(1)
        : 0;
    
    elements.totalClassifications.textContent = total;
    elements.spamDetected.textContent = spamCount;
    elements.hamDetected.textContent = hamCount;
    elements.avgConfidence.textContent = `${avgConfidence}%`;
}

/**
 * Load model information
 */
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.MODELS}`);
        if (response.ok) {
            const modelInfo = await response.json();
            state.modelInfo = modelInfo;
            displayModelInfo(modelInfo);
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

/**
 * Display model information
 */
function displayModelInfo(modelInfo) {
    if (!modelInfo || modelInfo.error) {
        elements.modelDetails.innerHTML = '<p class="message error">Model information not available. Please train models first.</p>';
        return;
    }
    
    const bestModel = modelInfo.best_model;
    const results = modelInfo.results[bestModel];
    
    elements.modelDetails.innerHTML = `
        <div class="model-summary">
            <h4>Active Model: ${bestModel}</h4>
            <p><strong>Best Model (by F1 Score):</strong> ${bestModel}</p>
            <p><strong>F1 Score:</strong> ${(results.f1_score * 100).toFixed(2)}%</p>
            <p><strong>Accuracy:</strong> ${(results.accuracy * 100).toFixed(2)}%</p>
            <p><strong>Precision:</strong> ${(results.precision * 100).toFixed(2)}%</p>
            <p><strong>Recall:</strong> ${(results.recall * 100).toFixed(2)}%</p>
        </div>
        <div class="model-stats">
            <h5>Model Performance Comparison:</h5>
            ${Object.entries(modelInfo.results).map(([model, metrics]) => `
                <div class="model-stat-item">
                    <span class="model-name">${model}</span>
                    <span class="model-score">${(metrics.f1_score * 100).toFixed(1)}%</span>
                </div>
            `).join('')}
        </div>
    `;
}

/**
 * Show message to user
 */
function showMessage(message, type = 'success') {
    // Remove existing messages
    const existingMessage = document.querySelector('.message');
    if (existingMessage) {
        existingMessage.remove();
    }
    
    // Create new message
    const messageEl = document.createElement('div');
    messageEl.className = `message ${type}`;
    messageEl.textContent = message;
    
    // Add to appropriate section
    const currentSection = document.querySelector(`#${state.currentSection}-section`);
    if (currentSection) {
        currentSection.insertBefore(messageEl, currentSection.firstChild);
    }
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (messageEl.parentNode) {
            messageEl.remove();
        }
    }, 3000);
}

/**
 * Utility function to format confidence as percentage
 */
function formatConfidence(confidence) {
    return `${(confidence * 100).toFixed(1)}%`;
}

// Export functions for testing
window.mailly = {
    classifyEmail,
    loadModelInfo,
    updateAnalytics,
    showMessage
};