document.addEventListener('DOMContentLoaded', () => {
    const terminalOutput = document.getElementById('terminal-output');
    const inputField = document.getElementById('query-input');
    const sendButton = document.getElementById('send-btn');
    const micButton = document.getElementById('mic-btn');

    // Helper to append a line to the terminal
    function appendTerminalLine(text, isUser = false) {
        // Remove blink cursor from previous line if it exists
        const cursor = document.querySelector('.cursor-blink');
        if (cursor) {
            cursor.remove();
        }

        const p = document.createElement('p');
        p.style.marginBottom = '8px';
        
        if (isUser) {
            p.innerHTML = `<span style="color: var(--color-secondary-fixed);">> USER: ${text}</span>`;
        } else {
            p.innerHTML = `> ${text}`;
        }
        
        terminalOutput.appendChild(p);
        
        // Add cursor to a new line
        const cursorP = document.createElement('p');
        cursorP.style.marginTop = '16px';
        cursorP.style.opacity = '0.5';
        cursorP.innerHTML = `<span class="cursor-blink"></span>`;
        terminalOutput.appendChild(cursorP);

        // Scroll to bottom
        terminalOutput.scrollTop = terminalOutput.scrollHeight;
    }

    // Handle sending a text query
    function handleSend() {
        const query = inputField.value.trim();
        if (query) {
            appendTerminalLine(query, true);
            inputField.value = '';
            
            // Mock backend response
            setTimeout(() => {
                appendTerminalLine('PROCESSING QUERY...');
            }, 500);
            
            setTimeout(() => {
                appendTerminalLine('RETRIEVING CONTEXT FROM QDRANT...');
            }, 1500);
            
            setTimeout(() => {
                appendTerminalLine('SYSTEM: This is a placeholder response. Backend integration pending.', false);
            }, 3000);
        }
    }

    sendButton.addEventListener('click', handleSend);
    inputField.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSend();
        }
    });

    // Handle Mic button toggle
    let isRecording = false;
    micButton.addEventListener('click', () => {
        isRecording = !isRecording;
        if (isRecording) {
            micButton.classList.add('active');
            micButton.style.backgroundColor = 'var(--color-secondary-fixed)';
            micButton.style.color = 'var(--color-black)';
            appendTerminalLine('LISTENING FOR AUDIO INPUT...');
        } else {
            micButton.classList.remove('active');
            micButton.style.backgroundColor = 'transparent';
            micButton.style.color = 'var(--color-secondary-fixed)';
            appendTerminalLine('AUDIO RECORDING STOPPED. PROCESSING...');
            
            // Mock transcription
            setTimeout(() => {
                appendTerminalLine('TRANSCRIPT: "Hello Hacker House Goa"', true);
            }, 1000);
            setTimeout(() => {
                appendTerminalLine('SYSTEM: Voice processing placeholder. Backend integration pending.');
            }, 2500);
        }
    });
});
