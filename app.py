<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat WebUI</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/font-awesome.min.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/atom-one-dark.min.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/katex.min.css') }}">
    <script src="{{ url_for('static', filename='js/highlight.min.js') }}"></script>
    <script src="{{ url_for('static', filename='js/marked.min.js') }}"></script>
    <script src="{{ url_for('static', filename='js/dexie.min.js') }}"></script>
    <script src="{{ url_for('static', filename='js/mammoth.browser.min.js') }}"></script>
    <script src="{{ url_for('static', filename='js/pdf.min.js') }}"></script>
    <script src="{{ url_for('static', filename='js/react.production.min.js') }}"></script>
    <script src="{{ url_for('static', filename='js/react-dom.production.min.js') }}"></script>
    <script src="{{ url_for('static', filename='js/babel.min.js') }}"></script>
    <script src="{{ url_for('static', filename='js/svelte.min.js') }}"></script>
    <script src="{{ url_for('static', filename='js/katex.min.js') }}"></script>
    <script src="{{ url_for('static', filename='js/index.umd.min.js') }}"></script>
    <script>
        window.pdfjsLib = window.pdfjsLib || {};
        window.pdfjsLib.GlobalWorkerOptions = window.pdfjsLib.GlobalWorkerOptions || {};
        window.pdfjsLib.GlobalWorkerOptions.workerSrc = '{{ url_for("static", filename="js/pdf.worker.min.js") }}';
    </script>
    <link rel="shortcut icon" href="{{ url_for('static', filename='images/favicon.ico') }}" type="image/x-icon">
    <link rel="icon" href="{{ url_for('static', filename='images/favicon.ico') }}" type="image/x-icon">
</head>
<body>
    <div class="container">
        <div class="left-side">
            <div id="top-part">
                <button type="button" id="close-sidebar"><img src="/static/images/icons/sidebar.svg" alt="Toggle Sidebar" class="icon-svg"></button>
                <button type="button" id="new-chat"><img src="/static/images/icons/chat.svg" alt="New Chat" class="icon-svg"></button>
            </div>
            <div id="middle-part">
                <button type="button" id="private-chat"><i class="fa fa-user-circle-o" aria-hidden="true"></i>Private Chat</button>
                <button type="button" id="deep-query"><i class="fa fa-qrcode" aria-hidden="true"></i>Deep Query</button>
                <div id="chat-history">
                </div>
            </div>
            <div id="bottom-part">
                <div class="export-buttons">
                    <button type="button" class="export-button" id="export-markdown">
                        <i class="fa fa-file-text-o" aria-hidden="true"></i><span>Export as Markdown</span>
                    </button>
                    <button type="button" class="export-button" id="export-json">
                        <i class="fa fa-file-code-o" aria-hidden="true"></i><span>Export as JSON</span>
                    </button>
                </div>
            </div>
        </div>
        <div class="right-side">
            <div class="top-panel">
                <div class="custom-select">
                    <div class="select-selected" onclick="toggleDropdown(this)">Select Model<i class="fa fa-angle-down" aria-hidden="true"></i></div>
                    <div class="select-items" id="select-items">
                        <input type="text" id="model-search" class="model-search" placeholder="Search a model">
                    </div>
                </div>
                <button type="button" id="additional-setting" onclick="openAdditionalPopup()"><i class="fa fa-sliders fa-inverse" aria-hidden="true"></i></button>
                <button type="button" id="user-setting" onclick="openPopup()"><i class="fa fa-cog fa-inverse" aria-hidden="true"></i></button>
            </div>
            <div class="middle-panel">
                <div class="chat-wrapper">
                    <div id="scroller">
                        <div class="message-container" id="chat-messages"></div>
                        <div id="anchor"></div>
                    </div>
                </div>
            </div>
            <div class="bottom-panel">
                <form action="" onsubmit="sendMessage(event)">
                    <button type="button" id="upload-files"><i class="fa fa-paperclip fa-inverse"></i></button>
                    <textarea type="text" name="user-input" id="user-input" placeholder="Enter your message" cols="50"></textarea>
                    <button type="submit" id="submit-button"><i class="fa fa-arrow-circle-up fa-inverse"></i></button>
                </form>
            </div>
        </div>
        <div id="settings-popup" class="popup">
            <div class="popup-content">
                <span class="close-btn" onclick="closePopup()">&times;</span>
                <h2><i class="fa fa-cog" aria-hidden="true"></i> Settings</h2>
                <form>
                    <div class="input-group">
                        <label for="base-url">
                            <i class="fa fa-link" aria-hidden="true"></i> Base URL
                        </label>
                        <input type="text" id="base-url" name="base-url" placeholder="Example: https://openrouter.ai/api/v1">
                    </div>

                    <div class="input-group">
                        <label for="api-key">
                            <i class="fa fa-key" aria-hidden="true"></i> API Key
                        </label>
                        <div class="password-input-wrapper">
                            <input type="password" id="api-key" name="api-key" placeholder="Enter your API Key">
                            <button type="button" class="toggle-password">
                                <i class="fa fa-eye" aria-hidden="true"></i>
                            </button>
                        </div>
                    </div>
                    
                    <button type="button" onclick="saveSettings()">
                        <i class="fa fa-save" aria-hidden="true"></i> Save Settings
                    </button>
                </form>
            </div>
        </div>
        <div id="additional-settings-popup" class="popup">
            <div class="popup-content">
                <span class="close-btn" onclick="closeAdditionalPopup()">&times;</span>
                <h2><i class="fa fa-sliders" aria-hidden="true"></i> Additional Settings</h2>
                <form>
                    <div class="input-group">
                        <label for="system-content">
                            <i class="fa fa-commenting" aria-hidden="true"></i> System Prompt
                        </label>
                        <textarea 
                            id="system-content" 
                            name="system-content" 
                            placeholder="Enter system content that defines AI behavior"
                            rows="4"
                        ></textarea>
                    </div>

                    <div class="input-group">
                        <label>
                            <i class="fa fa-cogs" aria-hidden="true"></i> Model Parameters
                        </label>
                        <div class="parameter-buttons">
                            <button type="button" class="parameter-button" data-mode="precise">Precise</button>
                            <button type="button" class="parameter-button" data-mode="balanced">Balanced</button>
                            <button type="button" class="parameter-button" data-mode="creative">Creative</button>
                            <button type="button" class="parameter-button" data-mode="custom">Custom</button>
                        </div>
                        <textarea 
                            id="model-parameters" 
                            name="model-parameters" 
                            placeholder="Example: temperature=0.7, top_p=0.95"
                            rows="3"
                            style="display: none;"
                        ></textarea>
                    </div>

                    <button type="button" onclick="saveAdditionalSettings()">
                        <i class="fa fa-save" aria-hidden="true"></i> Save Settings
                    </button>
                </form>
            </div>
        </div>
    </div>

    <script src="{{ url_for('static', filename='js/scripts.js') }}"></script>
</body>
</html>
