<div align="center">
             <img src="/MeMyselfAi.png" width="400" />
             <h1>MeMyselfAi</h1>
</div>

# MeMyselfAI - Cross-platform Desktop App

A Cross Platform chat application that wraps ollama and llama.cpp build.

## Features
- 💬 Clean chat interface
- 🤖 Local model support (uses llama.cpp)
- ☁️ Ollama Model Library integration.
- ⚡ Real-time streaming responses
- 📁 Model management

<div align="center">
             <img src="/window1.png" width="700" />
             
</div>

## Setup

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Choose a Backend
Select from Local, (llama-server) Ollama, (ollama serve) or HuggingFace. (requires HF API key)
More backends coming soon! (gRPC, WebSockets and HTTP+SSE support is planned)

### 3. Add Your Models
Choose, Configure models in Files/Manage Models. 

### 4. Run from code
```bash
python3 main.py
```

## Configuration

The app stores:
- **config.json** - App settings (backend settings, model directory)
- **conversation_history.json** - Chat history (optional)
- **system_prompts** - Manage custom Ai pesonalities!

## Usage

1. Launch the app
2. Configure backend, generation parameters.
3. Select a model from the dropdown
4. Start chatting!
(Ollama integration now built in and runs in background.)

<div align="center">
             <img src="/window2.png" width="700" />

<div align="center">
             <img src="/window3.png" width="700" />

## System Prompt Manager

- Select, Create and save your System Prompts!
- Give your models some personality!
- Custom tailor them for specific tasks!

<div align="center">
             <img src="/window4.png" width="700" />

## Troubleshooting

**Model not loading?**
- Verify model file exists and is readable
- Check terminal output for errors

**No response?**
- Ensure llama.cpp binary works: `./llama-server --version`
- Check model format is GGUF
- Try a smaller model first

## Future Features
- [ ] Network server mode (DONE!)
- [ ] gRPC/WebSocket/HTTP+SSE support (DONE)
- [ ] Multi-model support
- [ ] Conversation export
- [ ] System prompt templates (DONE!)

