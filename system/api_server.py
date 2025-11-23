from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import asyncio
from pathlib import Path
import base64

app = FastAPI(title="Local LLM API")

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    language: Optional[str] = None

class AgentRequest(BaseModel):
    agent_name: str
    task: str

class CodeRequest(BaseModel):
    description: str
    language: str = "python"
    context_files: Optional[List[str]] = None

class SpeechRequest(BaseModel):
    text: str
    language: str = "en"

class TrainingRequest(BaseModel):
    model_name: str
    dataset_path: Optional[str] = None
    config: Optional[Dict] = None

class APIServer:
    def __init__(self, llm_engine, speech_engine, agent_manager, coding_assistant):
        self.llm_engine = llm_engine
        self.speech_engine = speech_engine
        self.agent_manager = agent_manager
        self.coding_assistant = coding_assistant

        self.setup_routes()

    def setup_routes(self):

        @app.get("/")
        async def root():
            return {
                "name": "Local LLM System API",
                "version": "1.0.0",
                "endpoints": [
                    "/chat",
                    "/agent",
                    "/code",
                    "/speech/transcribe",
                    "/speech/synthesize"
                ]
            }

        @app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            try:
                response = self.llm_engine.generate(
                    request.message,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )

                return ChatResponse(
                    response=response,
                    language=request.language
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/chat/stream")
        async def chat_stream(request: ChatRequest):
            async def generate():
                response = self.llm_engine.generate(
                    request.message,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )

                for chunk in response.split():
                    yield f"data: {chunk}\n\n"
                    await asyncio.sleep(0.01)

            return StreamingResponse(generate(), media_type="text/event-stream")

        @app.post("/agent/create")
        async def create_agent(name: str):
            try:
                agent = self.agent_manager.create_agent(name)
                return {"agent_id": agent.id, "name": agent.name}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/agent/list")
        async def list_agents():
            try:
                agents = self.agent_manager.list_agents()
                return {"agents": agents}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/agent/run")
        async def run_agent(request: AgentRequest):
            try:
                agent = self.agent_manager.get_agent_by_name(request.agent_name)
                if not agent:
                    raise HTTPException(status_code=404, detail="Agent not found")

                response = self.agent_manager.run_agent(agent.id, request.task)
                return {"response": response}

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/code/generate")
        async def generate_code(request: CodeRequest):
            try:
                code = self.coding_assistant.generate_code(
                    request.description,
                    request.language
                )
                return {"code": code, "language": request.language}

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/code/assist")
        async def code_assist(request: CodeRequest):
            try:
                response = self.coding_assistant.assist(
                    request.description,
                    request.language,
                    request.context_files
                )
                return {"response": response}

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/code/review")
        async def code_review(code: str, language: str = "python"):
            try:
                review = self.coding_assistant.code_review(code, language)
                return {"review": review}

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/speech/transcribe")
        async def transcribe_audio(file: UploadFile = File(...), language: Optional[str] = None):
            try:
                temp_path = Path("/tmp") / file.filename
                with open(temp_path, "wb") as f:
                    content = await file.read()
                    f.write(content)

                transcription = self.speech_engine.transcribe_file(
                    str(temp_path),
                    language
                )

                temp_path.unlink()

                return {"transcription": transcription}

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/speech/synthesize")
        async def synthesize_speech(request: SpeechRequest):
            try:
                output_path = self.speech_engine.text_to_speech(
                    request.text,
                    request.language,
                    play=False
                )

                with open(output_path, "rb") as f:
                    audio_data = f.read()

                audio_base64 = base64.b64encode(audio_data).decode()

                return {
                    "audio": audio_base64,
                    "format": "wav"
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/models/list")
        async def list_models():
            try:
                models = self.llm_engine.get_available_models()
                return {"models": models}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/models/load")
        async def load_model(model_name: str, use_vllm: bool = True):
            try:
                self.llm_engine.load_model(model_name, use_vllm)
                return {"status": "loaded", "model": model_name}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/models/info")
        async def model_info():
            try:
                info = self.llm_engine.get_model_info()
                return info
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "llm_loaded": self.llm_engine.loaded_model_name is not None,
                "agents_active": len(self.agent_manager.agents)
            }

    def start(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(app, host=host, port=port)
