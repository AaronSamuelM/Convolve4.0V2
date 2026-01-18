"""Optimized FastAPI Mental Health API - Modular Version"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import os
import uuid
import asyncio
import sys

from mentalhealth import MentalHealthAssistant, AsyncMultimodalProcessor

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from hashlib import sha256 as _sha256

def hash_password(p: str) -> str:
    """Optimized password hashing - avoid dot operations"""
    encode = str.encode
    hexdigest = _sha256(encode(p)).hexdigest
    return hexdigest()


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ChatRequest(BaseModel):
    user_id: str
    query: str
    is_guest: bool = False


app = FastAPI(
    title="Mental Health API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class AssistantPool:
    """Connection pool pattern for assistant instances"""
    def __init__(self, max_size: int = 10):
        self._pool = []
        self._max_size = max_size
        self._lock = asyncio.Lock()
    
    async def acquire(self, user_id: str):
        async with self._lock:
            for assistant in self._pool:
                if assistant.user_id == user_id:
                    self._pool.remove(assistant)
                    return assistant
            
            return MentalHealthAssistant(user_id)
    
    async def release(self, assistant):
        async with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(assistant)


assistant_pool = AssistantPool()


@app.get("/")
async def home():
    """Home endpoint"""
    return JSONResponse({
        "status": "ok",
        "message": "Mental Health API is running",
        "version": "2.0.0"
    })


@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse({"status": "healthy"})


@app.post("/auth/register")
async def register(request: RegisterRequest):
    """User registration endpoint"""
    try:
        name = request.name
        email = request.email
        password = request.password
        
        password_hash = hash_password(password)
        
        temp_assistant = MentalHealthAssistant(user_id="temp_check")
        existing = await asyncio.to_thread(
            temp_assistant.qdrant.get_user_by_email, email
        )
        
        if existing:
            raise HTTPException(status_code=409, detail="User exists")
        
        user_id = str(uuid.uuid4())
        
        assistant = MentalHealthAssistant(user_id)
        await asyncio.to_thread(
            assistant.create_user_profile,
            name=name,
            email=email,
            password_hash=password_hash
        )
        
        return JSONResponse({
            "user_id": user_id,
            "email": email
        })
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/auth/login")
async def login(request: LoginRequest):
    """User login endpoint"""
    try:
        email = request.email
        password = request.password
        temp_assistant = MentalHealthAssistant(user_id="temp_check")
        user_profile = await asyncio.to_thread(
            temp_assistant.qdrant.get_user_by_email, email
        )
        
        if not user_profile:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        stored_hash = user_profile.password_hash
        if stored_hash != hash_password(password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        return JSONResponse({"user_id": user_profile.user_id})
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/auth/guest")
async def guest():
    """Guest access endpoint"""
    try:
        return JSONResponse({"user_id": f"guest_{uuid.uuid4()}"})
    except Exception as e:
        print(f"[API] Guest error: {str(e)}")
        raise HTTPException(status_code=500, detail="Guest creation failed")
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with async processing"""
    try:
        user_id = request.user_id
        query = request.query
        is_guest = request.is_guest
        
        if is_guest:
            assistant = MentalHealthAssistant(user_id, is_guest=True)
            result = await assistant.process_query_async(query) 
            return JSONResponse(result)
        assistant = await assistant_pool.acquire(user_id)
        try:
            await asyncio.to_thread(assistant.initialize)
            result = await assistant.process_query_async(query)
            return JSONResponse(result)
        finally:
            await assistant_pool.release(assistant)
    except Exception as e:
        print(f"[API] Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    is_guest: str = Form("false")
):
    """File upload endpoint with multimodal processing"""
    try:
        is_guest_bool = is_guest.lower() == "true"
        filename = f"{uuid.uuid4()}_{file.filename}"
        path = os.path.join(UPLOAD_DIR, filename)
        
        contents = await file.read()
        with open(path, 'wb') as f:
            f.write(contents)
        
        if is_guest_bool:
            _, meta = await AsyncMultimodalProcessor.process_file(path, user_id)
            return JSONResponse({"processed": meta})
        
        assistant = await assistant_pool.acquire(user_id)
        try:
            await asyncio.to_thread(assistant.initialize)
            result = await assistant.process_query_async(
                f"User uploaded {file.filename}",
                file_path=path
            )
            
            return JSONResponse(result)
        finally:
            await assistant_pool.release(assistant)
    
    except Exception as e:
        print(f"[API] Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Application startup"""
    print("[API] Mental Health API starting up...")
    print(f"[API] Upload directory: {UPLOAD_DIR}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    print("[API] Shutting down...")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 10000))
    
    if sys.platform == 'win32':
        print("⚠️  Running on Windows - using uvicorn (single process)")
        print("💡 For production on Windows, run multiple instances behind nginx/IIS")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            reload=True
        )
    else:
        import multiprocessing
        workers = multiprocessing.cpu_count() * 2 + 1
        print(f"🚀 Running with {workers} workers")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            workers=workers,
            log_level="info"
        )
