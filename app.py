from flask import Flask, request, jsonify
import os, json, uuid, hashlib, asyncio
from datetime import datetime

app = Flask(__name__)

_assistant_module = None

def get_assistant_module():
    global _assistant_module
    if _assistant_module is None:
        from mentlhealth import MentalHealthAssistant, AsyncMultimodalProcessor
        _assistant_module = {
            'MentalHealthAssistant': MentalHealthAssistant,
            'AsyncMultimodalProcessor': AsyncMultimodalProcessor
        }
    return _assistant_module

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Mental Health API is running",
        "version": "1.0.0"
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.post("/auth/register")
def register():
    try:
        data = request.json
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if not name or not email or not password:
            return jsonify({"error": "Name, email and password required"}), 400

        password_hash = hash_password(password)

        modules = get_assistant_module()
        MentalHealthAssistant = modules['MentalHealthAssistant']

        temp_assistant = MentalHealthAssistant(user_id="temp_check")
        existing = temp_assistant.qdrant.get_user_by_email(email)

        if existing:
            return jsonify({"error": "user exists"}), 409

        user_id = str(uuid.uuid4())

        assistant = MentalHealthAssistant(user_id)
        assistant.create_user_profile(name=name, email=email, password_hash=password_hash)

        return jsonify({
            "user_id": user_id,
            "email": email
        })

    except Exception as e:
        app.logger.error(f"Registration error: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500


@app.post("/auth/login")
def login():
    try:
        data = request.json
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        modules = get_assistant_module()
        MentalHealthAssistant = modules['MentalHealthAssistant']

        temp_assistant = MentalHealthAssistant(user_id="temp_check")
        user_profile = temp_assistant.qdrant.get_user_by_email(email)

        if not user_profile:
            return jsonify({"error": "invalid credentials"}), 401

        if user_profile.password_hash != hash_password(password):
            return jsonify({"error": "invalid credentials"}), 401

        return jsonify({"user_id": user_profile.user_id})

    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Login failed"}), 500


@app.post("/auth/guest")
def guest():
    try:
        return jsonify({"user_id": f"guest_{uuid.uuid4()}"})
    except Exception as e:
        app.logger.error(f"Guest error: {str(e)}")
        return jsonify({"error": "Guest creation failed"}), 500


@app.post("/api/chat")
def chat():
    try:
        data = request.json
        user_id = data.get("user_id")
        query = data.get("query")
        is_guest = data.get("is_guest", False)

        if not user_id or not query:
            return jsonify({"error": "user_id and query required"}), 400

        modules = get_assistant_module()
        MentalHealthAssistant = modules['MentalHealthAssistant']

        if is_guest:
            assistant = MentalHealthAssistant(user_id, is_guest=True)
            assistant.initialize()
            resp = assistant.llm.generate_response(
                query, [], None, [], [], {}
            )[0]
            return jsonify({
                "user_id": user_id,
                "response": resp,
                "timestamp": datetime.now().isoformat()
            })

        assistant = MentalHealthAssistant(user_id)
        assistant.initialize()
        result = asyncio.run(assistant.process_query_async(query))
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500


@app.post("/api/upload")
def upload():
    try:
        user_id = request.form.get("user_id")
        is_guest = request.form.get("is_guest", "false").lower() == "true"

        if "file" not in request.files:
            return jsonify({"error": "file required"}), 400

        file = request.files["file"]
        filename = f"{uuid.uuid4()}_{file.filename}"
        path = os.path.join(UPLOAD_DIR, filename)
        file.save(path)

        modules = get_assistant_module()
        MentalHealthAssistant = modules['MentalHealthAssistant']
        AsyncMultimodalProcessor = modules['AsyncMultimodalProcessor']

        if is_guest:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _, meta = loop.run_until_complete(
                AsyncMultimodalProcessor.process_file(path, user_id)
            )
            return jsonify({"processed": meta})

        assistant = MentalHealthAssistant(user_id)
        assistant.initialize()
        result = assistant.process_query(
            f"User uploaded {file.filename}", file_path=path
        )
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

@app.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)