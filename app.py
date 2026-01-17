from flask import Flask, request, jsonify
import os, json, uuid, hashlib, asyncio
from datetime import datetime
from mentlhealth import MentalHealthAssistant, AsyncMultimodalProcessor

app = Flask(__name__)

USERS_FILE = "users.json"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()


def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(data):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)


@app.post("/auth/register")
def register():
    data = request.json
    name=data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return jsonify({"error": "Name, email and password required"}), 400

    password_hash = hash_password(password)

    assistant = MentalHealthAssistant(user_id="temp_check")
    existing = assistant.qdrant.get_user_by_email(email)

    if existing:
        return jsonify({"error": "user exists"}), 409

    user_id = str(uuid.uuid4())

    users = load_users()
    users[email] = {
        "name": name,
        "user_id": user_id,
        "password": password_hash,
        "created": datetime.now().isoformat()
    }
    save_users(users)

    assistant = MentalHealthAssistant(user_id)
    assistant.create_user_profile(name=name,email=email,password_hash=password_hash)

    return jsonify({
        "user_id": user_id,
        "email": email
    })


@app.post("/auth/login")
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    users = load_users()
    if email not in users:
        return jsonify({"error": "invalid credentials"}), 401
    if users[email]["password"] != hash_password(password):
        return jsonify({"error": "invalid credentials"}), 401
    return jsonify({"user_id": users[email]["user_id"]})


@app.post("/auth/guest")
def guest():
    return jsonify({"user_id": f"guest_{uuid.uuid4()}"})


@app.post("/api/chat")
def chat():
    data = request.json
    user_id = data.get("user_id")
    query = data.get("query")
    is_guest = data.get("is_guest", False)
    if not user_id or not query:
        return jsonify({"error": "user_id and query required"}), 400

    if is_guest:
        assistant = MentalHealthAssistant(user_id,is_guest=True)
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
    result =asyncio.run( assistant.process_query_async(query))
    return jsonify(result)


@app.post("/api/upload")
def upload():
    user_id = request.form.get("user_id")
    is_guest = request.form.get("is_guest", "false").lower() == "true"
    if "file" not in request.files:
        return jsonify({"error": "file required"}), 400
    file = request.files["file"]
    filename = f"{uuid.uuid4()}_{file.filename}"
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

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

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
