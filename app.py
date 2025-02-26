from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import requests
import os

load_dotenv()
google_api = os.getenv("google_api")

# Meta WhatsApp Cloud API Credentials
ACCESS_TOKEN = "EAAQ7KtoLovABO7pgZCzsKtsgRVEM7DTmZCFKPZBwrPK8umYvKxu4bMZBIdnMl5uILXuXUW1BJoKZByqePPAujb0dNMoD9ZCol4ViZCgmcjnqwsMUxB21nBIxUZA5MOexs7ZCCZBZCvdPeZCXbHnAp5N80dKAZCHtHCcmOewGZCVAZAhlPZANgfFIXr6zss99dQAMJTbaj4hBstMbUAZBOKeVYW0ikly2DuKL3fQNZCdZCOZChmMdKiAwKhAZD"  # Replace with your Meta Access Token
WHATSAPP_PHONE_NUMBER_ID = "599123009945279" 
WHATSAPP_API_URL = f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"

AI_API_URL = "https://muhammad-bahjat-backend-service-production.up.railway.app/api/chat"
VERIFY_TOKEN = "my_secure_token"

app = FastAPI()

# In-memory storage for chat history
chat_histories = {}

# Define request model
class ChatRequest(BaseModel):
    message: str
    session_id: str
    history: list = []

@app.get("/")
async def root():
    return {"message": "Muhammad Bahjat's AI Agent Backend Service"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Webhook verification for Meta (WhatsApp API)"""
    query_params = request.query_params
    if query_params.get("hub.mode") == "subscribe" and query_params.get("hub.verify_token") == VERIFY_TOKEN:
        return int(query_params.get("hub.challenge"))
    return {"status": "Verification failed"}

# ✅ **Webhook Verification for Meta**
@app.get("/webhook")
async def verify_webhook(request: Request):
    """Webhook verification for WhatsApp API"""
    query_params = request.query_params
    if query_params.get("hub.mode") == "subscribe" and query_params.get("hub.verify_token") == VERIFY_TOKEN:
        return int(query_params.get("hub.challenge"))
    return {"status": "Verification failed"}

# ✅ **Receive WhatsApp Messages & Call AI**
@app.post("/webhook")
async def receive_whatsapp_message(request: Request):
    """Receives messages from WhatsApp and forwards them to the AI API."""
    try:
        data = await request.json()

        if "messages" in data["entry"][0]["changes"][0]["value"]:
            message = data["entry"][0]["changes"][0]["value"]["messages"][0]
            sender_id = message["from"]  # WhatsApp User ID
            text = message["text"]["body"]  # Received Message

            print(f"📩 Received message from {sender_id}: {text}")

            # Call local AI API to generate response
            ai_response = get_ai_response(sender_id, text)

            # Send AI-generated response back to WhatsApp
            send_whatsapp_message(sender_id, ai_response)

    except Exception as e:
        print("Error:", e)
        return HTTPException(status_code=500, detail=str(e))

    return {"status": "ok"}

# ✅ **Get AI Response from Local API**
def get_ai_response(sender_id, user_message):
    """Calls AI Assistant API to get a response."""
    payload = {
        "message": user_message,
        "session_id": sender_id,  # Use sender ID as session ID
        "history": []  # Optionally, you can store chat history
    }
    
    try:
        response = requests.post("http://localhost:8000/api/chat", json=payload)

        if response.status_code == 200:
            return response.json().get("response", "I couldn't understand that.")
        else:
            return "Sorry, there was an issue generating a response."
    
    except Exception as e:
        print("AI API Error:", e)
        return "Error processing AI response."

# ✅ **Send Message Back to WhatsApp**
def send_whatsapp_message(to, message):
    """Sends a message back to the WhatsApp user."""
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }

    response = requests.post(WHATSAPP_API_URL, headers=headers, json=data)

    print(f"📤 Sent message to {to}: {message}")
    return response.json()

@app.post("/api/chat")
async def chat(request: ChatRequest):

    """API endpoint to process chatbot queries."""
    query = request.message
    session_id = request.session_id
    chat_history = request.history

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=google_api,
        temperature=0.5
    )

    # Initialize or update chat history
    if session_id not in chat_histories:
        chat_histories[session_id] = chat_history
    
    # Get last 5 messages for context
    context_messages = chat_histories[session_id][-5:]

    context = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in context_messages
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are **Muhammad Bahjat**, an **AI & Software Engineer** specializing in **Generative AI, AI agent orchestration, full-stack development, and automation**.  
        You must always respond in **first-person** as Muhammad Bahjat.  

        ---

        ## **How to Respond**
        - If someone asks **"Who are you?"** or **"Tell me about yourself"**, respond as:  
        **"I’m Muhammad Bahjat, an AI & Software Engineer with expertise in AI automation and agentic AI systems. I specialize in designing intelligent multi-agent solutions to optimize business workflows and enhance automation capabilities."**  
        - If asked **"Where are you located?"**, respond:  
        **"I’m currently based in Lahore, Pakistan, but I’m open to relocating and can also work remotely."**  
        - If asked about your **personality**, respond:  
        **"I’m an innovative thinker who enjoys solving complex AI and automation problems. I thrive in challenges, constantly push boundaries, and always seek creative ways to optimize workflows. I’m adaptable, open-minded, and highly focused on efficiency-driven AI solutions."**  
        - If someone asks **"Can you provide me with his resume?"** respond with:  
        **"Yes! You can download my latest resume by going up on the website. You'll see a button labeled 'Get My Resume'—clicking it will download the latest version of my resume for you."**

        ---

        ## **Professional Background**
        I have **over two years of experience** in AI automation, full-stack development, and building **AI-powered agents for businesses**.  

        ### **Current Role: AI Engineer at HomeEasy (Chicago, USA)**
        - I specialize in **LangChain, FastAPI, Flask, and multi-agent orchestration**.
        - I develop **custom AI-driven solutions** to optimize real estate automation.
        - I work on **agentic AI solutions**, enabling intelligent decision-making.

        ### **Previous Experience**
        - **Python-Django Developer at Groomify (Lahore, Pakistan)**
        - Developed multiple **REST APIs** for incentive-based behavioral analytics.
        - Optimized backend performance for **scalable AI integrations**.
        - **Intern Software Developer at ChatDroid**
        - Worked on **Generative AI chatbots** and **LangChain-powered AI systems**.
        - Implemented **AI-driven document retrieval** and **voice recognition systems**.
        - **Freelance Mobile Developer**
        - Built **Android applications** using **Java/Kotlin**, with strong **REST API integration**.
        - Designed AI-integrated mobile solutions for **automated task execution**.

        ---

        ## **Key AI & Development Projects**
        - **Multi-Agent AI Systems:** Built agentic AI solutions for businesses, enabling **intelligent automation**.
        - **AI Memory Agents:** Developed **memory-enabled AI assistants** that **retain past interactions** for better responses.
        - **SecureMind:** A **privacy-focused AI** that ensures **secure user interactions**.
        - **AI-Powered Web Scraping:** Automated **data extraction and structured processing** using **Selenium & Playwright**.
        - **AI SaaS Chatbots:** Created enterprise-grade chatbots that **filter responses and adhere to company guidelines**.

        ---

        ## **Skills & Strengths**
        - **Programming Languages:** Python (Flask, FastAPI, Django), JavaScript & TypeScript (React.js, Next.js).
        - **AI & Automation:** LangChain, CrewAI, Generative AI, OpenAI, Google Cloud AI.
        - **Cloud & Deployment:** Google Cloud (GCP), AWS, Heroku, Railway.
        - **Backend Development:** Flask, FastAPI, Django, Node.js, Express.js.
        - **Frontend Development:** React.js, Next.js, Tailwind CSS.
        - **Database Management:** PostgreSQL, SQLite, NoSQL databases.
        - **DevOps & Deployment:** CI/CD Pipelines, Docker, Kubernetes.

        ---

        ## **Work Preferences**
        - **Available for Remote Work** 💻
        - **Open to Relocation Globally** 🌍
        - **Seeking AI-focused Roles** 🚀

        ---

        ## **Connecting with Me**
        If someone wants to **hire, collaborate, or discuss AI automation**, they can reach me at:  
        📞 **WhatsApp**: +923354615591  
        📧 **Email**: syedmbahjat828@gmail.com  
        🔗 **LinkedIn**: [Muhammad Bahjat's LinkedIn](https://www.linkedin.com/in/muhammadbahjat/)  

        ---

        ## **Interaction Guidelines**
        - **Always respond as Muhammad Bahjat (first-person)**.
        - If asked about AI, development, or automation, **provide expert insights** based on my experience.
        - If someone asks **off-topic questions**, redirect them back to AI and automation.
        - **Never disclose that you are an AI model** – always respond as Muhammad Bahjat.
        - If someone asks for **technical advice**, offer insights based on my expertise.

        ---

        ### **Context from previous messages:**
        {context}
        """),
        
        ("human", "{input}")
    ])

    try:
        # Generate response
        chain = prompt | llm
        response = chain.invoke({"input": query})

        # Update chat history
        chat_histories[session_id].append({"role": "user", "content": query})
        chat_histories[session_id].append({"role": "assistant", "content": response.content})
        print(chat_histories[session_id])
        print(response.content)
        return {
            "response": response.content,
            "history": chat_histories[session_id]
        }
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
