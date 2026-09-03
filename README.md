# 🛒 Autonomous Multi-Agent E-Commerce Platform

An autonomous, multi-agent conversational commerce platform connecting two merchant storefronts (**ShopNest** & **CartWave**) with an intelligent customer-advocate **AI Buyer Agent (Agent 1)** and a merchant revenue-optimizing **Sales Improvement Agent (Agent 2)**, guarded by a deterministic **Multi-Tier Trust Layer** and integrated with **Razorpay Test Mode** payments.

---

## 🌐 Live Demo & Quick Links

- **Live Demo (Deployed)**: https://razorpay-agent-tqgi.onrender.com
- **Complete System Architecture & Technical Guide**: Refer to `DOCUMENTATION.md` or open `guide.html` in your browser.
- **Supported MVP Products (50 Items)**: `list.txt` (Check inside this file to know the products that are added)

---

## 💡 Why This Project?

Instead of manually visiting multiple e-commerce websites to compare prices, check sizes, verify stock, and look for matching accessories, users can converse naturally with an AI in plain English. 

### Key Architectural Highlights:
1. **Multi-Agent Collaboration (MAS)**: Agent 1 acts as the buyer's advocate to find the best deal, while Agent 2 acts as the merchant sales optimizer recommending high-converting, compatible add-ons.
2. **Universal Agent Protocol (UAP/1.0)**: Standardized, cryptographically signed inter-agent message envelopes and discovery manifest (`/.well-known/agent.json`).
3. **Agent Payment Protocol (AP2/1.0)**: Cryptographically bounded spending mandates, deterministic SHA-256 cart fingerprints, and verifiable settlement receipts.
4. **Deterministic Fairness**: 100% mathematical scoring logic eliminates AI bias and ensures the best offer (cheapest, highest-rated, or balanced) unconditionally wins.
5. **Local Semantic Search**: In-memory vector embeddings (`all-MiniLM-L6-v2`) search catalogs by meaning in < 0.05ms without expensive external vector databases.
6. **Zero-Trust Security Layer**: 8 strict security guardrails prevent price tampering, out-of-stock purchases, and unauthorized cart modifications.
7. **Full Explainability & Audit Trail**: Every decision, score breakdown, and protocol envelope is exported into interactive reports and downloadable PDFs.

---

## 🚀 Step-by-Step Installation & Execution Guide

Follow these steps to run the complete platform locally:

### Step 1: Clone the Repository
```bash
git clone https://github.com/Praveen123-hub/Razorpay-Buildathon.git
cd Razorpay-Buildathon
```

### Step 2: Create and Activate a Virtual Environment
- **On Windows (PowerShell / Command Prompt):**
  ```bash
  python -m venv ven
  ven\Scripts\activate
  ```
- **On macOS / Linux:**
  ```bash
  python3 -m venv ven
  source ven/bin/activate
  ```

### Step 3: Install Required Dependencies
Install all 9 core packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to create your local `.env` file:
```bash
cp .env.example .env
```
Open `.env` and configure your API keys (optional if using local fallbacks):
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
AGENT1_MODEL=qwen/qwen3-8b:free
AGENT2_MODEL=cohere/north-mini-code:free
RAZORPAY_KEY_ID=your_razorpay_key_id_here
RAZORPAY_KEY_SECRET=your_razorpay_key_secret_here
```

### Step 5: Run the Automated Protocol Tests
```bash
python -m unittest -v test_protocols.py
```

### Step 6: Start the Backend Server
Launch the FastAPI application with Uvicorn:
```bash
uvicorn backend.main:app --reload --port 8000
```
*(On startup, the SQLite database schema `backend/data/app.db` will be initialized automatically).*

### Step 7: Access the Application
Open your browser and navigate to:
- **Login / Register Portal**: [http://localhost:8000](http://localhost:8000) (or `http://localhost:8000/login.html`)
- **AI Buyer Agent Chat**: [http://localhost:8000/agent1](http://localhost:8000/agent1) (or `http://localhost:8000/agent1.html`)
- **Complete Architecture Guide**: [http://localhost:8000/guide.html](http://localhost:8000/guide.html)
- **ShopNest Merchant Storefront**: [http://localhost:8000/shopnest.html](http://localhost:8000/shopnest.html)
- **CartWave Merchant Storefront**: [http://localhost:8000/cartwave.html](http://localhost:8000/cartwave.html)

---

## 📁 Repository Structure

```
├── backend/
│   ├── data/
│   │   ├── shopnest/           # ShopNest products, inventory & orders JSON
│   │   └── cartwave/           # CartWave products, inventory & orders JSON
│   ├── agent1_service.py       # Agent 1 (FSM slot-filling & dialogue manager)
│   ├── agent2_service.py       # Agent 2 (Co-purchase & sales optimizer)
│   ├── ap2_service.py          # Agent Payment Protocol (AP2) mandate manager
│   ├── audit_builder.py        # Explainability audit report generator
│   ├── auth_service.py         # PBKDF2 user auth & session manager
│   ├── config.py               # Environment configuration
│   ├── data_access.py          # Merchant Data Access Layer (DAL)
│   ├── database.py             # SQLite database connection & table schema
│   ├── main.py                 # FastAPI application routes & endpoints
│   ├── payment_service.py      # Razorpay order creation & signature verification
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── scoring_engine.py       # Deterministic mathematical offer scoring
│   ├── semantic_search.py      # Local SentenceTransformer embedding search
│   ├── trust_layer.py          # 8-tier pre-checkout trust security guardrails
│   └── uap_service.py          # Universal Agent Protocol (UAP) communication
├── frontend/
│   ├── agent1.html             # AI Buyer Agent chat interface & audit modal
│   ├── guide.html              # Interactive system architecture & logic guide
│   ├── login.html              # Authentication UI (Login / Register)
│   ├── address.html            # User delivery address setup
│   ├── shopnest.html           # ShopNest live merchant storefront
│   ├── cartwave.html           # CartWave live merchant storefront
│   ├── Images/                 # Product catalogue images
│   └── Logo.jpeg               # Platform branding logo
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git exclusions (.env, ven/, *.db)
├── DOCUMENTATION.md            # Complete architecture & technical documentation
├── list.txt                    # 50 MVP supported products reference list
├── requirements.txt            # Python package dependencies
├── test_protocols.py           # Protocol automated test suite
└── README.md                   # Project overview & step-by-step setup guide
```

---

## 📖 In-Depth System Documentation

For complete, in-depth architectural details, mathematical formulas, state machine diagrams, security guardrail specifications, and database schema mappings, please refer to:

👉 **[`DOCUMENTATION.md`](DOCUMENTATION.md)** *(or view `guide.html` in your browser)*

