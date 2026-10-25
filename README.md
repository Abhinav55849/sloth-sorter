# 🦥 Sloth Sorter

**AI-powered local file organizer** — built for the [Hacktoberfest Open-Source AI Challenge](https://hacktoberfest.com) Week 3.

Sloth Sorter scans a messy folder, uses a **local AI model** (Gemma 3 via Ollama) to suggest categories and cleaner filenames, then lets you review and apply the changes — all without cloud APIs or API keys.

---

## ✨ Features

- **Local AI** — uses Ollama + Gemma 3 (4B) running entirely on your machine
- **Smart categorization** — classifies files into Documents, Images, Code, Study, Installers, Archives, or Other
- **Cleaner filenames** — AI suggests human-readable names while preserving original extensions
- **Dry-run preview** — see every proposed change before anything moves
- **Selective apply** — pick exactly which files to reorganize
- **Safe undo** — reverse the last run with one click
- **Extension fallback** — works even without Ollama via built-in extension rules
- **Zero cloud dependency** — no OpenAI, no Google, no API keys

## 🛡️ Safety

- Files are **never** deleted or overwritten
- Filename collisions are resolved automatically (`_1`, `_2`, …)
- All paths are validated to stay within your selected folder
- AI-suggested filenames are sanitized (no path traversal, no reserved characters)
- Hidden files and subdirectories are skipped
- Nothing moves until you explicitly click **Apply**

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Install |
|---|---|
| Python 3.10+ | [python.org](https://www.python.org/downloads/) |
| Ollama | [ollama.com/download](https://ollama.com/download) |
| Gemma 3 4B | `ollama pull gemma3:4b` |

### Setup

```bash
# Clone the repo
git clone https://github.com/<your-user>/sloth-sorter.git
cd sloth-sorter

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull the AI model (if you haven't already)
ollama pull gemma3:4b
```

### Run

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

> **Tip:** If Ollama isn't running, the app still works using extension-based fallback rules — just without AI-suggested filenames.

---

## 🧪 Testing

Tests use **pytest** with temporary directories — no real files are touched and no running Ollama instance is needed.

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
sloth-sorter/
├── app.py              # Streamlit UI (main entry point)
├── scanner.py          # File discovery and metadata extraction
├── classifier.py       # Ollama/Gemma 3 integration + fallback rules
├── mover.py            # Safe file moves, collision handling, undo
├── requirements.txt
├── .gitignore
├── LICENSE             # MIT
├── README.md
└── tests/
    ├── __init__.py
    ├── test_scanner.py
    ├── test_classifier.py
    └── test_mover.py
```

| Module | Responsibility |
|---|---|
| `scanner.py` | Non-recursive directory scan, metadata collection, text snippet extraction |
| `classifier.py` | AI prompt building, Ollama HTTP calls, JSON parsing, fallback rules, filename sanitization |
| `mover.py` | Safe `shutil.move`, collision avoidance, path containment, JSON logging, undo |
| `app.py` | Streamlit UI with 3-stage flow: Select → Preview → Apply/Undo |

---

## 🔧 Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web UI
- **Ollama** — local model inference server
- **Gemma 3 (4B)** — open-weight AI model by Google
- **requests** — HTTP client for the Ollama API
- **pytest** — testing framework

---

## 📝 License

[MIT](LICENSE) — use it, fork it, improve it.
