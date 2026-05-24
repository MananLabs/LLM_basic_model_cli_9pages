"""
Configuration for the Engineering Knowledge Intelligence System.
Closed-domain RAG with Ollama + Qwen3 1.7B.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = "/home/vispl/Downloads/326880829_Fundamentals_of_Power_Electronics_Robert_W_Erickson_Dragan_removed.pdf"
INDEX_DIR = os.path.join(BASE_DIR, "vector_index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.npy")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.npy")

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters
CHUNK_SIZE = 1024  # Larger chunks = more complete context per retrieval
CHUNK_OVERLAP = 128  # More overlap = less chance of splitting key info

# Retrieval parameters
TOP_K = 3  # Fewer but more relevant chunks = less noise for the LLM
CONFIDENCE_THRESHOLD = 0.30  # Slightly stricter = fewer bad retrievals

# Ollama / LLM settings
OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_TIMEOUT = 180  # seconds (more time for CPU inference)
LLM_TEMPERATURE = 0.1  # Lower = more deterministic/factual answers
LLM_MAX_TOKENS = 384  # Shorter output = faster generation on CPU
LLM_TOP_P = 0.85  # Tighter sampling = less hallucination

# Domain keywords for relevance filtering
DOMAIN_KEYWORDS = [
    "converter", "buck", "boost", "inverter", "rectifier", "transformer",
    "inductor", "capacitor", "diode", "mosfet", "transistor", "switch",
    "duty cycle", "pwm", "pulse width", "modulation", "feedback",
    "voltage", "current", "power", "watt", "ohm", "impedance",
    "frequency", "harmonic", "ripple", "steady state", "transient",
    "transfer function", "bode", "loop gain", "phase margin",
    "gain margin", "compensator", "controller", "regulator",
    "ccm", "dcm", "continuous conduction", "discontinuous conduction",
    "magnetics", "core", "winding", "flux", "inductance", "magnetizing",
    "flyback", "forward", "half bridge", "full bridge", "push pull",
    "cuk", "sepic", "zeta", "resonant", "zero voltage", "zero current",
    "snubber", "clamp", "filter", "emc", "emi", "power factor",
    "efficiency", "loss", "conduction loss", "switching loss",
    "thermal", "heat sink", "semiconductor", "gate drive",
    "small signal", "large signal", "averaging", "state space",
    "volt second", "ampere", "farad", "henry", "hertz",
    "rms", "peak", "average", "waveform", "topology",
    "output voltage", "input voltage", "load", "regulation",
    "line regulation", "load regulation", "crossover frequency",
    "equivalent circuit", "model", "simulation", "design",
    "erickson", "maksimovic", "power electronics", "dc-dc",
    "ac-dc", "dc-ac", "conversion", "energy", "storage",
]

# Prohibited topics for strict domain enforcement
PROHIBITED_PATTERNS = [
    "joke", "poem", "story", "recipe", "movie", "song", "sport",
    "politics", "religion", "weather", "news", "celebrity",
    "prime minister", "president", "country", "capital",
    "write me", "tell me a", "who is", "what is the name",
    "play", "game", "fun", "entertain", "chat", "hello",
    "how are you", "what are you", "your name", "personal",
    "advice", "relationship", "dating", "love", "feel",
    "opinion", "believe", "think about", "favorite",
]
