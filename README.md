# Engineering Knowledge Intelligence System

A **closed-domain** conversational AI interface for the engineering textbook:

> **Fundamentals of Power Electronics**  
> Robert W. Erickson & Dragan Maksimovic  
> 2nd Edition (2001)

## Architecture

```
PDF Dataset
    |
    v
[PDF Extractor] --> [Text Chunker] --> [Embedding Model] --> [FAISS Index]
                                                                    |
User Query --> [Domain Guard] --> [Retriever] --> [Response Generator] --> CLI Output
```

### Components

| Module | Purpose |
|--------|---------|
| `config.py` | All configuration parameters |
| `pdf_extractor.py` | Extracts text from PDF, chunks it |
| `indexer.py` | Builds FAISS vector index from chunks |
| `retriever.py` | Semantic search over the index |
| `domain_guard.py` | Rejects off-topic queries |
| `response_generator.py` | Formats structured engineering responses |
| `main.py` | CLI interface orchestrating all modules |

## Setup

### Prerequisites
- Python 3.9+
- The PDF file in this directory

### Quick Start (Windows)

```
setup.bat
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build the vector index (one-time, takes a few minutes)
python indexer.py

# Run the system
python main.py
```

## Usage

```
>> Explain the buck converter steady-state operation
>> What is inductor volt-second balance?
>> Derive the conversion ratio of a boost converter
>> How does DCM affect the buck converter?
```

Off-topic queries are automatically rejected:
```
>> Tell me a joke
ERROR: This query is outside the scope of the engineering dataset.
Please ask questions strictly related to the provided engineering book.
```

## Design Principles

1. **Closed-domain only** - answers come exclusively from the indexed PDF
2. **No hallucination** - if the dataset doesn't contain it, the system refuses
3. **Domain guard** - pre-filters queries before retrieval
4. **Confidence threshold** - post-retrieval validation rejects low-confidence matches
5. **Traceable** - every response cites source pages from the book
