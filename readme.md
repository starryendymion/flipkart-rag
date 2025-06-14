# 🛍️ Flipkart Product Description Generator & RAG System

A comprehensive AI-powered system for generating enhanced product descriptions and implementing Retrieval-Augmented Generation (RAG) for intelligent product search and recommendations.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-supported-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### 🤖 **AI-Powered Description Generation**
- **LLM Integration**: Uses Qwen2.5-1.5B-Instruct for high-quality text generation
- **Context-Aware**: Leverages product specifications, pricing, and brand information
- **Batch Processing**: Efficient processing of large product catalogs
- **Resume Capability**: Continues from where it left off if interrupted

### 🔍 **Vector Database & Similarity Search**
- **Semantic Search**: Uses sentence transformers for meaningful product discovery
- **FAISS Integration**: Lightning-fast similarity search with GPU acceleration
- **Scalable**: Handles millions of products efficiently
- **Persistent Storage**: Save and load vector databases

### 🧠 **RAG (Retrieval-Augmented Generation)**
- **Contextual Answers**: Provides intelligent responses based on product data
- **Interactive Chat**: Real-time question-answering interface
- **Multi-Product Context**: Considers multiple relevant products for comprehensive answers
- **Customizable Retrieval**: Adjustable number of products to consider

### 🐳 **Docker & GPU Support**
- **CUDA Optimized**: Full GPU acceleration for faster processing
- **Containerized**: Easy deployment with Docker and docker-compose
- **Multi-Architecture**: Supports both CPU and GPU environments
- **Volume Persistence**: Data persistence across container restarts

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA Docker runtime (for GPU support)
- NVIDIA drivers (for GPU support)

### 1. Clone the Repository
```bash
git clone https://github.com/starryendymion/flipkart-product-rag.git
cd flipkart-product-rag
```

### 2. Prepare Your Data
Place your Flipkart CSV file in the `data/` directory:
```bash
mkdir -p data
cp your_flipkart_products.csv data/flipkart_products.csv
```

**Required CSV columns:**
- `product_name`
- `pid` (product ID)
- `product_url`
- Other optional columns: `brand`, `description`, `product_specifications`, etc.

### 3. Build and Run with Docker
```bash
# Build the Docker image
docker-compose build

# Run the container
docker-compose up -d

# Access the interactive interface
docker-compose exec flipkart-rag python main.py
```

### 4. Alternative: Direct Docker Run
```bash
# Build
docker build -t flipkart-rag .

# Run with GPU support
docker run --gpus all -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/vector_db:/app/vector_db \
  flipkart-rag
```

## 📋 Usage

### Interactive Mode (Recommended)
```bash
python main.py --mode interactive
```

### Command Line Options
```bash
# Generate descriptions only
python main.py --mode generate --input-csv data/products.csv --max-items 100

# Build vector database
python main.py --mode build-db --output-csv output/enhanced_products.csv

# Start RAG system
python main.py --mode rag --vector-db-path vector_db

# Full pipeline
python main.py --mode interactive
```

### Using Individual Components

#### 1. Generate Enhanced Descriptions
```python
from flipkart_generator import FlipkartDescriptionGenerator

generator = FlipkartDescriptionGenerator(
    input_csv_path="data/products.csv",
    output_csv_path="output/enhanced_products.csv"
)
generator.process_products(max_items=100)
```

#### 2. Build Vector Database
```python
from vector_rag_system import build_vector_database

vector_db = build_vector_database(
    csv_file_path="output/enhanced_products_final.csv",
    vector_db_path="vector_db"
)
```

#### 3. Query with RAG
```python
from vector_rag_system import start_rag_system

rag_system = start_rag_system("vector_db")
# This starts an interactive chat interface
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Input CSV      │───▶│ Description      │───▶│ Enhanced CSV    │
│  (Raw Products) │    │ Generator        │    │ (3 columns)     │
└─────────────────┘    │ (Qwen2.5 LLM)    │    └─────────────────┘
                       └──────────────────┘             │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ RAG System      │◀───│ Vector Database  │◀───│ Embedding Model │
│ (Q&A Interface) │    │ (FAISS Index)    │    │ (SentenceT5)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Example Output

### Generated Descriptions
```csv
product_name,product_url,generated_description
"Samsung Galaxy S21","https://flipkart.com/...",Samsung Galaxy S21 delivers flagship performance with its powerful Snapdragon processor and stunning 6.2-inch Dynamic AMOLED display. Features advanced triple-camera system with 64MP telephoto lens for professional photography. Perfect for users seeking premium smartphone experience with 5G connectivity.
```

### RAG Query Examples
```
User: "Find me a good smartphone under ₹50000"
Assistant: Based on the available products, I found several excellent smartphones under ₹50000. The Samsung Galaxy S21 offers flagship performance with its powerful Snapdragon processor, stunning 6.2-inch Dynamic AMOLED display, and advanced triple-camera system. The iPhone 12 provides premium iOS experience with A14 Bionic chip and excellent camera quality. Both phones offer 5G connectivity and are perfect for users seeking high-end features within your budget.

Retrieved Products:
1. Samsung Galaxy S21 (Score: 0.892)
2. iPhone 12 (Score: 0.867)
3. OnePlus 9 Pro (Score: 0.834)
```

## 🔧 Configuration

### Environment Variables
```bash
# GPU Settings
CUDA_VISIBLE_DEVICES=0,1
NVIDIA_VISIBLE_DEVICES=all

# Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct

# Processing Settings
BATCH_SIZE=32
MAX_TOKENS=300
```

### Resource Requirements

| Component | CPU | RAM | GPU VRAM | Storage |
|-----------|-----|-----|----------|---------|
| Minimum | 4 cores | 8GB | 4GB | 10GB |
| Recommended | 8 cores | 16GB | 8GB | 50GB |
| Optimal | 16 cores | 32GB | 16GB+ | 100GB |

## 📁 Project Structure

```
flipkart-product-rag/
├── 📄 Dockerfile                 # Docker configuration
├── 📄 docker-compose.yml         # Docker Compose setup
├── 📄 requirements.txt           # Python dependencies
├── 📄 main.py                    # Main application entry
├── 📄 flipkart_generator.py      # Description generator
├── 📄 vector_rag_system.py       # Vector DB & RAG system
├── 📁 data/                      # Input CSV files
├── 📁 output/                    # Generated descriptions
├── 📁 vector_db/                 # Vector database files
├── 📁 models/                    # Downloaded model cache
└── 📄 README.md                  # This file
```

## 🛠️ Development

### Local Setup (Without Docker)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Adding New Features
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🔍 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Verify CUDA in container
docker-compose exec flipkart-rag python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues
```bash
# Reduce batch size
python main.py --batch-size 16

# Use CPU-only mode
python main.py --device cpu
```

#### Model Download Issues
```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')"
```

### Performance Optimization

1. **GPU Memory**: Use `--batch-size` to control memory usage
2. **CPU Cores**: Set `OMP_NUM_THREADS` for optimal CPU usage
3. **Storage**: Use SSD for vector database for faster I/O
4. **Network**: Cache models locally to avoid repeated downloads

## 📈 Performance Metrics

| Operation | Time (GPU) | Time (CPU) | Accuracy |
|-----------|------------|------------|----------|
| Description Generation | ~2s/product | ~8s/product | High |
| Vector DB Creation | ~1min/10k products | ~5min/10k products | N/A |
| RAG Query | ~0.5s | ~2s | High |

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines:

1. **Issues**: Report bugs or suggest features
2. **Pull Requests**: Submit improvements or fixes
3. **Documentation**: Help improve docs and examples
4. **Testing**: Add tests for new features

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

## 🙏 Acknowledgments

- **Qwen Team** for the excellent language model
- **Sentence Transformers** for semantic embeddings
- **FAISS Team** for efficient similarity search
- **Hugging Face** for the transformers library
- **Docker** for containerization support

</div>