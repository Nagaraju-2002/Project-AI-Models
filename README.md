# **Advanced AI-Driven Document Assistant**  

A robust Flask-based AI application that integrates cutting-edge NLP technologies to analyze documents, perform question answering, sentiment analysis, text summarization, and multilingual translation.  

## **Features**  
- **Document Analysis** – Extract insights, summaries, and key information from documents.  
- **AI-Powered Q&A** – Uses transformers to provide intelligent question-answering capabilities.  
- **Sentiment Analysis** – Detects sentiment polarity (positive, negative, neutral) from text.  
- **Text Summarization** – Generates concise summaries while preserving key information.  
- **Multilingual Translation** – Supports automatic translation across multiple languages.  
- **Scalable Pipelines** – Optimized with FAISS and RecursiveCharacterTextSplitter for performance.  

## **Technologies Used**  
- **Programming Language:** Python  
- **Frameworks & Libraries:** Flask, Hugging Face, OpenAI GPT API, Google Generative AI API, LangChain, FAISS  
- **Data Processing:** PyPDF2, SpacyTextBlob, NumPy, Pandas  
- **API Development:** REST APIs for seamless integration  


## **Setup and Installation**  

### **Clone the Repository**  
```
git clone https://github.com/Nagaraju-2002/Project-AI-Models.git
cd Project-AI-Models
```

### **Create a Virtual Environment**  
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **Install Dependencies**  
```
pip install -r requirements.txt
```

### **Set Up API Keys**  
Create a `.env` file in the root directory and add your API keys:  
```
OPENAI_API_KEY=api_key
GOOGLE_API_KEY=api_key
```

### **Run the Application**  
```
python app.py
```

## **Future Enhancements**  
- Frontend integration using React  
- Docker support for containerized deployment  
- Optimization of AI models for faster response times  

  
