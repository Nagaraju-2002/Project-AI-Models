from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
import openai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import secrets
import pytesseract
from pdf2image import convert_from_path
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from speech_recognition import Recognizer, AudioFile
from gtts import gTTS
from speech_recognition import Recognizer, AudioFile
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
load_dotenv()
app.secret_key = secrets.token_hex(16)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
openai.api_key = OPENAI_API_KEY

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob") 
google_model = genai.GenerativeModel('gemini-1.5-flash')
google_chat = google_model.start_chat(history=[])

PREDEFINED_RESPONSES = {
    "hi": "Hello, I am here to help you with queries related to Awesome Review. How may I assist you today?",
    "hello": "Hello, I am here to help you with queries related to Awesome Review. How may I assist you today?",
    "hello, how are you doing?": "I am doing good. Thank you. How may I assist you today?",
    "who are you?": "I am an AI assistant here to help with queries related to Awesome Review.",
    "are you sure of the answer you gave above?": (
        "If you have any doubts, kindly cross-check with the support team by sending an email to support@awesomereview.com."
    ),
    "what is awesome review and what services does it offer?": (
        "Awesome Review offers a premier Internal Medicine board review course that assists in "
        "certification and recertification. We simplify intricate concepts using animations, making abstract "
        "and complex topics more understandable. Our approach maintains a 100% pass rate among numerous "
        "physicians and programs, ensuring readiness for board certification in just 63 hours."
    ),
    "courses location": (
        "In 2025, the courses are scheduled as follows: 3 in New Jersey, 1 in Orlando, 1 in Texas, "
        "1 in California, 1 in Scottsdale, and 1 in Chicago."
    ),
    "what is the course fee?": "The course fee is $1195.",
    "does the fee vary based on the event location?": "No, the course fee remains the same for all events.",
    "how many courses does awesome review conduct annually?": (
        "The number of courses can vary each year. In 2024, we are conducting 8 courses."
    ),
    "2025 board review courses": (
        "In 2025, there are a total of 9 events scheduled. There are no courses scheduled for August, October, "
        "November, and December."
    ),
    "summary of the pending options": (
        "**In-Person Option:**\n"
        "- Experience a comprehensive in-person setting with responsible safety measures in place.\n"
        "- Immerse yourself fully.\n"
        "- Network with fellow physicians for support.\n"
        "- Engage directly with the instructor.\n"
        "\n"
        "Join us in person at any of our 9 course dates and locations. "
        "For more details, kindly visit: https://awesomereview.com/abim-board-review-course-schedule\n"
        "\n"
        "**Live Stream Option:**\n"
        "- Conveniently attend the course in real time from your home or office.\n"
        "- Full immersion experience through live streaming.\n"
        "- Interact with the instructor via online chat.\n"
        "- Receive course materials directly.\n"
        "\n"
        "Participate in live streaming during any of our 9 course dates.\n"
        "\n"
        "No matter which option you choose, your selection includes:\n"
        "- 63 hours of board-focused content by a nationally known educator.\n"
        "- Over 600 pages of laser-focused materials with nearly 2000 board-relevant questions.\n"
        "- Interaction opportunities with the instructor.\n"
        "- Over 12 hours of supplementary video content post-course.\n"
        "- 63 AMA PRA Category 1 CME Credits.\n"
        "- 63 ABIM MOC Points."
    ),
    "is it permissible to record live stream sessions using recording software?": (
        "Recording any of the course sessions during the live stream is strictly forbidden and considered a criminal offense. "
        "Advanced software is employed to detect and prevent unauthorized recording, and legal action may be taken against violators."
    ),
    "which streaming platform does awesome review employ for live streaming?": (
        "Awesome Review uses a sophisticated, tailor-made streaming platform designed in-house to ensure security, "
        "reliability, and user-friendliness for participants."
    ),
    "i have yet to receive the course materials. what is the current status?": (
        "Please contact our support team at support@awesomereview.com with details about your issue. "
        "We’ll provide you with the most up-to-date information."
    ),
    "how can i rectify an error in my email address or change my email address?": (
        "To update your email address associated with Awesome Review registration, send an email to "
        "support@awesomereview.com with your name and the correct email address."
    ),
    "how can i proceed with canceling my course enrollment?": (
        "Course cancellations can be made within 72 hours of registration to qualify for a refund, minus a 6% credit card handling fee. "
        "After 72 hours, the registration fee is non-refundable. Transfers to another date and location are available."
    ),
    "what steps should i take if i am dissatisfied with the awesome review course and want to raise a complaint?": (
        "If dissatisfied, contact customer support at 201-905-0102 or email support@awesomereview.com. "
        "Our team is committed to addressing and resolving your concerns."
    ),
    "what is the success rate of awesome review?": (
        "Awesome Review has an impressively high success rate, maintaining a 100% pass rate for numerous physicians and programs."
    ),
    "i logged in to access my videos, but not all of them are available. what should i do?": (
        "You receive 12 hours of supplemental high-yield videos, not the entire course."
    ),
    "i haven't received my books and my course starts next week.": (
        "Books are typically shipped a week before the course starts. Contact us at 201-905-0102 or email support@awesomereview.com for assistance."
    ),
    "how do i claim cme credits and obtain my certificate for attending the awesome review board review course?": (
        "Claim your credits at CME University using the course code. Create your login on CME University's website if you don’t already have one."
    ),
    "how long do i have access to online materials after the course?": (
        "High-yield videos are available until December 1st of the year the course was taken."
    ),
    "are hotel accommodations included in the course fee?": (
        "No, attendees are responsible for their hotel accommodations."
    ),
}



generator = pipeline('text-generation', model='gpt2')
sentiment_model = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
zero_shot_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
qna_model = pipeline("question-answering", model="Intel/dynamic_tinybert")
english_to_french_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
english_to_arabic_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")


 
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")

def translate_text_optimized(text: str, language_code: str) -> str:
    """
    Efficiently translates text into the specified target language using mBART.

    Args:
        text (str): The text to translate.
        language_code (str): The target language code (e.g., 'fr_XX' for French).

    Returns:
        str: The translated text.

    Raises:
        ValueError: If the language code is not supported.
    """
    # List of supported language codes
    supported_languages = [
        "ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT",
        "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK",
        "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE",
        "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN",
        "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"
    ]
    
    
    if language_code not in supported_languages:
        raise ValueError(f"Unsupported language code '{language_code}'. Supported codes are: {supported_languages}")
    
    
    tokenizer.src_lang = "en_XX"
    
   
    inputs = tokenizer(text, return_tensors="pt")
    
    
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[language_code]
    )
    
   
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
text_to_translate = "How are you"
target_language_code = "ta_IN"  
translation = translate_text_optimized(text_to_translate, target_language_code)
print(f"Translated Text: {translation}")


def get_pdf_text(pdf_files, max_pages=50):
    """Extract text from PDF files"""
    text = ""

    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        page_count = min(len(pdf_reader.pages), max_pages)
        for page_num in range(page_count):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Split text into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=3000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Save text chunks to FAISS vector store"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def speech_to_text(audio_file_path):
    """Convert speech in an audio file to text"""
    recognizer = Recognizer()
    with AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

def text_to_speech(text, output_path="response.mp3", lang="en", tld="com"):
    """Convert text to speech with customizable accents"""
    tts = gTTS(text, lang=lang, tld=tld)
    tts.save(output_path)
    return output_path
def text_to_speech_with_custom_voice(text, custom_voice_path, output_path="custom_response.mp3"):
    """
    Convert text to speech using a custom voice model or audio.
    This assumes a pre-trained model or a TTS framework supporting voice cloning.
    """
    try:
        response = f"Generated TTS with custom voice: {custom_voice_path} for text: {text}"
        
        
        with open(output_path, "wb") as f:
            f.write(response.encode())  
        return output_path
    except Exception as e:
        raise Exception(f"Error during custom voice synthesis: {str(e)}")
def refine_sentiment_analysis(user_input):
    doc = nlp(user_input)
    polarity = doc._.blob.polarity

    if "so high" in user_input or "too expensive" in user_input:
        sentiment = "NEGATIVE"
    else:
        sentiment = "POSITIVE" if polarity > 0 else "NEGATIVE" if polarity < 0 else "NEUTRAL"

    confidence = abs(polarity)
    return {
        "response": f"Sentiment: {sentiment}, Confidence: {confidence:.2f}"
    }

@app.route('/upload_custom_voice', methods=['POST'])
def upload_custom_voice():
    """
    Endpoint to upload a custom voice model or file.
    Expected format: .wav or .mp3 file representing the custom voice.
    """
    if 'custom_voice' not in request.files:
        return jsonify({"error": "No voice file uploaded."}), 400

    voice_file = request.files['custom_voice']
    filename = secure_filename(voice_file.filename)

    custom_voice_path = os.path.join(app.config['UPLOAD_FOLDER'], "custom_voice_" + filename)
    voice_file.save(custom_voice_path)

    session['custom_voice_path'] = custom_voice_path
    flash("Custom voice uploaded successfully!", "success")
    return jsonify({"message": "Custom voice uploaded successfully.", "path": custom_voice_path})



@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process_document", methods=["POST"])
def process_document():
    """
    Handles both PDF document and audio upload.
    Processes the document and optionally saves the uploaded audio file for future use.
    """
  
    if 'document' in request.files:
        pdf_files = request.files.getlist('document')
        raw_text = ""
        for pdf in pdf_files:
            filename = secure_filename(pdf.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf.save(pdf_path)
            text = get_pdf_text([pdf], max_pages=50)
            raw_text += text

        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            session['uploaded_text'] = raw_text
            flash("PDF documents processed and text extracted successfully!", "success")
        else:
            flash("No text found in the uploaded PDFs. Please check the document content.", "warning")
    else:
        flash("No PDF file uploaded. Please select a document.", "warning")

    if 'voice' in request.files:
        audio_file = request.files['voice']
        if audio_file.filename:
            audio_filename = secure_filename(audio_file.filename)
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_audio_" + audio_filename)
            audio_file.save(audio_path)

            
            session['uploaded_audio_path'] = audio_path
            flash("Audio file uploaded and saved successfully!", "success")
        else:
            flash("No audio file selected or uploaded.", "warning")

    return redirect(url_for('document_query'))


@app.route("/ask_question", methods=["POST"])
def ask_question():
    question = request.form['question']
    document_text = session.get('uploaded_text', '')

    if question and document_text:
        answers = []
        try:
            google_response = google_chat.send_message(f"Context: {document_text}\nQuestion: {question}")
            answers.append({"Google Generative AI": google_response.text})
        except Exception as e:
            answers.append({"Google Generative AI": f"Error: {str(e)}"})

        try:
            hf_response = qna_model(question=question, context=document_text)
            answers.append({"Hugging Face Q&A": hf_response['answer']})
        except Exception as e:
            answers.append({"Hugging Face Q&A": f"Error: {str(e)}"})

        return render_template("document_query.html", answers=answers)
    else:
        return render_template("document_query.html", answer="Document text is empty. Please upload a document with content.")

@app.route("/document_query", methods=["GET"])
def document_query():
    return render_template("document_query.html")

@app.route('/stt', methods=['POST'])
def stt_handler():
    """Handle Speech-to-Text requests with customizable accents"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files['audio']
    audio_filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)

   
    audio_file.save(audio_path)

    try:
        text = speech_to_text(audio_path)
        flash("Audio file processed and text extracted successfully!", "success")
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/tts', methods=['POST'])
def tts_handler():
    """Handle Text-to-Speech requests with customizable accents."""
    data = request.json
    text = data.get('message', '').strip()
    lang = data.get('lang', 'en')
    tld = data.get('tld', 'com')

    if not text:
        return jsonify({"error": "No text provided for TTS."}), 400

    try:
       
        custom_voice_path = session.get('custom_voice_path', None)

        if custom_voice_path:
           
            audio_path = text_to_speech_with_custom_voice(text, custom_voice_path)
        else:
           
            audio_path = text_to_speech(text, lang=lang, tld=tld)

        return send_file(audio_path, mimetype="audio/mp3", as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/google', methods=['POST'])
def google_chat_response():
    user_input_text = request.json.get('message')
    document_text = session.get('uploaded_text', '')

    try:
        if document_text:
            response_raw = google_chat.send_message(f"Context: {document_text}\nMessage: {user_input_text}")
            return jsonify({"response": response_raw.text})
        else:
            return jsonify({"response": "Document text is empty. Please upload a document with content."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chatgpt', methods=['POST'])
def chatgpt_response():
    user_input = request.json.get('message')
    document_text = session.get('uploaded_text', '')

    try:
        if document_text:
            context = f"Document Context: {document_text}\nUser Question: {user_input}"

            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": context}],
                max_tokens=300
            )

         
            return jsonify({"response": response['choices'][0]['message']['content']})

        else:
            return jsonify({"response": "Document text is empty. Please upload a document with content."})
    except openai.error.OpenAIError as e:
        
        app.logger.error(f"OpenAI API error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
       
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred."}), 500

    
@app.route('/huggingface', methods=['POST'])
def huggingface_response():
    user_input = request.json.get('message')
    
    text = "Data science is the study of data to extract meaningful insights that can be used to guide business decisions. It's a multidisciplinary field that combines principles from mathematics, statistics, computer engineering, and artificial intelligence."

    try:
        combined_input = f"Context: {text}\nMessage: {user_input}"
        print(f"Combined Input: {combined_input}") 
        result = generator(combined_input, max_length=100, num_return_sequences=1)
        print(f"Result: {result}") 
        return jsonify({"response": result[0]['generated_text']})
    except Exception as e:
        print(f"Error: {e}") 
        return jsonify({"error": str(e)}), 500



@app.route('/sentiment', methods=['POST'])
def sentiment_response():
    user_input = request.json.get('message', '').strip().lower() 
    
    if not user_input:
        return jsonify({"response": "Input text is empty. Please provide text for sentiment analysis."})

    
    predefined_response = PREDEFINED_RESPONSES.get(user_input)
    try:
        
        doc = nlp(user_input)
        polarity = doc._.blob.polarity
        sentiment = "POSITIVE" if polarity > 0 else "NEGATIVE" if polarity < 0 else "NEUTRAL"
        sentiment_result = f"Sentiment: {sentiment}, Confidence: {abs(polarity):.2f}"

        
        if predefined_response:
            response_text = f"{predefined_response}\n\n{sentiment_result}"
        else:
            response_text = sentiment_result

        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/classifier', methods=['POST'])
def classifier_response():
    user_input = request.json.get('message', '').strip()
    try:
        result = zero_shot_classifier(user_input, candidate_labels=["science", "technology", "politics", "sports", "environment"])
        top_label = result['labels'][0]
        confidence = result['scores'][0]
        response_text = f"Label: {top_label}, Confidence: {confidence:.2f}"
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/translator', methods=['POST'])
def translator_response():
    user_input = request.json.get('message', '').strip()
    try:
        
        result = english_to_french_translator(user_input)
        return jsonify({"response": result[0]['translation_text']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    

@app.route('/qna', methods=['POST'])
def qna_response():
    user_input = request.json.get('message')
    
    
    predefined_text = """
    Data science is the study of data to extract meaningful insights that can be used to guide business decisions. 
    It's a multidisciplinary field that combines principles from mathematics, statistics, computer engineering, 
    and artificial intelligence. Data scientists use tools like machine learning, data visualization, and statistical modeling 
    to analyze data and inform decisions.
    """

    try:
        if predefined_text: 
            result = qna_model(question=user_input, context=predefined_text)
            return jsonify({"response": result['answer']})
        else:
            return jsonify({"response": "Predefined text is empty. Please define a context."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ner', methods=['POST'])
def ner_response():
    user_input = request.json.get('message', '').strip()
    try:
        doc = nlp(user_input)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        return jsonify({"response": entities})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/arabic_translator', methods=['POST'])
def arabic_translator():
    try:
        data = request.json
        text = data.get("message", "")
        if not text:
            return jsonify({"response": "No text provided for translation."}), 400
       
        translated = english_to_arabic_translator(text)
        return jsonify({"response": translated[0]['translation_text']})
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500



if __name__ == "__main__":
    app.run(debug=True)