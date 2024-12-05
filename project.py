import gradio as gr
import openai
import base64
import time
import os
import spacy
import faiss
import uuid
from sentence_transformers import SentenceTransformer
import mimetypes
import json

# OpenAI API 설정
openai.api_key = "EMPTY"
openai.api_base = "https://8b14-34-125-146-230.ngrok-free.app/v1"

# 모델 설정
models = openai.Model.list()
model = models.data[0].id

# SentenceTransformer 설정
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# SpaCy 설정
nlp = spacy.load("en_core_web_sm")

# FAISS 설정
user_document_stores = {}
user_faiss_indices = {}

def encode_base64_content_from_file(file_path: str) -> str:
    if not file_path:
        return ""
    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode("utf-8")
    return encoded_string

def run_openai_inference(messages) -> str:
    """Send messages to OpenAI API and get the response."""
    chat_completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=1024,
    )
    result = chat_completion.choices[0].message.content
    return result

def get_file_type(file_path) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if mime_type.startswith("image"):
            return "Image"
        elif mime_type == "application/pdf":
            return "PDF"
        else:
            return "Unknown"
    return "Unknown"

def initialize_user_data(user_id):
    """사용자별 Document Store와 FAISS Index 초기화."""
    if user_id not in user_document_stores:
        user_document_stores[user_id] = {}
        user_faiss_indices[user_id] = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())

# PDF 처리 기능
def process_pdf(file_path, user_id):
    """PDF 파일을 처리하고 사용자별로 저장."""
    initialize_user_data(user_id)

    # 사용자별 document_store와 index 가져오기
    document_store = user_document_stores[user_id]
    index = user_faiss_indices[user_id]

    with open(file_path, "r") as file:
        text = file.read()

    # SpaCy로 청킹
    doc = nlp(text)
    chunks = [sent.text for sent in doc.sents]

    # 문장 임베딩 생성
    embeddings = embedder.encode(chunks)

    # FAISS에 추가
    index.add(embeddings)

    # 사용자 document store에 저장
    doc_id = len(document_store)
    document_store[doc_id] = {"chunks": chunks, "embeddings": embeddings}
    return doc_id

def retrieve_top_k(query, user_id, k=5):
    """사용자별 FAISS에서 관련 청크 검색."""
    index = user_faiss_indices[user_id]
    document_store = user_document_stores[user_id]
    
    query_embedding = embedder.encode([query])[0]
    distances, indices = index.search([query_embedding], k)
    
    results = []
    for idx in indices[0]:
        for doc_id, doc_data in document_store.items():
            if idx < len(doc_data["chunks"]):
                results.append(doc_data["chunks"][idx])
                break
    return results

def add_image(content, image_path):
  image_base64 = encode_base64_content_from_file(image_path)
  content.append({"type": "image_url",
               "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})

def add_message(history: list, conversation: list, chat_input, user_id):
    content = []
    pdfs = []

    for file_path in chat_input["files"]:
        history.append({"role": "user", "content": {"path": file_path}})
        file_type = get_file_type(file_path)
        if file_type == "Image":
            add_image(content, file_path)
        elif file_type == "PDF":
            doc_id = process_pdf(file_path, user_id)
            pdfs.append(doc_id)
    
    if chat_input["text"]:
        history.append({"role": "user", "content": chat_input["text"]})
        content.append({"type": "text", "text": chat_input["text"]})

    conversation.append({
        "role": "user",
        "content": content,
    })

    if pdfs:
        query = chat_input["text"]
        top_k_chunks = retrieve_top_k(query, user_id, k=5)
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "다음은 관련 PDF 청크입니다:"},
                {"type": "text", "text": "\n".join(top_k_chunks)},
            ],
        })
    return history, conversation, gr.MultimodalTextbox(value=None, interactive=False), json.dumps(history, indent=4), json.dumps(conversation, indent=4)

def bot(history: list, conversation: list):
    response = run_openai_inference(conversation)

    conversation.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": response},
        ],
    })

    history.append({"role": "assistant", "content": ""})
    for character in response:
        history[-1]["content"] += character
        print("history -1 : ", history[-1]['content'])
        time.sleep(0.05)
        yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")
    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
    )
    conversation_state = gr.State([])
    user_id = gr.State(str(uuid.uuid4()))

    history_json = gr.Textbox(label="History (JSON)", lines=10, interactive=False)
    conversation_json = gr.Textbox(label="Conversation (JSON)", lines=10, interactive=False)

    # 레이아웃 구성
    with gr.Row():
        chatbot
        chat_input
        history_json
        conversation_json

    chat_msg = chat_input.submit(
        add_message,
        inputs=[chatbot, conversation_state, chat_input, user_id],
        outputs=[chatbot, conversation_state, chat_input, history_json, conversation_json],
    )
    bot_msg = chat_msg.then(
        bot,
        inputs=[chatbot, conversation_state],
        outputs=chatbot,
    )
    bot_msg.then(
        lambda: 
            gr.MultimodalTextbox(interactive=True),
            None,
            [chat_input]
    )
    bot_msg.then(
        lambda chatbot, conversation: (
            json.dumps(chatbot, indent=4),  # history_json 값
            json.dumps(conversation, indent=4),  # conversation_json 값
        ),
        inputs=[chatbot, conversation_state],
        outputs=[history_json, conversation_json],
    )

demo.queue().launch(share=False)
