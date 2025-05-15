import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import gspread
from google.oauth2 import service_account
from datetime import datetime

#torch.classes = None  

def save_into_sheets(user_input, predictions):
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    
    credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)
    client = Client(scope=scope,creds=credentials)
    spreadsheetname = "Token Classification Logs"
    spread = Spread(spreadsheetname,client = client)
    worksheet = sh.worksheet("sheet1")
    df = DataFrame(worksheet.get_all_records())
    

    timestamp = datetime.now().isoformat()
    predictions_list = ", ".join([f"{w}:{l}" for w, l in predictions])
    col = [predictions_list,timestamp]
    spread.df_to_sheet(dataframe[col],sheet = spreadsheetname,index = False)

# Set up the Streamlit app
st.set_page_config(page_title="Token Classification", layout="wide")
st.title("🧠 Token Classification for Abbreviation Detection")
st.markdown("Detect abbreviations (AC) and their long forms (LF)")


# Load the best model
model_name = "slightlycodic/TC-ABB-BERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

label_list = ['O', 'B-AC', 'B-LF', 'I-LF']

label_colors = {
    "B-AC": "#f39c12",   
    "B-LF": "#27ae60",   
    "I-LF": "#2ecc71",   
    "O": None            
}
text_input = st.text_area("Enter a sentence:")

if st.button("🔍 Detect Abbreviations"):
    if not text_input.strip():
        st.warning("Please enter a sentence.")
    else:
        words = text_input.strip().split()
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True
        )
        inputs = {k: v for k, v in encoding.items() if k != "offset_mapping"}
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        word_ids = encoding.word_ids()
        predicted_labels = predictions[0].tolist()

        results = []
        seen = set()
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            label = label_list[predicted_labels[idx]]
            word = words[word_id]
            results.append((word, label))
            seen.add(word_id)
                
        try:
            save_into_sheets(text_input.strip(), results)
        except Exception as e:
            st.error("Failed to save log to Google Sheets.")
            st.exception(e)    
        

        styled_sentence = ""
        for word, label in results:
            if label == "O":
                styled_sentence += f"<span style='margin-right: 6px;'>{word}</span> "
            else:
                color = label_colors.get(label, "#eeeeee")
                styled_sentence += (
                    f"<span style='margin: 4px; display: inline-block; vertical-align: middle;'>"
                    f"<span style='background-color: {color}; border-radius: 8px; padding: 6px 10px;"
                    f"display: inline-flex; align-items: center; gap: 6px;'>"
                    f"<span style='font-weight: 500;'>{word}</span>"
                    f"<span style='background-color: rgba(0,0,0,0.1); border-radius: 4px; padding: 2px 6px;"
                    f"font-size: 11px; font-weight: bold;'>{label}</span>"
                    f"</span></span>"
                )
        st.markdown("### 🧾 Tagged Sentence")
        st.markdown(styled_sentence, unsafe_allow_html=True)
st.markdown(
    '<a href="https://docs.google.com/spreadsheets/d/1RKW3WQ9v8KthoPPuHaUTWT32M36u88g6h8_3jOLjPwo/edit?usp=sharing" target="_blank">📊 View Logs in Google Sheets</a>',
    unsafe_allow_html=True
)
