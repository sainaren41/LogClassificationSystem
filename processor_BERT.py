
import joblib
from sentence_transformers import SentenceTransformer

transform_model = SentenceTransformer('all-MiniLM-L6-v2')
classifier_model = joblib.load('models/log_classifier.joblib')

def classify_with_bert(log_message):
    msg_embed = transform_model.encode([log_message]) 
    probality = classifier_model.predict_proba(msg_embed)[0]
    if max(probality) < 0.5:
        return "Unclassified"

    predict_class = classifier_model.predict(msg_embed)

    return predict_class[0]

if __name__ == "__main__":
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)