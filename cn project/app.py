from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno4/main/spam.csv", encoding='ISO-8859-1')
df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)
df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Train Naive Bayes model
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
X_train, y_train = df['Message'], df['Spam']
clf.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_spam', methods=['POST'])
def check_spam():
    data = request.get_json()
    email_content = data.get('email_content')
    prediction = clf.predict([email_content])[0]
    result = "This is a Spam Email!" if prediction == 1 else "This is a Safe Email!"
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
