from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pandas as pd
import matplotlib.pyplot as plt
import os
import sqlite3
from flask_session import Session
import matplotlib
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random


matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = ''  # Replace with your secret key

# Configure session to use filesystem
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load dataset
def load_dataset(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    print("Dataset columns:", df.columns)  # Debugging output to check column names
    return df

class ConversationState:
    def __init__(self):
        self.context = {"is_job_position_verified": False}
        self.candidate_responses = []
        self.evaluation_metrics = []
        self.scores = []  # Track scores for each response
        self.correct_answers = 0  # Track the number of correct answers
        self.incorrect_answers = 0  # Track the number of incorrect answers

    def update_context(self, key, value):
        self.context[key] = value

    def get_context(self, key):
        return self.context.get(key, None)

    def add_candidate_response(self, question, response, score):
        self.candidate_responses.append({'question': question, 'response': response, 'score': score})
        self.scores.append(score)  # Add the score to the scores list
        if score >= 0.5:  # Define threshold for considering an answer as "correct"
            self.correct_answers += 1
        else:
            self.incorrect_answers += 1

    def reset(self):
        self.context = {}
        self.candidate_responses = []
        self.scores = []
        self.correct_answers = 0
        self.incorrect_answers = 0

    def calculate_average_score(self):
        if self.scores:
            return sum(self.scores) / len(self.scores)
        return 0
    
    def get_results(self):
        return {"correct_answers": self.correct_answers, "incorrect_answers": self.incorrect_answers}


class InterviewBot:
    def __init__(self, dataset):
        self.dataset = dataset
        self.vectorizer = TfidfVectorizer()

    def get_questions_for_position_and_Module(self, company, job_position, Module):
        # Adjust the query to include the Module
        questions = self.dataset[
            (self.dataset['company_name'].str.lower() == company.lower()) & 
            (self.dataset['Job_Position'].str.lower() == job_position.lower()) &
            (self.dataset['Module'].str.lower() == Module.lower())
        ]['Questions'].tolist()
        random.shuffle(questions)
        return questions

    def get_answer_for_question(self, question):
        return self.dataset[self.dataset['Questions'] == question]['Answers'].values[0]

    def start_interview(self, company, job_position, Module, state):
        state.update_context("company", company)
        state.update_context("job_position", job_position)
        state.update_context("Module", Module)
        state.update_context("questions", self.get_questions_for_position_and_Module(company, job_position, Module))
        state.update_context("current_question_index", 0)
        state.update_context("asked_questions", set())  # Track asked questions
        return self.get_next_question(state)

    def get_next_question(self, state):
        questions = state.get_context("questions")
        current_index = state.get_context("current_question_index")
        asked_questions = state.get_context("asked_questions")

        # Find the next question that hasn't been asked yet
        while current_index < len(questions) and questions[current_index] in asked_questions:
            current_index += 1

        if current_index < len(questions):
            question = questions[current_index]
            state.update_context("current_question_index", current_index + 1)
            asked_questions.add(question)
            state.update_context("asked_questions", asked_questions)
            return question
        else:
            return None

    def verify_job_position(self, company, job_position, Module, state):
        self.dataset[['company_name', 'Job_Position', 'Module']] = self.dataset[['company_name', 'Job_Position', 'Module']].fillna('')
        available_positions = self.dataset[['company_name', 'Job_Position', 'Module']].apply(
            lambda x: (
                x['company_name'].lower(),
                x['Job_Position'].lower(),
                x['Module'].lower()
            ),
            axis=1
        ).tolist()
        if (company.lower(), job_position.lower(), Module.lower()) in available_positions:
            first_question = self.start_interview(company, job_position, Module, state)
            return True, first_question
        else:
            return False, None
        
    def compute_similarity(self, user_response, correct_answer):
        responses = [user_response, correct_answer]
        tfidf_matrix = self.vectorizer.fit_transform(responses)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cosine_sim[0][0]  # Return the similarity score

    def respond(self, user_input, state, company=None, domain=None, Module=None):
        if state.get_context("is_job_position_verified") is None:
            state.update_context("is_job_position_verified", False)
        if company and domain and Module and not state.get_context("is_job_position_verified"):
            is_verified, first_question = self.verify_job_position(company, domain, Module, state)
            if is_verified:
                state.update_context("is_job_position_verified", True)
                return f"Let's get started with your interview for the {state.get_context('job_position')} role at {state.get_context('company')} focusing on {state.get_context('Module')}. Ready? Here’s your first question: {first_question}. Good luck!"
            else:
                return "Your self-introduction doesn’t seem to be valid. It’s important to share a clear and relevant introduction to make the best use of the AI assistant. Kindly log out and revisit the page when you’re ready to try again. We’re here to support you every step of the way!"

        if user_input.lower() == "quit":
            state.reset()
            return "Interview session ended. Thank you for choosing the Interview Prep Chatbot! Your preparation doesn’t have to stop here—consider upgrading to unlock more questions, advanced features, and take your readiness to the next level."

        current_index = state.get_context("current_question_index") - 1
        questions = state.get_context("questions")
        if current_index < len(questions):
            current_question = questions[current_index]
            correct_answer = self.get_answer_for_question(current_question)
            similarity_score = self.compute_similarity(user_input, correct_answer)
            state.add_candidate_response(current_question, user_input, similarity_score)

            next_question = self.get_next_question(state)
            if next_question:
                if similarity_score < 0.2:
                    return f"Your score is {similarity_score * 100:.2f}%. Don't worry, every response is a step forward! Let's keep going and see what comes next: {next_question}"
                elif similarity_score < 0.4:
                    return f"You're getting there! Your response shows potential with {similarity_score * 100:.2f}%. Ready for the next challenge? Here it is: {next_question}"
                elif similarity_score < 0.6:
                    return f"Nice work! You're on the right track with {similarity_score * 100:.2f}%. Let's continue with the next question: {next_question}"
                elif similarity_score < 0.8:
                    return f"Great job! Your response is quite strong at {similarity_score * 100:.2f}%. Let's keep the momentum going: {next_question}"
                else:
                    return f"Fantastic! You nailed it, Your response is {similarity_score * 100:.2f}% aligned with what we're looking for. Let's move on to the next question: {next_question}"
            else:
                average_score = state.calculate_average_score()
                results = state.get_results()
                if average_score < 0.5:
                    return (
                        "Interview completed.\n"
                        f"Your final score is {average_score * 100:.2f}% with {results['correct_answers']} correct and {results['incorrect_answers']} incorrect answers.\n"
                        "It seems there’s still some room for growth, but remember every mistake is a step toward improvement! "
                        "Keep practicing, and you’ll be amazed at how much progress you can make. "
                        "Don't forget, learning is a journey.\n"
                        "Type 'quit' to end the conversation or Upgrade the plan for more questions and features to help you sharpen your skills."
                    )
                else:
                    return (
                        "Interview completed. Well done!\n"
                        f"Interview completed. Congratulations! Your final score is {average_score * 100:.2f}% with {results['correct_answers']} correct and {results['incorrect_answers']} incorrect answers."
                        "Great job! You’re on the right track and demonstrating strong potential. "
                        "Keep refining your skills and aiming for perfection. Remember, consistent practice will only push you further ahead!\n"
                        "Type 'quit' to end the conversation or Upgrade the plan for more questions and advanced insights to take your preparation to the next level."
                    )
        else:
            return "Congratulations on completing the interview! We appreciate your effort and responses. To continue advancing your skills and gain access to a wider range of questions and premium features, consider upgrading your plan. Elevate your preparation and achieve your career goals with our enhanced resources and support!"

# SQLite database setup
def get_db_connection():
    conn = sqlite3.connect('user_data.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    if 'user_id' not in session:
        print("User is not logged in. Redirecting to login page.")
        return redirect(url_for('login'))
    print(f"User {session.get('email')} is logged in. Displaying interview bot page.")
    return render_template('index.html', email=session.get('email'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        confirm_password = request.form['confirm-password']

        if password != confirm_password:
            return 'Passwords do not match!'

        conn = get_db_connection()
        conn.execute('INSERT INTO users (name, email, phone, password) VALUES (?, ?, ?, ?)',
                     (name, email,phone, password))
        conn.commit()
        conn.close()

        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND password = ?',
                            (email, password)).fetchone()
        conn.close()

        if user:
            session['user_id'] = user['id']
            session['email'] = user['email']
            session['state'] = pickle.dumps(ConversationState()).hex()  # Initialize session state
            return redirect(url_for('index'))
        else:
            return 'Invalid credentials!'

    return render_template('login.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('state', None)
    return redirect(url_for('login'))

@app.route('/index6_c', methods=['GET','POST'])
def pageC():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    # Handle POST request parameters
    if request.method == 'POST':
        company = request.form.get('company')
        domain = request.form.get('domain')
        Module = request.form.get('Module')
    else:
        # Handle GET request parameters
        company = request.args.get('company')
        domain = request.args.get('domain')
        Module=request.args.get('Module')

    # Load the state from the session
    state = pickle.loads(bytes.fromhex(session['state']))

    # Get the results
    results = state.get_results()
    correct_answers = results['correct_answers']
    incorrect_answers = results['incorrect_answers']

    # Render the page with the results
    return render_template('index6_c.html', 
                           company=company, 
                           domain=domain,
                           Module=Module,
                           correct_answers=correct_answers,
                           incorrect_answers=incorrect_answers)

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Ensure state is in session
    if 'state' not in session:
        session['state'] = pickle.dumps(ConversationState()).hex()

    user_input = request.json.get('message')
    company = request.json.get('company')
    domain = request.json.get('domain')
    Module = request.json.get('Module')

    # Load the state from the session
    state = pickle.loads(bytes.fromhex(session['state']))
    if user_input.lower() == "quit":
        # Reset state if needed
        # state.reset()
        # Save the updated state back to the session
        session['state'] = pickle.dumps(state).hex()
        # Redirect to /index6_c with company and domain parameters
        return redirect(url_for('pageC', company=company, domain=domain, Module=Module))
    
    response = bot.respond(user_input, state, company=company, domain=domain, Module=Module)

    # Save the updated state back to the session
    session['state'] = pickle.dumps(state).hex()
    return jsonify({'response': response})

@app.route('/get_data', methods=['GET'])
def get_data():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    file_path = os.path.join(os.path.dirname(__file__), 'interview.csv')
    try:
        # Read the CSV file with ISO-8859-1 encoding
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
        
        # Replace NaN values with an empty string or any other placeholder
        data = data.fillna('')

        # Convert the DataFrame to a list of dictionaries and return as JSON
        return jsonify(data.to_dict(orient='records'))
    except Exception as e:
        print(f"Error reading CSV file: {e}")  # Log error to console
        return jsonify({"error": str(e)}), 500


@app.route('/get_df', methods=['GET'])
def get_df():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    state = pickle.loads(bytes.fromhex(session['state']))
    metrics = bot.complete_interview(state)
    return jsonify(metrics)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Load the state from the session
    state = pickle.loads(bytes.fromhex(session['state']))
    
    # Get the results
    results = state.get_results()
    correct_answers = results['correct_answers']
    incorrect_answers = results['incorrect_answers']

    # Render the dashboard page with the results
    return render_template('dashboard1.html', 
                           correct_answers=correct_answers, 
                           incorrect_answers=incorrect_answers)

@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    state = pickle.loads(bytes.fromhex(session['state']))
    metrics = bot.complete_interview(state)
    return render_template('dashboard1.html', metrics=metrics)

if __name__ == "__main__":
    dataset = load_dataset('interview.csv')
    bot = InterviewBot(dataset)
    app.run(host='0.0.0.0', port=5001)
