import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib

# Load the trained random forest model
rf = joblib.load('random_forest_model.joblib')

# Load the dataset
df = pd.read_csv('merged.csv')

# Get a list of unique team names
teams = sorted(df['HomeTeam'].unique())

# Define a function to make predictions
def predict_winner(home_team, away_team):
    # Get the latest match stats for the two teams
    latest_stats = df.loc[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) | (
                (df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))].tail(1)

    # Check if there's enough data to make a prediction
    if len(latest_stats) == 0:
        return 'Not enough data to make a prediction'

    # Prepare the input data for the random forest model
    X = latest_stats[['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']]

    # Make a prediction using the random forest model
    y_pred = rf.predict(X)

    # Map the predicted label to the corresponding team name
    winner = home_team if y_pred == 1 else away_team

    return winner

# Define the GUI
root = tk.Tk()
root.title("Soccer Match Predictor")

# Define style for the widgets
style = ttk.Style()

# Configure the style for the labels
style.configure("TLabel", font=("Helvetica", 12), foreground="#333")

# Configure the style for the buttons
style.configure("TButton", font=("Helvetica", 12))

# Configure the style for the entry fields
style.configure("TEntry", font=("Helvetica", 12), padding=10)

# Create the widgets
home_label = ttk.Label(root, text="Home Team:")
home_entry = ttk.Entry(root)
away_label = ttk.Label(root, text="Away Team:")
away_entry = ttk.Entry(root)
predict_button = ttk.Button(root, text="Predict Winner")

# Apply the style to the widgets
home_label.pack(pady=10)
home_entry.pack()
away_label.pack(pady=10)
away_entry.pack()
predict_button.pack(pady=20)
predict_button.configure(style="TButton")

# Define the function to handle button click events
def on_predict_click():
    home_team = home_entry.get()
    away_team = away_entry.get()

    if home_team not in teams:
        result_label.config(text=f"{home_team} is not a valid team name. Please try again.")
        return

    if away_team not in teams:
        result_label.config(text=f"{away_team} is not a valid team name. Please try again.")
        return

    winner = predict_winner(home_team, away_team)
    result_label.config(text=f"The predicted winner is {winner}")

# Bind the function to the button click event
predict_button.config(command=on_predict_click)

# Add a label to display the result
result_label = ttk.Label(root, text="", foreground="green", font=("Helvetica", 14, "bold"))
result_label.pack(pady=20)

# Start the main event loop
root.mainloop()
