# Baseball-Pitch-Sequence-Prediction
This GitHub repository is dedicated to the development and training of a Transformer-based deep learning model for predicting the next pitch type in a baseball game, utilizing historical data on ball-strike counts and pitch sequences to incorporate transition probabilities for more accurate and context-aware predictions.

### Synthetic Baseball Pitch Sequence Simulation Using Markov Models

#### Overview
This section of the project involves simulating realistic baseball pitch sequences using a Markov model approach. The simulation is based on the probabilistic transitions between different states, defined by the count of balls and strikes during an at-bat. The output is a synthetic dataset that includes detailed sequences of pitch types and outcomes (ball, strike, or hit).

#### Description
The simulation leverages Markov models to generate sequences where each state represents a specific ball-strike count, and transitions depend only on the current state. This method effectively captures the dynamics of a baseball game where the type and outcome of each pitch can influence subsequent pitches.

#### Key Features
- **Transition Probabilities:** Defined for each possible count, detailing the likelihood of each type of pitch and its outcome.
- **Pitch Simulation:** For each state, the model selects a pitch type based on defined probabilities and determines the result of the pitch.
- **Count Updates:** Updates the ball and strike count based on the outcome of each pitch, respecting the rules of baseball regarding walks and strikeouts.
- **At-Bat Simulation:** Integrates pitch simulation and count updating to simulate the sequence of pitches in a complete at-bat.
- **Dataset Generation:** Multiple at-bats are simulated to produce a comprehensive dataset. This dataset includes columns for balls, strikes, pitch type, and pitch outcome, which can be used to train machine learning models for predictive analytics in sports.

#### Utility
This simulation tool is invaluable for analysts and enthusiasts interested in understanding pitching strategies and game dynamics. It provides a foundation for developing predictive models that could forecast pitch types based on the game situation, enhancing strategic decisions in real-time. The generated dataset can be employed to train and evaluate machine learning models, facilitating deeper insights into the factors influencing pitcher behavior and outcomes in baseball.

#### Output
The output of this simulation is a CSV file (`baseball_pitch_data.csv`) containing the synthetic dataset. Each entry in the dataset reflects an individual pitch within an at-bat, capturing the balls, strikes, type of pitch, and its outcome before the pitch was thrown.

### How to Use
To run the simulation and generate the dataset, execute the provided Python scripts in the `scripts` folder. Ensure that Python and required libraries (`pandas`, `matplotlib`, `seaborn`) are installed. The scripts can be run from any standard Python environment.

This dataset can be directly used for training predictive models or can be further analyzed to extract insights into pitching strategies under different game scenarios.

## Next steps

Add prediction attention-based transformer, long short-term memory (LSTM), and hidden Markov model (HMM) sequence prediction models and compare performance.
