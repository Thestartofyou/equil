import numpy as np

# Define the signaling game parameters
num_states = 2
num_actions = 2

# Define the payoffs for the Receiver
receiver_payoffs = np.array([[5, 0], [0, 10]])

# Define the Sender's types, actions, and beliefs
sender_types = [0, 1]  # Types: 0 (Low quality), 1 (High quality)
sender_actions = [0, 1]  # Actions: 0 (Don't signal), 1 (Signal)
sender_beliefs = np.array([[[0.8, 0.2], [0.4, 0.6]],  # Beliefs for type 0
                           [[0.2, 0.8], [0.6, 0.4]]])  # Beliefs for type 1

# Define the Receiver's beliefs
receiver_beliefs = np.array([[[0.6, 0.4], [0.4, 0.6]],  # Beliefs if signal is observed
                              [[0.4, 0.6], [0.6, 0.4]]])  # Beliefs if signal is not observed

# Define the update rules for Sender's beliefs
sender_update_rules = [
    lambda beliefs, action: beliefs / np.sum(beliefs, axis=1, keepdims=True) if action == 0 else beliefs,
    lambda beliefs, action: beliefs / np.sum(beliefs, axis=1, keepdims=True)
]

# Define the update rule for Receiver's beliefs
def update_receiver_beliefs(receiver_beliefs, sender_action):
    return receiver_beliefs[sender_action]

# Check if the given strategy profile, beliefs, and update rules constitute a PBE
def is_perfect_bayesian_equilibrium(sender_actions, sender_beliefs, receiver_beliefs, sender_update_rules):
    for sender_type in range(len(sender_types)):
        for sender_action in sender_actions:
            sender_belief = sender_beliefs[sender_type][sender_action]
            sender_belief = sender_update_rules[sender_type](sender_belief, sender_action)
            receiver_belief = receiver_beliefs[sender_action]
            if not np.allclose(sender_belief, receiver_belief):
                return False
    return True

# Example usage
if is_perfect_bayesian_equilibrium(sender_actions, sender_beliefs, receiver_beliefs, sender_update_rules):
    print("The given strategy profile, beliefs, and update rules constitute a Perfect Bayesian Equilibrium (PBE).")
else:
    print("The given strategy profile, beliefs, and update rules do not constitute a PBE.")
