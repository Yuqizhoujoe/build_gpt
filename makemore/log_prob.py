# %% [markdown]
### Log Probability
# %%
import matplotlib.pyplot as plt
import numpy as np

# Example probabilities
probabilities = np.linspace(0.01, 1, 100)
print(f"probabilities: {probabilities}")

# Calculate log probabilities
log_probabilities = np.log(probabilities)
print(f"log_probabilities: {log_probabilities}")

# Plot the graph
plt.figure(figsize=(6, 4))
plt.plot(probabilities, log_probabilities, color='blue')
plt.title('Logarithmic Score vs Probability')
plt.xlabel('Probability Assigned to True Event')
plt.ylabel('Logarithmic Score')
plt.grid(True)
plt.show()
# %%
