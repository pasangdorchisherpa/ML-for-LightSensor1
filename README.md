# ML-for-LightSensor1
Just a small ML project, has random generated data!

__________________

# Overview

I have a light sensor pointed through a pipe that contains liquid. The more liquid there is, the more it blocks the light. The sensor can't directly tell me how much liquid is present — it can only report light intensity. So the goal is to train a neural network to figure out the liquid level just by looking at the light readings.
To do this without real hardware, I simulate fake sensor data using a simple physics rule: base_light = 1.0 - 0.7 * level. This means a full pipe gives dim light, an empty pipe gives bright light. I also add a small amount of random noise to each reading to make it realistic — real sensors are never perfectly clean.
Instead of feeding the NN a single reading, I feed it a window of 32 readings at once. One reading is too noisy to be useful — but a pattern of 32 readings contains enough information to make a confident prediction. This is the same reason the real system uses 128-sample windows.
The NN learns the mapping: light pattern → liquid level. Once trained, it can take a new window of readings and predict the liquid level it's never seen before.
Finally, I convert that prediction into a valve command using a simple rule — the higher the liquid level, the more the valve closes. This closes the loop: sensor → inference → control action.

_________________

Raw light readings (32 samples)

        ↓
Scale the data (÷ 1.2)

        ↓
Neural Network (32 → 64 → 1)
        
        ↓
Predicted liquid level [0 to 1]
        
        ↓
Valve command (1 - level)
