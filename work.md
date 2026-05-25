
streamer.py = mimics PLC (sends via TCP/IP)
ml_predictor.py = mimics Linux/Jetson ML computer
predictions.csv = mimics Database
simulator.py = mimics Mosquitto display


streamer.py
mimics PLC, sends via TCP/IP every 5 seconds

↓ TCP/IP

ml_predictor.py
receives data, cleans, predicts
publishes to Mosquitto via MQTT
also stores raw + predictions to CSV/database

↓ MQTT

simulator.py
subscribes to Mosquitto
displays live levels as they arrive



\- If it's a simple dense NN — yes, one row of 128 values goes in, one prediction comes out

\- If it's an LSTM — it needs a sequence of rows, maybe last 10-20 readings, to predict based on time patterns



Look at the model's input shape in the code. If it's `(128,)` — one row is enough. If it's `(20, 128)` or similar — it needs a sequence. Tell me what you see and I'll help you handle it.



\*\*"How does ML predict level 1 and level 2 from light sensor data?"\*\*

The light sensor measures the interface between aqueous and organic. The model was trained to map those 128 optical readings to two outputs — estimated level of liquid 1 and liquid 2. So yes, 128 inputs → 2 output predictions. One row in, two numbers out.



\*\*"Why is the ML predictor not the server?"\*\*

This is actually a great architectural instinct. You're right — it could be. But here's why the streamer is the server in your proof of concept:



The streamer mimics the PLC — in the real plant the PLC holds sensor data in registers and waits for something to read from it. The ML code reads from it like a client. So the streamer being the server matches the real architecture more accurately.



In the real plant it'll be:

```

PLC (server, holds sensor registers)

← ML code on Linux/Jetson (client, reads registers, sends predictions back)

```



\*\*"On floats in Modbus"\*\*

Standard Modbus registers are 16 bit integers — can't hold 176.2 directly. Two solutions:



Simple approach — multiply by 10 before sending, divide by 10 after receiving. 176.2 becomes 1762 as integer, receiver divides back to 176.2. Loses one decimal place but fine for your use case.



Proper approach — use two consecutive registers per float value. Pymodbus handles this with `client.read\_holding\_registers` and struct packing. More complex but precise.



For proof of concept — use the multiply by 10 approach. Keep it simple.



\---



\*\*Now here's your small version to build and test\*\*



Three files. Start simple with TCP sockets first, swap to Modbus after it works.



\---



\*\*File 1 — streamer.py (the fake PLC/server)\*\*



```python

import pandas as pd

import time

import socket

import json



\# Load your CSV

df = pd.read\_csv('six\_cell\_data\_sheet.csv')



\# TCP server setup

server = socket.socket(socket.AF\_INET, socket.SOCK\_STREAM)

server.bind(('localhost', 9999))

server.listen(1)



print("Streamer waiting for ML code to connect...")

conn, addr = server.accept()

print(f"ML code connected from {addr}")



\# Stream one row every 5 seconds

for index, row in df.iterrows():

&#x20;   data = row.to\_dict()

&#x20;   message = json.dumps(data) + '\\n'

&#x20;   conn.sendall(message.encode())

&#x20;   print(f"Sent row {index}")

&#x20;   time.sleep(5)



conn.close()

server.close()

```



\---



\*\*File 2 — ml\_predictor.py (the client)\*\*



```python

import socket

import json

import numpy as np

import pandas as pd

import csv

from tensorflow import keras



\# Load your existing model

model = keras.models.load\_model('your\_model.h5')



\# Connect to streamer

client = socket.socket(socket.AF\_INET, socket.SOCK\_STREAM)

client.connect(('localhost', 9999))

print("Connected to streamer")



\# Output CSV setup

output\_file = open('predictions.csv', 'w', newline='')

writer = csv.writer(output\_file)

writer.writerow(\['timestamp', 'level\_1\_predicted', 'level\_2\_predicted'])



buffer = ''



while True:

&#x20;   # Receive data

&#x20;   chunk = client.recv(4096).decode()

&#x20;   if not chunk:

&#x20;       break

&#x20;   

&#x20;   buffer += chunk

&#x20;   

&#x20;   # Process complete messages

&#x20;   while '\\n' in buffer:

&#x20;       line, buffer = buffer.split('\\n', 1)

&#x20;       if not line:

&#x20;           continue

&#x20;           

&#x20;       # Parse incoming row

&#x20;       row\_data = json.loads(line)

&#x20;       

&#x20;       # Convert to numpy array for model

&#x20;       # Adjust column selection to match your model input

&#x20;       input\_data = np.array(list(row\_data.values())).reshape(1, -1)

&#x20;       

&#x20;       # Normalize — same as training

&#x20;       input\_normalized = input\_data / 400.0

&#x20;       

&#x20;       # Predict

&#x20;       prediction = model.predict(input\_normalized, verbose=0)

&#x20;       

&#x20;       level\_1 = float(prediction\[0]\[0])

&#x20;       level\_2 = float(prediction\[0]\[1])

&#x20;       

&#x20;       print(f"Level 1: {level\_1:.3f} | Level 2: {level\_2:.3f}")

&#x20;       

&#x20;       # Store to CSV

&#x20;       import datetime

&#x20;       writer.writerow(\[datetime.datetime.now(), level\_1, level\_2])

&#x20;       output\_file.flush()



output\_file.close()

client.close()

```



\---



\*\*File 3 — simulator.py (the display)\*\*



```python

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.animation as animation

import time



fig, ax = plt.subplots(figsize=(8, 6))



def animate(frame):

&#x20;   try:

&#x20;       # Read latest predictions

&#x20;       df = pd.read\_csv('predictions.csv')

&#x20;       

&#x20;       if df.empty:

&#x20;           return

&#x20;       

&#x20;       # Get latest row

&#x20;       latest = df.iloc\[-1]

&#x20;       level\_1 = latest\['level\_1\_predicted']

&#x20;       level\_2 = latest\['level\_2\_predicted']

&#x20;       

&#x20;       ax.clear()

&#x20;       

&#x20;       # Draw liquid levels

&#x20;       bars = ax.bar(

&#x20;           \['Aqueous (Level 1)', 'Organic (Level 2)'],

&#x20;           \[level\_1, level\_2],

&#x20;           color=\['#2196F3', '#FF9800'],

&#x20;           width=0.4

&#x20;       )

&#x20;       

&#x20;       ax.set\_ylim(0, 1)

&#x20;       ax.set\_ylabel('Predicted Level (normalized)')

&#x20;       ax.set\_title('REE Separator — Live Level Prediction')

&#x20;       

&#x20;       # Add value labels on bars

&#x20;       for bar, val in zip(bars, \[level\_1, level\_2]):

&#x20;           ax.text(

&#x20;               bar.get\_x() + bar.get\_width()/2,

&#x20;               bar.get\_height() + 0.02,

&#x20;               f'{val:.3f}',

&#x20;               ha='center',

&#x20;               fontsize=12,

&#x20;               fontweight='bold'

&#x20;           )

&#x20;       

&#x20;       # Show total rows processed

&#x20;       ax.text(0.02, 0.95, f'Readings processed: {len(df)}',

&#x20;               transform=ax.transAxes, fontsize=10, color='gray')

&#x20;               

&#x20;   except Exception as e:

&#x20;       pass



ani = animation.FuncAnimation(fig, animate, interval=2000)

plt.tight\_layout()

plt.show()

```



\---



\*\*How to run all three\*\*



Open three separate terminals:



Terminal 1:

```

python streamer.py

```



Terminal 2:

```

python ml\_predictor.py

```



Terminal 3:

```

python simulator.py

```



Watch data flow from streamer → predictor → simulator in real time.



\---



\*\*Once this works — swap to Modbus\*\*



Replace the socket code in streamer.py and ml\_predictor.py with pymodbus. The ML logic and simulation stay identical. Only the communication layer changes. That swap proves Modbus works without risking the whole system.



\---



\*\*For your supervisor's presentation\*\*



Once all three run together — record a short screen capture showing:

\- Streamer sending rows

\- ML code receiving and predicting

\- Simulator bars updating in real time



That's a presentable proof of concept. Simple, clear, demonstrates the full pipeline end to end. Your supervisor can show that to stakeholders and say the feature works in principle.



Get streamer.py running first tonight. Everything else builds from there.

