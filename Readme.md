# BlackBox - Official Writeup

## Challenge Information

**Challenge Name:** BlackBox  
**Category:** Reverse Engineering / Forensics  
**Difficulty:** Medium  
**Points:** 500  
**Flag:** `LNMHACKS{d33p_l34rn1ng_r3v3rs3_3ng1n33r1ng}`

## Challenge Description

Dr. Elena Voss, a brilliant AI researcher, vanished three months ago after discovering something dangerous. Her last message: *"I've hidden it where they'd never look - compressed, encoded, in plain sight."* All that remains is a mysterious file from her workstation. Can you uncover what she was protecting before corporate security erases the evidence?

**Files Provided:** `challenge.tflite`

## Solution

### Step 1: Identify the File Type

First, let's examine what we're dealing with:

```bash
$ file challenge.tflite
challenge.tflite: data

$ hexdump -C challenge.tflite | head
00000000  54 46 4c 33 00 00 00 00  ...
```

The file appears to be a TensorFlow Lite model (`.tflite` extension is a hint). TFLite models are used for deploying machine learning models on mobile and embedded devices.

### Step 2: Explore the Model Structure

We can visualize the model using Netron:

```bash
$ pip install netron
$ netron challenge.tflite
```

Opening in Netron reveals:
- Input layer (10 neurons)
- Multiple hidden layers (FullyConnected/Dense layers)
- Output layer (1 neuron)

The model has several layers with weights and biases. The bias values are where data might be hidden.

### Step 3: Extract Model Tensors

Let's write a Python script to extract and analyze the model's internal data:

```python
import tensorflow as tf
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='challenge.tflite')
interpreter.allocate_tensors()

# Get all tensor details
tensor_details = interpreter.get_tensor_details()

# Extract each tensor
for detail in tensor_details:
    tensor_data = interpreter.get_tensor(detail['index'])
    print(f"Tensor: {detail['name']}")
    print(f"Shape: {tensor_data.shape}")
    print(f"Data: {tensor_data.flatten()[:20]}...")  # First 20 values
    print()
```

### Step 4: Convert Tensor Data to ASCII

Looking at the bias values from `hidden_layer_1`, we notice they look like ASCII character codes:

```python
# Extract bias from first hidden layer
bias = interpreter.get_tensor(bias_tensor_index)

# Convert float values to integers (ASCII codes)
ascii_values = bias.astype(np.uint8)

# Convert to string
decoded = ascii_values.tobytes().decode('utf-8', errors='ignore')
printable = ''.join(c for c in decoded if c.isprintable())

print(f"Decoded string: {printable}")
```

Output:
```
Decoded string: TE5NSEFDS1N7ZDMzcF9sMzRybjFuZ19yM3YzcnMzXzNuZzFuMzNyMW5nfQ==
```

This looks like Base64! Notice the `==` padding at the end.

### Step 5: Decode Base64

```python
import base64

base64_string = "TE5NSEFDS1N7ZDMzcF9sMzRybjFuZ19yM3YzcnMzXzNuZzFuMzNyMW5nfQ=="
flag = base64.b64decode(base64_string).decode('utf-8')

print(f"Flag: {flag}")
```

Output:
```
Flag: LNMHACKS{d33p_l34rn1ng_r3v3rs3_3ng1n33r1ng}
```

### Complete Solution Script

```python
#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import base64

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='challenge.tflite')
interpreter.allocate_tensors()

# Get all tensor details
tensor_details = interpreter.get_tensor_details()

print("=== Extracting Hidden Data ===\n")

# Check each tensor
for detail in tensor_details:
    tensor_data = interpreter.get_tensor(detail['index'])
    
    try:
        # Convert to ASCII
        flat_data = tensor_data.flatten()
        uint8_data = flat_data.astype(np.uint8)
        decoded = uint8_data.tobytes().decode('utf-8', errors='ignore')
        printable = ''.join(c for c in decoded if c.isprintable())
        
        if printable and len(printable) > 10:
            print(f"Tensor: {detail['name']}")
            print(f"ASCII: {printable}")
            
            # Check if it's base64
            if '=' in printable or all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in printable.strip()):
                try:
                    flag = base64.b64decode(printable.strip()).decode('utf-8')
                    if 'LNMHACKS{' in flag:
                        print(f"\n🎉 FLAG: {flag}\n")
                except:
                    pass
    except:
        pass
```

## Key Concepts

1. **TensorFlow Lite Models**: Binary format for ML model deployment
2. **Model Weights & Biases**: Neural network parameters that can store arbitrary data
3. **Steganography**: Hiding data in plain sight within legitimate file formats
4. **Base64 Encoding**: Common encoding scheme that's easily reversible
5. **ASCII Conversion**: Converting numeric data to text characters

## Tools Used

- `file` - Identify file type
- `netron` - Visualize neural network models
- `tensorflow` - Load and extract TFLite model data
- `base64` - Decode base64 strings
- Python - Scripting and automation

## Learning Takeaways

- Machine learning model files can hide arbitrary data
- Always examine binary files for embedded information
- Recognize common encoding patterns (Base64, hex, etc.)
- Neural network models have many places to hide data (weights, biases, metadata)

## Flag

`LNMHACKS{d33p_l34rn1ng_r3v3rs3_3ng1n33r1ng}`