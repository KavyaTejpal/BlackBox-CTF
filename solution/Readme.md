# Solution to `BlackBox`

The TensorFlow Lite model file needs to be analysed. TensorFlow can be used to do so.

Load the model and extract all tensor data from the layers. Look at the bias values in each layer and try to convert them to ASCII characters one by one.

The attack steps:
1. **Load the TFLite model**: Use `tf.lite.Interpreter` to load `challenge.tflite`
2. **Extract tensor details**: Get all tensors and their data
3. **Convert to ASCII**: Treat tensor values (especially biases) as ASCII character codes
4. **Identify Base64**: Look for printable strings that appear to be Base64 encoded (ending in `==`)
5. **Decode Base64**: Decode the Base64 string to reveal the flag

A python script can be created to show how extraction can be done by iterating through all tensors, converting their values to ASCII, and attempting Base64 decoding on any suspicious strings.

**Flag:** `LNMHACKS{d33p_l34rn1ng_r3v3rs3_3ng1n33r1ng}`