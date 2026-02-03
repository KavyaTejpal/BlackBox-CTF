# BlackBox, 200 points

## Author: [KavyaTejpal](__https://github.com/KavyaTejpal__)

## Problem

Dr. Elena Voss, a brilliant AI researcher, vanished three months ago after discovering something dangerous. Her last message: "I've hidden it where they'd never look - compressed, encoded, in plain sight." All that remains is a mysterious file from her workstation. Can you uncover what she was protecting before corporate security erases the evidence?

A challenge file `challenge.tflite` is given.

## Producing it

Use a Python script to generate `challenge.tflite` which is made such that a particular tensor's bias values are set to ASCII values of a Base64-encoded flag. The flag is hidden within the TensorFlow Lite model's internal parameters.