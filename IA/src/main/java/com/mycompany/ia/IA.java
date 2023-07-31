/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.mycompany.ia;

import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;


/**
 *
 * @author 
 */


public class IA {


     private static final int CATEGORIES = 5;
    private static final int EPOCHS = 10;
    private static final int BATCH_SIZE = 32;

    public static void main(String[] args) {

        // Load the data
        String[] documents = {
            "O tempo está quente hoje",
            "A nova música do Justin Bieber é ótima",
            "O presidente foi eleito",
            "A economia está em recessão",
            "A nova tecnologia é revolucionária"
        };

        // Create the model
        TensorFlow tf = TensorFlow.instance();
        tf.createSession();
        
        // Create the input tensor
        Tensor inputTensor = tf.constant(documents);

        // Create the output tensor
        Tensor outputTensor = tf.matmul(inputTensor, tf.constant(new int[]{128, CATEGORIES}));

        // Compile the model
        tf.compile(outputTensor, "softmax");

        // Train the model
        tf.fit(inputTensor, outputTensor, EPOCHS, BATCH_SIZE);

        // Evaluate the model
        int accuracy = tf.evaluate(inputTensor, outputTensor)[1];
        System.out.println("Accuracy: " + accuracy);

        // Classify a new document
        String newDocument = "O novo álbum do Taylor Swift é incrível";
        int category = tf.predict(newDocument)[0];
        System.out.println("Category: " + category);
    }
}


