
package com.mycompany.ia;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.nn.LSTM;
import org.tensorflow.op.nn.Softmax;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;
import java.util.Random;

public class NewsClassifier {
    public static void main(String[] args) {
        // Dados de treinamento (exemplo simplificado)
        float[][] trainingData = {
            {0.1f, 0.2f, 0.3f, 0.4f, 0.5f}, // Notícia 1
            {0.5f, 0.4f, 0.3f, 0.2f, 0.1f}, // Notícia 2
            // ... adicione mais notícias de treinamento aqui
        };

        // Rótulos das notícias de treinamento (índice da categoria)
        int[] trainingLabels = {1, 0, /* ... */};

        // Hiperparâmetros do modelo
        int numCategories = 5; // Número de categorias (clima, música, política, economia, tecnologia)
        int inputSize = trainingData[0].length; // Tamanho da entrada (tamanho do vetor de notícias)
        int hiddenSize = 10; // Tamanho da camada oculta LSTM
        int batchSize = 1; // Tamanho do lote (neste exemplo, usamos apenas um exemplo de treinamento por vez)
        int numEpochs = 100; // Número de épocas de treinamento

        // Construir o grafo do TensorFlow
        try (Graph graph = new Graph()) {
            Ops ops = Ops.create(graph);

            // Criar os placeholders para as entradas (dados de treinamento) e os rótulos
            Placeholder<Float> inputPlaceholder = ops.placeholder(Float.class, Placeholder.shape(batchSize, inputSize));
            Placeholder<Integer> labelPlaceholder = ops.placeholder(Integer.class, Placeholder.shape(batchSize));

            // Criar a camada LSTM
            LSTM lstmLayer = LSTM.create(ops, hiddenSize, null, null, null, 0);
            LSTM.LSTMCell lstmCell = lstmLayer.cell();

            // Inicializar as variáveis para as matrizes de pesos e biases da camada LSTM
            Variable<Float> lstmWeights = ops.variable(ops.randomUniform(ops.constant(new long[]{inputSize + hiddenSize, 4 * hiddenSize}), Float.class), Float.class);
            Variable<Float> lstmBiases = ops.variable(ops.zeros(ops.constant(new long[]{4 * hiddenSize}), Float.class), Float.class);

            // Calcular a saída da camada LSTM
            LSTM.LSTMTuple lstmOutput = lstmCell.call(
                    LSTM.LSTMCell.Args.builder()
                            .seqLen(ops.constant(batchSize))
                            .inputs(inputPlaceholder)
                            .c(ops.zeros(ops.constant(new long[]{batchSize, hiddenSize}), Float.class))
                            .h(ops.zeros(ops.constant(new long[]{batchSize, hiddenSize}), Float.class))
                            .weights(lstmWeights)
                            .biases(lstmBiases)
                            .build()
            );

            // Criar a camada de saída com ativação Softmax
            Variable<Float> outputWeights = ops.variable(ops.randomUniform(ops.constant(new long[]{hiddenSize, numCategories}), Float.class), Float.class);
            Variable<Float> outputBiases = ops.variable(ops.zeros(ops.constant(new long[]{numCategories}), Float.class), Float.class);
            Softmax<Float> softmaxLayer = Softmax.create(ops);
            Softmax<Float> softmaxOutput = softmaxLayer.create(ops.math.add(ops.math.matMul(lstmOutput.output(), outputWeights), outputBiases));

            // Calcular a perda usando a entropia cruzada categórica
            Tensor<Integer> labelsTensor = Tensor.create(trainingLabels, Integer.class);
            Tensor<Float> loss = ops.math.mean(ops.nn.sparseSoftmaxCrossEntropyWithLogits(softmaxOutput, labelsTensor).loss());

            // Criar o otimizador (Gradiente Descendente)
            float learningRate = 0.01f;
            Variable<Float> learningRateVar = ops.variable(ops.constant(learningRate), Float.class);
            ops.train.setLearningRate(learningRateVar);
            ops.train.getOptimizer(Train.GradientDescentBuilder.class).build().minimize(loss);

            // Inicializar as variáveis do grafo
            try (Session session = new Session(graph)) {
                session.runner().initializeVariables().run();

                // Treinamento do modelo
                Random random = new Random();
                for (int epoch = 0; epoch < numEpochs; epoch++) {
                    float totalLoss = 0;
                    for (int i = 0; i < trainingData.length; i++) {
                        // Obter uma notícia de treinamento aleatoriamente
                        int randomIndex = random.nextInt(trainingData.length);
                        float[] input = trainingData[randomIndex];
                        int label = trainingLabels[randomIndex];

                        // Treinar o modelo com uma única notícia
                        float[] inputTensorData = input;
                        int[] labelTensorData = {label};
                        try (Tensor<Float> inputTensor = Tensor.create(new long[]{batchSize, inputSize}, FloatBuffer.wrap(inputTensorData));
                             Tensor<Integer> labelTensor = Tensor.create(new long[]{batchSize}, IntBuffer.wrap(labelTensorData))) {

                            Tensor<?>[] output = session.runner()
                                    .feed(inputPlaceholder.asOutput(), inputTensor)
                                    .feed(labelPlaceholder.asOutput(), labelTensor)
                                    .fetch(loss)
                                    .run()
                                    .toArray();

                            float batchLoss = output[0].floatValue();
                            totalLoss += batchLoss;
                        }
                    }

                    float avgLoss = totalLoss / trainingData.length;
                    System.out.println("Epoch " + epoch + ", Loss: " + avgLoss);
                }
            }
        }
    }
}
