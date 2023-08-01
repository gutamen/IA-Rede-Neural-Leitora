/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.mycompany.ia;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Properties;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;


/**
 *
 * @author 
 */


public class IA {

private static final List<String> categories = List.of("wheather", "music", "politic", "economy", "tecnology");

    public static void main(String[] args) {
        
        

        // Documento de texto de entrada para classificação
        String inputText = "Insira aqui o texto da notícia que você deseja classificar.";
        int numCategories = 5;
        // Carregar o modelo pré-treinado (certifique-se de que o caminho esteja correto)
        String modelPath = "path/to/your/model";
        try (Graph graph = new Graph()) {
            byte[] modelBytes = TensorFlow.loadLibrary(modelPath);
            graph.importGraphDef(modelBytes);

            // Realizar pré-processamento do texto (por exemplo, tokenização e padding)
            float[] inputTensorData = preprocessText(inputText);

            // Crie uma sessão e execute o modelo para fazer a classificação
            try (Session session = new Session(graph);
                 Tensor<Float> inputTensor = Tensor.create(new long[]{1, inputTensorData.length}, FloatBuffer.wrap(inputTensorData));
                 Tensor outputTensor = session.runner()
                         .feed("input_tensor_name", inputTensor)
                         .fetch("output_tensor_name")
                         .run()
                         .get(0)) {

                // Obtenha o resultado da classificação
//                Float[][] predictions = outputTensor.copyTo(new Float[1][numCategories])[0];
                
                float[] predictions = new float[numCategories];
                outputTensor.copyTo(predictions);
                int predictedCategoryIndex = argmax(predictions);
                String predictedCategory = getCategoryName(predictedCategoryIndex);

                //Imprima a categoria prevista
                System.out.println("Categoria prevista: " + predictedCategory);
            }
        }    
    }

    private static float[] preprocessText(String text) {
        // Configurar o pipeline de NLP do CoreNLP
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // Processar o texto com o pipeline de NLP
        CoreDocument document = new CoreDocument(text);
        pipeline.annotate(document);

        // Obter a lista de tokens do texto
        List<CoreLabel> tokens = document.tokens();

        // Realizar outras etapas de pré-processamento, como remoção de stopwords e padding
        // Aqui, vamos apenas converter os tokens em um vetor numérico
        float[] inputTensorData = new float[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            // Represente cada token com um valor numérico simples (índice do token na lista de categorias)
            String token = tokens.get(i).originalText().toLowerCase();
            int tokenIndex = categories.indexOf(token);
            if (tokenIndex >= 0) {
                inputTensorData[i] = tokenIndex;
            } else {
                // Se o token não estiver em nenhuma categoria, atribua um valor padrão (por exemplo, -1)
                inputTensorData[i] = -1;
            }
        }

        return inputTensorData;
    }

    private static int argmax(float[] array) {
      int maxIndex = 0;
        float maxValue = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxIndex = i;
                maxValue = array[i];
            }
        }

        return maxIndex;
    }
//
    
    private static String getCategoryName(int categoryIndex) {
        // Mapeie o índice da categoria para o nome da categoria (clima, música, política, economia, tecnologia)
        String returnS = null;
        
        switch(categoryIndex){
            case(0):
                returnS = "weather";
            case(1):
                returnS = "music";
            case(2):
                returnS = "politic";
            case(3):
                returnS = "economy";
            case(4):
                returnS = "tecnology";
        }
        
        return returnS;
    }
    
   
}


