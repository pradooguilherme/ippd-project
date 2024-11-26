#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// Estrutura para representar um neurônio
typedef struct
{
    float *pesos;
    float saida;
} Neuronio;

typedef struct
{
    float features[4];
    int label[3];
} IrisData;

IrisData *handleIris(const char *path, int *size)
{
    FILE *file = fopen(path, "r");
    if (!file)
        return NULL;

    IrisData *data = NULL;
    char line[256];
    *size = 0;

    while (fgets(line, sizeof(line), file))
    {
        data = realloc(data, (*size + 1) * sizeof(IrisData));
        if (!data)
        {
            printf("Erro ao alocar memória.\n");
            fclose(file);
            return NULL;
        }
        char *token = strtok(line, ",");
        for (int i = 0; i < 4; i++)
        {
            if (token != NULL)
            {
                data[*size].features[i] = atof(token);
                token = strtok(NULL, ",");
            }
            else
            {
                data[*size].features[i] = 0.0f;
            }
        }
        if (token)
        {
            if (strcmp(token, "Iris-setosa\n") == 0 || strcmp(token, "Iris-setosa") == 0)
            {
                data[*size].label[0] = 1;
                data[*size].label[1] = 0;
                data[*size].label[2] = 0;
            }
            else if (strcmp(token, "Iris-versicolor\n") == 0 || strcmp(token, "Iris-versicolor") == 0)
            {
                data[*size].label[0] = 0;
                data[*size].label[1] = 1;
                data[*size].label[2] = 0;
            }
            else if (strcmp(token, "Iris-virginica\n") == 0 || strcmp(token, "Iris-virginica") == 0)
            {
                data[*size].label[0] = 0;
                data[*size].label[1] = 0;
                data[*size].label[2] = 1;
            }
            else
            {
                data[*size].label[0] = 0;
                data[*size].label[1] = 0;
                data[*size].label[2] = 0;
            }
        }
        else
        {
            data[*size].label[0] = 0;
            data[*size].label[1] = 0;
            data[*size].label[2] = 0;
        }
        (*size)++;
    }

    fclose(file);
    return data;
}

// Função de shuffle usando o algoritmo Fisher-Yates
void shuffleIrisData(IrisData *array, int n)
{
    if (n > 1)
    {
        srand(time(NULL));
        for (int i = n - 1; i > 0; i--)
        {
            int j = rand() % (i + 1);
            IrisData temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}

float funcaoAtivacao1(float x)
{
    float r = 1.0f / (1.0f + expf(-x));
    return r;
}

float derivadaFuncAtivacao1(float y)
{
    float derivada = y * (1.0f - y);
    return derivada;
}

// Função perceptron
float *perceptron(Neuronio *camada, int numNeuronios, float *entradas, int numEntradas, float bias, int *tamanho)
{
    float *arrSomatorios = malloc(numNeuronios * sizeof(float));
    if (!arrSomatorios)
        return NULL;

    for (int neuronio = 0; neuronio < numNeuronios; neuronio++)
    {
        float somatorio = 0.0f;
        for (int entrada = 0; entrada < numEntradas; entrada++)
        {
            somatorio += entradas[entrada] * camada[neuronio].pesos[entrada];
        }
        somatorio += bias;
        // Aplicar função de ativação
        float r = funcaoAtivacao1(somatorio);
        camada[neuronio].saida = r;
        arrSomatorios[neuronio] = somatorio;
    }

    *tamanho = numNeuronios;
    return arrSomatorios;
}

// Função respostaCamada
float *respostaCamada(Neuronio *camada, int numNeuronios)
{
    float *resp = malloc(numNeuronios * sizeof(float));
    if (!resp)
        return NULL;
    for (int neuronio = 0; neuronio < numNeuronios; neuronio++)
    {
        resp[neuronio] = camada[neuronio].saida;
    }
    return resp;
}

// Função para calcular o gradiente de erro da saída
float *gradienteErroSaida(float *saidaRede, int *saidaEsperada, int tamanho)
{
    float *erro = malloc(tamanho * sizeof(float));
    if (!erro)
        return NULL;

    for (int i = 0; i < tamanho; i++)
    {
        float derivada = derivadaFuncAtivacao1(saidaRede[i]);
        float erroI = (saidaEsperada[i] - saidaRede[i]) * derivada;
        erro[i] = erroI;
    }
    return erro;
}

// Função para corrigir os pesos dos neurônios
void correcaoErro(Neuronio *camada, int numNeuronios, float aprendizado, float *erro, float *entradaNeuronio, int numEntradas)
{
    for (int neuronio = 0; neuronio < numNeuronios; neuronio++)
    {
        for (int conn = 0; conn < numEntradas; conn++)
        {
            camada[neuronio].pesos[conn] += aprendizado * erro[neuronio] * entradaNeuronio[conn];
        }
    }
}

// Função para corrigir o bias
// Atualização da função correcaoBias para aceitar o tamanho do erro
float correcaoBias(float bias, float *erro, int tamanhoErro, float aprendizado)
{
    float sumError = 0.0f;
    for (int i = 0; i < tamanhoErro; i++)
    {
        sumError += erro[i];
    }
    bias += aprendizado * sumError;
    return bias;
}

// Função para comparar arrays (predição vs esperado)
int compararArrays(int *a, int *b, int tamanho)
{
    for (int i = 0; i < tamanho; i++)
    {
        if (a[i] != b[i])
        {
            return 0;
        }
    }
    return 1;
}

// Função que determina o resultado da rede neural
int resultados(float *respCamadaSaida, int *esperado, int tamanho)
{
    // Encontrar o índice com o maior valor em respCamadaSaida
    int maxIndex = 0;
    for (int i = 1; i < tamanho; i++)
    {
        if (respCamadaSaida[i] > respCamadaSaida[maxIndex])
        {
            maxIndex = i;
        }
    }
    // Criar a array de predição
    int predicao[3] = {0, 0, 0};
    predicao[maxIndex] = 1;

    // Comparar predicao com esperado
    if (compararArrays(predicao, esperado, tamanho))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

// Função para calcular o gradiente de erro na camada oculta
float *gradienteErroOculto(Neuronio *camadaOculta, Neuronio *camadaSaida, float *erroSaida, int numNeuroniosOcultos, int numNeuroniosSaida)
{
    float *erroOculto = malloc(numNeuroniosOcultos * sizeof(float));
    if (!erroOculto)
        return NULL;

    for (int j = 0; j < numNeuroniosOcultos; j++)
    {
        float sum = 0.0f;
        for (int k = 0; k < numNeuroniosSaida; k++)
        {
            sum += erroSaida[k] * camadaSaida[k].pesos[j];
        }
        // Aplica a derivada da função de ativação
        erroOculto[j] = sum * derivadaFuncAtivacao1(camadaOculta[j].saida);
    }
    return erroOculto;
}

int main()
{
    struct timeval start, stop;

    // Inicialização dos pesos e neurônios de entrada
    Neuronio neuroniosEntrada[4];
    for (int i = 0; i < 4; i++)
    {
        neuroniosEntrada[i].pesos = malloc(sizeof(float));
        neuroniosEntrada[i].pesos[0] = 1.0f; // Inicialização exemplo
        neuroniosEntrada[i].saida = 0.0f;
    }

    // Inicialização dos neurônios ocultos
    int numNeuroniosOcultos = 15000; // Número menor para simplicidade
    Neuronio neuronioOculto[numNeuroniosOcultos];
    for (int i = 0; i < numNeuroniosOcultos; i++)
    {
        neuronioOculto[i].pesos = malloc(4 * sizeof(float));
        for (int j = 0; j < 4; j++)
        {
            neuronioOculto[i].pesos[j] = ((float)rand() / (float)(RAND_MAX)) * 0.1f;
        }
        neuronioOculto[i].saida = 0.0f;
    }

    // Inicialização dos neurônios de saída
    int numNeuroniosSaida = 3;
    Neuronio neuroniosSaida[numNeuroniosSaida];
    for (int i = 0; i < numNeuroniosSaida; i++)
    {
        neuroniosSaida[i].pesos = malloc(numNeuroniosOcultos * sizeof(float));
        for (int j = 0; j < numNeuroniosOcultos; j++)
        {
            neuroniosSaida[i].pesos[j] = ((float)rand() / (float)(RAND_MAX)) * 0.1f;
        }
        neuroniosSaida[i].saida = 0.0f;
    }

    // Biases
    float biasEntrada = -0.3f;
    float biasOculto = -0.3f;
    float biasSaida = -0.3f;

    // Taxa de aprendizado
    float aprendizado = 0.283f;

    // Carregar os dados
    int size;
    IrisData *data = handleIris("iris.data", &size);

    // Embaralha os dados
    shuffleIrisData(data, size);

    // Divide os dados em conjuntos de treinamento e teste
    int trainSize = (int)(0.8 * size);
    int testSize = size - trainSize;

    IrisData *trainingData = malloc(trainSize * sizeof(IrisData));
    IrisData *testData = malloc(testSize * sizeof(IrisData));

    memcpy(trainingData, data, trainSize * sizeof(IrisData));
    memcpy(testData, data + trainSize, testSize * sizeof(IrisData));

    gettimeofday(&start, NULL);
    // Treinamento
    int epocas = 500;
    for (int e = 0; e < epocas; e++)
    {
        int acertos = 0;
        for (int i = 0; i < trainSize; i++)
        {
            // Forward pass
            float entradas[4];
            for (int j = 0; j < 4; j++)
            {
                entradas[j] = trainingData[i].features[j];
            }

            // Camada Oculta
            int tamanhoOculto = 0;
            float *somatorioOculto = perceptron(neuronioOculto, numNeuroniosOcultos, entradas, 4, biasOculto, &tamanhoOculto);
            float *respOculto = respostaCamada(neuronioOculto, numNeuroniosOcultos);

            // Camada Saída
            int tamanhoSaida = 0;
            float *somatorioSaida = perceptron(neuroniosSaida, numNeuroniosSaida, respOculto, numNeuroniosOcultos, biasSaida, &tamanhoSaida);
            float *respSaida = respostaCamada(neuroniosSaida, numNeuroniosSaida);

            // Cálculo do erro
            float *erroSaida = gradienteErroSaida(respSaida, trainingData[i].label, numNeuroniosSaida);
            float *erroOculto = gradienteErroOculto(neuronioOculto, neuroniosSaida, erroSaida, numNeuroniosOcultos, numNeuroniosSaida);


            // Atualização dos pesos da camada de saída
            correcaoErro(neuroniosSaida, numNeuroniosSaida, aprendizado, erroSaida, respOculto, numNeuroniosOcultos);

            // Atualização dos pesos da camada oculta
            correcaoErro(neuronioOculto, numNeuroniosOcultos, aprendizado, erroOculto, entradas, 4);

            // Atualização dos biases
            // Ajustes na chamada da função correcaoBias no main
            biasSaida = correcaoBias(biasSaida, erroSaida, numNeuroniosSaida, aprendizado);
            biasOculto = correcaoBias(biasOculto, erroOculto, numNeuroniosOcultos, aprendizado);

            // Verificar se acertou
            int acertou = resultados(respSaida, trainingData[i].label, numNeuroniosSaida);
            acertos += acertou;

            // Liberar memória
            free(somatorioOculto);
            free(respOculto);
            free(somatorioSaida);
            free(respSaida);
            free(erroSaida);
            free(erroOculto);
        }
        printf("Época %d: Acertos = %d de %d\n", e + 1, acertos, trainSize);
    }
    gettimeofday(&stop, NULL);
    printf("Tempo de treinamento: %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

    gettimeofday(&start, NULL);
    // Teste
    int acertosTeste = 0;
    for (int i = 0; i < testSize; i++)
    {
        // Forward pass
        float entradas[4];
        for (int j = 0; j < 4; j++)
        {
            entradas[j] = testData[i].features[j];
        }

        // Camada Oculta
        int tamanhoOculto = 0;
        float *somatorioOculto = perceptron(neuronioOculto, numNeuroniosOcultos, entradas, 4, biasOculto, &tamanhoOculto);
        float *respOculto = respostaCamada(neuronioOculto, numNeuroniosOcultos);

        // Camada Saída
        int tamanhoSaida = 0;
        float *somatorioSaida = perceptron(neuroniosSaida, numNeuroniosSaida, respOculto, numNeuroniosOcultos, biasSaida, &tamanhoSaida);
        float *respSaida = respostaCamada(neuroniosSaida, numNeuroniosSaida);

        // Verificar se acertou
        int acertou = resultados(respSaida, testData[i].label, numNeuroniosSaida);
        acertosTeste += acertou;

        // Liberar memória
        free(somatorioOculto);
        free(respOculto);
        free(somatorioSaida);
        free(respSaida);
    }
    gettimeofday(&stop, NULL);
    printf("Tempo de teste: %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
    printf("Acertos no teste: %d de %d\n", acertosTeste, testSize);

    // Liberação de memória
    for (int i = 0; i < 4; i++)
    {
        free(neuroniosEntrada[i].pesos);
    }
    for (int i = 0; i < numNeuroniosOcultos; i++)
    {
        free(neuronioOculto[i].pesos);
    }
    for (int i = 0; i < numNeuroniosSaida; i++)
    {
        free(neuroniosSaida[i].pesos);
    }

    free(trainingData);
    free(testData);
    return 0;
}