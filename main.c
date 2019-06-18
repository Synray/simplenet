#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>
#include <assert.h>
#include <string.h>

#include "rng.h"

/* Windows doesn't support C very well, so more changes to the code might be necessary to compile on Windows. */
/* For example, Windows apparently deprecated most of the stdio.h functions. */
#ifdef _WIN32
    #define restrict
    #include <Winsock2.h> //ntohl
#else
    #include <arpa/inet.h> //ntohl
#endif

/*************************************Math*************************************/

static inline double sigmoid(const double x)
{
    return 1.0 / (1.0 + exp(-x));
}

static inline double sigmoid_d(const double x)
{
    return x * (1.0 - x);
}

/***********************************Networks***********************************/

#ifdef _WIN32
    #define RANDNUM(_min,_max) (min + ((max - min) * rng32_random()/(UINT32_MAX));})
#else
    #define RANDNUM(_min,_max) ({const float min = (float)(_min), max = (float)(_max); min + ((max - min) * rng32_random()/(UINT32_MAX));})
#endif

#ifndef _WIN32
    #ifdef DEBUG_PRINT_NEURONS
        #define DBG_PRINT(...) printf(__VA_ARGS__)
    #else
        #define DBG_PRINT(...)
    #endif
#endif

struct TrainData
{
    int numData;
    uint32_t width, height;
    uint32_t numInputs;
    uint32_t numOutputs;
    double **restrict inputs;
    double **restrict outputs;
};

struct Neuron
{
    float bias;
    float sum;
    float value;
}
#ifdef __GNUC__
    __attribute__((packed))
#endif
;


struct Layer
{
    struct Neuron *restrict first;
    struct Neuron *restrict last;
    int32_t firstWeight;
};

struct Network
{
    /* Dimensions */
    int32_t numInputs;
    int32_t numOutputs;
    int32_t numNeurons;
    int32_t numWeights;

    /* Layers and neurons */
    struct Layer *restrict firstLayer;
    struct Layer *restrict lastLayer;
    struct Neuron *restrict neurons;

    /* Parameters */
    double *restrict weights;
    double *restrict outputs;
    double *restrict gradients;
    double *restrict biasGradients;
    double *restrict weightGradients;

    /* Hyperparameters */
    double learnRate;

    /* Evaluation variables */
    double MSE_value;
    uint32_t num_MSE;
    uint32_t correct;
};

struct Network *net_init(int nlayers, ...);
void net_configure(struct Network *const restrict net);
double *net_feedforward(struct Network *const restrict net, const double *const restrict inputs);
void net_compute_error(struct Network *restrict net, const double *restrict desired);
void net_backpropogate(struct Network *restrict net);
void net_update_weight_grads(struct Network *restrict net);
void net_update_params_batch(struct Network *restrict net, const uint32_t num_data);
void net_clear_gradients(struct Network *restrict net);
void net_test_batch(struct Network *restrict net, struct TrainData *const restrict test);
void net_epoch(struct Network *restrict net, struct TrainData *const restrict train, struct TrainData *const restrict test, const int numEpochs);

/* Initialize a Network from the list of layer sizes
 * The first layer is an input layer, and should match the size of the input data
 */
struct Network *net_init(int nlayers, ...)
{
    uint32_t layerSizes[nlayers];

    struct Network *restrict net = calloc(1, sizeof(struct Network));

    /* Allocate the layers array */
    net->firstLayer = calloc(nlayers, sizeof(struct Layer));
    net->lastLayer = net->firstLayer + nlayers;

    /* Store the sizes in the va_list in an array */
    va_list layer_sizes;
    va_start(layer_sizes, nlayers);
    for (int i = 0; i < nlayers; ++i)
    {
        uint32_t size = va_arg(layer_sizes, uint32_t);
        layerSizes[i] = size;
        net->numNeurons += size;
    }
    va_end(layer_sizes);

    /* Allocate one block for all the neurons at once */
    net->neurons = calloc(net->numNeurons, sizeof(struct Neuron));

    /* Give each Layer layerSize neurons out of the block */
    uint32_t neurons_so_far = 0;
    for (struct Layer *restrict l = net->firstLayer; l < net->lastLayer; ++l)
    {
        l->first = net->neurons + neurons_so_far;
        l->last = l->first + layerSizes[l - net->firstLayer];
        neurons_so_far += layerSizes[l - net->firstLayer];
    }
    net->numInputs = layerSizes[0];
    net->numOutputs = layerSizes[nlayers - 1];
    net->outputs = calloc(net->numOutputs, sizeof(double));

    /* Calculate the number of weights in the network */
    /* Start from the second layer (since the first layer is just the actual input) */
    for (struct Layer *restrict l = net->firstLayer + 1; l < net->lastLayer; ++l)
    {
        int layer = l - net->firstLayer;

        /* inputs = number of neurons in the previous layer */
        int num_in = layerSizes[layer - 1];
        l->firstWeight = net->numWeights;
        net->numWeights += num_in * layerSizes[layer];
    }

    /* Allocate memory for the network's parameters as well as their gradients */
    net->weights = calloc(net->numWeights, sizeof(double));
    net->gradients = calloc(net->numNeurons, sizeof(double));
    net->biasGradients = calloc(net->numNeurons, sizeof(double));
    net->weightGradients = calloc(net->numWeights, sizeof(double));

    /* Initialize the weights and biases with random values */
    for (struct Neuron *restrict n = net->neurons; n < net->neurons + net->numNeurons; ++n)
    {
        n->bias = RANDNUM(-0.2, 0.2);
    }

    for (int i = 0; i < net->numWeights; ++i)
    {
        net->weights[i] = RANDNUM(-0.2, 0.2);
    }

    net_configure(net);

    return net;
}

void net_configure(struct Network *const restrict net)
{
    net->learnRate = 0.75;
}

double *net_feedforward(struct Network *const restrict net, const double *const restrict inputs)
{
    struct Neuron *const restrict first_neuron = net->firstLayer->first;

    /* Setup the inputs */
    for (int i = 0; i < net->numInputs; ++i)
    {
        first_neuron[i].value = inputs[i];
    }

    for (struct Layer *restrict l = net->firstLayer + 1; l < net->lastLayer; ++l)
    {
        int previous_layer_size = (l - 1)->last - (l - 1)->first;
        DBG_PRINT("Layer %ld\n", l - net->firstLayer);
        DBG_PRINT("Num inputs: %d\n", previous_layer_size);
        for (struct Neuron *restrict n = l->first; n < l->last; ++n)
        {
            DBG_PRINT("    Neuron %ld\n", n - l->first);
            n->sum = 0;
            const struct Neuron *const restrict prev = (l - 1)->first;
            for (int i = 0; i < previous_layer_size; ++i)
            {
                /* Ugly stuff to get around net->weights being 1D (because a 2D array on the heap is slower) */
                const int weightIndex = l->firstWeight + (n - l->first) * previous_layer_size + i;
                DBG_PRINT("\t%ld\n", weightIndex);
                n->sum += net->weights[weightIndex] * prev[i].value;
            }
            /* Add the bias */
            n->sum += n->bias;
            n->value = sigmoid(n->sum);
        }
    }

    for (int i = 0; i < net->numOutputs; ++i)
    {
        net->outputs[i] = (net->lastLayer - 1)->first[i].value;
    }
    return net->outputs;
}

void net_clear_gradients(struct Network *restrict net)
{
    memset(net->weightGradients, 0, net->numWeights * sizeof(double));
    memset(net->biasGradients, 0, net->numNeurons * sizeof(double));
}

void net_compute_error(struct Network *restrict net, const double *restrict desired)
{
    /* clear the gradients */
    memset(net->gradients, 0, net->numNeurons * sizeof(double));
    struct Neuron *restrict last_layer_begin = (net->lastLayer - 1)->first;

    double *restrict gradp = net->gradients + (last_layer_begin - net->firstLayer->first);
    for (struct Neuron *restrict n = last_layer_begin; n < (net->lastLayer - 1)->last; ++n)
    {
        const double error = *desired - n->value;
        *gradp = sigmoid_d(n->value) * error;

        net->biasGradients[n - last_layer_begin] += *gradp;

        desired++;
        gradp++;
    }
}

void net_backpropogate(struct Network *restrict net)
{
    struct Layer *const restrict lastLayer = net->lastLayer;
    struct Neuron *const restrict firstNeuron = net->firstLayer->first;

    for (struct Layer *restrict l = lastLayer - 2; l >= net->firstLayer; --l)
    {
        DBG_PRINT("Layer %ld\n", l - net->firstLayer);
        const double *const restrict prevGrad = net->gradients + ((l + 1)->first - firstNeuron);
        for (struct Neuron *restrict n = l->first; n < l->last; ++n)
        {
            DBG_PRINT("    Neuron %ld\n", n - l->first);
            const int numInputs = l->last - l->first;
            const int numOutputs = (l + 1)->last - (l + 1)->first;
            for (int i = 0; i < numOutputs; ++i)
            {
                const int weightIndex = (l + 1)->firstWeight + (n - l->first) + i * numInputs;
                DBG_PRINT("\t%d\n", weightIndex);
                net->gradients[n - firstNeuron] += net->weights[weightIndex] * prevGrad[i];
            }
            net->gradients[n - firstNeuron] *= sigmoid_d(n->value);
            net->biasGradients[n - firstNeuron] += net->gradients[n - firstNeuron];
        }
    }
}

void net_update_weight_grads(struct Network *restrict net)
{
    struct Neuron *const restrict firstNeuron = net->firstLayer->first;
    for (struct Layer *restrict l = net->firstLayer + 1; l < net->lastLayer; ++l)
    {
        const int num_inputs = (l - 1)->last - (l - 1)->first;
        DBG_PRINT("Layer %ld\n", l - net->firstLayer);
        /* DBG_PRINT("Num inputs: %d\n", num_inputs); */
        for (struct Neuron *restrict n = l->first; n < l->last; ++n)
        {
            DBG_PRINT("    Neuron %ld\n", n - l->first);
            const struct Neuron *const restrict prev = (l - 1)->first;
            for (int i = 0; i < num_inputs; ++i)
            {
                const int weightIndex = l->firstWeight + (n - l->first) * num_inputs + i;
                DBG_PRINT("\t%d\n", weightIndex);
                net->weightGradients[weightIndex] += prev[i].value * net->gradients[n - firstNeuron];
            }
        }
    }
}

void net_update_params_batch(struct Network *restrict net, const uint32_t batchSize)
{
    const double epsilon = net->learnRate / batchSize;

    for (uint32_t i = 0; i < net->numWeights; ++i)
    {
        net->weights[i] += epsilon * net->weightGradients[i];
    }

    for (int i = net->numInputs; i < net->numNeurons; ++i)
    {
        net->neurons[i].bias += epsilon * net->biasGradients[i];
    }
}

void net_test_batch(struct Network *restrict net, struct TrainData *const restrict test)
{
    net->MSE_value = net->num_MSE = 0;
    net->correct = 0;
    for (int i = 0; i < test->numData; ++i)
    {
        const double *const restrict outputs = net_feedforward(net, test->inputs[i]);

        int max = 0;
        for (int o = 1; o < net->numOutputs; ++o)
        {
            if (outputs[o] > outputs[max]) { max = o; }
            double error = test->outputs[i][o] - outputs[o];
            net->MSE_value += error * error;
            net->num_MSE++;
        }
        if (test->outputs[i][max] == 1.0) { net->correct++; }
    }
    printf("Num correct: %d / %d -- %.2f%%\n", net->correct, test->numData, 100.0 * (double)net->correct / (double)test->numData);
    printf("Test error: %.2f%%\n", 100.0 * net->MSE_value / (double)net->num_MSE);
}

void net_epoch(struct Network *restrict net, struct TrainData *const restrict train, struct TrainData *const restrict test, const int numEpochs)
{
    double **restrict batchIn = malloc(train->numData * sizeof(double *));
    double **restrict batchOut = malloc(train->numData * sizeof(double *));
    memcpy(batchIn, train->inputs, sizeof(double *) * train->numData);
    memcpy(batchOut, train->outputs, sizeof(double *) * train->numData);

    int batchSize = 10;

    for (int epoch = 1; epoch <= numEpochs; ++epoch)
    {
        /* shuffle the batches */
        for (int i = train->numData - 1; i > 0; --i)
        {
            int n = rng32_boundedrand(i + 1);
            double *restrict temp = batchIn[i];
            batchIn[i] = batchIn[n];
            batchIn[n] = temp;

            temp = batchOut[i];
            batchOut[i] = batchOut[n];
            batchOut[n] = temp;
        }


        for (int i = 0; i < train->numData / batchSize; ++i)
        {
            for (int j = 0; j < batchSize; ++j)
            {
                (void)net_feedforward(net, batchIn[i * batchSize + j]);
                net_compute_error(net, batchOut[i * batchSize + j]);
                net_backpropogate(net);
                net_update_weight_grads(net);
            }
            net_update_params_batch(net, batchSize);
            net_clear_gradients(net);
        }

        printf("Epoch %d: ", epoch);
        net_test_batch(net, test);

        /* Uncomment to print the training data error */
        /* puts("\tTrain errors:"); */
        /* net_test_batch(net, train); */

        /* Stop if the network passed. (Unlikely for a fully-connected network) */
        if (net->correct == test->numData)
        {
            printf("Passed the test\n");
            return;
        }
    }

    free(batchOut);
    free(batchIn);
}

/* MNIST Data Loader */

/* XXX: Allocated memory is not freed in fail cases */
/* Reads an mnist image file and a label file, and returns the data in a TrainData struct */
struct TrainData *mnist_read(const char *restrict imageName, const char *restrict labelName)
{
    FILE *restrict fp = fopen(imageName, "r");

    if (!fp)
    {
        perror("Can't open image file");
        return NULL;
    }

    uint32_t magicNumber = 0;
    fread(&magicNumber, sizeof magicNumber, 1, fp);

    magicNumber = ntohl(magicNumber);

    printf("Magic No: %08X (%u)\n", magicNumber, magicNumber);

    if (magicNumber >> 8 != 0x08)
    {
        printf("Wrong magic number\n");
        return NULL;
    }

    uint8_t numDimensions = magicNumber & 0xFF;
    if (numDimensions != 3)
    {
        printf("Wrong image file dimensions: %d dimensions\n", numDimensions);
        return NULL;
    }

    struct TrainData *restrict data = malloc(sizeof * data);
    uint32_t width, height;
    fread(&data->numData, sizeof data->numData, 1, fp);
    fread(&width, sizeof width, 1, fp);
    fread(&height, sizeof height, 1, fp);

    //Convert the dimensions to the CPU's number format
    width = ntohl(width);
    height = ntohl(height);
    data->numData = ntohl(data->numData);
    data->numInputs = width * height;

    printf("Images: %d\n", data->numData);
    printf("Width: %d, Height: %d\n", width, height);

    /* Allocate space for the pointers */
    data->inputs = calloc(data->numData, sizeof(double *));

    /* Allocate one array for all the image data */
    uint8_t *restrict imageData = calloc(data->numData * width * height, sizeof(uint8_t));
    double *restrict inputData = calloc(data->numData * width * height, sizeof(double));
    fread(imageData, width * height, data->numData, fp);
    fclose(fp);

    for (int i = 0; i < data->numData * width * height; ++i)
    {
        inputData[i] = (imageData[i]) / 255.0;
    }
    free(imageData);
    imageData = NULL;

    /*  Point each input pointer to an image in inputData */
    for (int i = 0; i < data->numData; ++i)
    {
        data->inputs[i] = inputData + i * width * height;
    }


    /* Read the labels from the label file */
    fp = fopen(labelName, "r");

    if (!fp)
    {
        perror("Can't open label file");
        return NULL;
    }

    fread(&magicNumber, sizeof magicNumber, 1, fp);
    magicNumber = ntohl(magicNumber);
    if (magicNumber >> 8 != 0x08)
    {
        puts("Wrong magic number.");
        return NULL;
    }

    numDimensions = magicNumber & 0xFF;
    if (numDimensions != 1)
    {
        printf("Wrong label file dimensions: %d dimensions\n", numDimensions);
        return NULL;
    }

    int numLabels = 0;
    fread(&numLabels, sizeof numLabels, 1, fp);
    numLabels = ntohl(numLabels);
    printf("Labels: %d\n", numLabels);
    if (numLabels != data->numData)
    {
        printf("Number of labels does not match number of images: %d labels vs %d images\n", numLabels, data->numData);
        return NULL;
    }
    data->outputs = calloc(numLabels * 10, sizeof(double *));
    uint8_t *restrict labelData = calloc(numLabels, sizeof(uint8_t));
    double *restrict outputData = calloc(numLabels * 10, sizeof(double));

    fread(labelData, sizeof(uint8_t), numLabels, fp);
    fclose(fp);

    /* Set the label output in outputData to 1.0 */
    for (int i = 0; i < numLabels; ++i)
    {
        outputData[i * 10 + labelData[i]] = 1.0;
        /* Point each output to a label */
        data->outputs[i] = outputData + i * 10;
    }
    free(labelData);


    return data;
}

int main(int argc, char *argv[])
{
    rng32_srandom(time(0) ^ (intptr_t)printf, (intptr_t)net_init);

    struct TrainData *restrict mnist = mnist_read("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    struct TrainData *restrict mnist_test = mnist_read("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    struct Network *const restrict net = net_init(3, mnist->numInputs, 30, 10);

    puts("\nBefore training:");
    net_test_batch(net, mnist_test);
    puts("");

    net_epoch(net, mnist, mnist_test, 10000);

    printf("\nOutputs after training:\n");
    net_test_batch(net, mnist_test);

    /* No cleanup is performed */
}
