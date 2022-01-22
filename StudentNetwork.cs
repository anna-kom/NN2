using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;

namespace AForge.WindowsForms
{
    public class Neuron
    {
        public double output = 0;
        public double error = 0;
        public double bias = -1;
        public double biasWeight = 0.01;

        public int countInput = 0;
        public double[] weights;
        public Neuron[] inputLayer;
        public static Random rnd = new Random();

        private void InitializeWeights(int cnt, double min, double max)
        {
            weights = new double[cnt];
            for (int i = 0; i < cnt; ++i)
                weights[i] = min + rnd.NextDouble() * (max - min);
        }

        public Neuron(Neuron[] previous)
        {
            if (previous == null)
                return;
            inputLayer = previous;
            countInput = inputLayer.Length;
            InitializeWeights(countInput, -1, 1);
        }

        public Neuron() { }

        public void Activate()
        {
            double m = 0;
            for (int i = 0; i < countInput; i++)
                m += inputLayer[i].output * weights[i];
            output = ActivationFunction(bias * biasWeight + m);
        }

        public static double ActivationFunction(double x)
        {
            return 1.0 / (1.0 + System.Math.Exp(-x));
        }


        private void OptimizeWeights(double alpha, int from, int to)
        {
            for (int i = from; i < to; ++i)
                weights[i] += alpha * inputLayer[i].output * error;
        }


        public void StartBackPropagation(double alpha)
        {
            error *= output * (1 - output);
            biasWeight += alpha * bias * error;
        }

        public void BackPropagation(double alpha)
        {
            StartBackPropagation(alpha);

            for (int i = 0; i < countInput; ++i)
                inputLayer[i].error += error * weights[i];

            OptimizeWeights(alpha, 0, countInput);
        }

        public void BackPropagationParallel(double alpha, int from, int to)
        {
            for (int i = from; i < to; ++i)
                inputLayer[i].error += error * weights[i];
            OptimizeWeights(alpha, from, to);
        }
    }

    public class StudentNetwork : BaseNetwork
    {
        public double alpha = 0.1;  // скорость обучения

        private Neuron[][] layers;

        private int countLayers;     // количество слоев
        private int countSensors;    // количетсво сенсоров
        private int countClasses;    // количетсво классов

        public StudentNetwork(int[] structure)
        {
            countLayers = structure.Length;
            countClasses = structure.Last();
            countSensors = structure[0];
            InitializeNetwork(structure);
        }

        public override void InitializeNetwork(int[] structure, double initialLearningRate = 0.1)
        {
            alpha = initialLearningRate;
            layers = new Neuron[countLayers][];

            // Слой входных нейронов
            layers[0] = new Neuron[countSensors];
            for (int j = 0; j < countSensors; ++j)
                layers[0][j] = new Neuron();

            // Остальные слои
            for (int i = 1; i < countLayers; ++i)
            {
                layers[i] = new Neuron[structure[i]];
                for (int j = 0; j < structure[i]; ++j)
                    layers[i][j] = new Neuron(layers[i - 1]);
            }
        }


        // Запускаем параллельное или непараллельное обучение
        private void StartRun(Sample sample, bool parallel = true)
        {
            if (parallel)
                RunParallel(sample);
            else
                Run(sample);
        }

        // Непараллельное обучение
        private void Run(Sample image)
        {
            // Заполняем слой входных нейронов
            for (int i = 0; i < image.input.Length; ++i)
                layers[0][i].output = image.input[i];

            // Активация нейронов (кроме входных)
            for (int i = 1; i < countLayers; ++i)
                for (int j = 0; j < layers[i].Length; ++j)
                    layers[i][j].Activate();

            // Заполняем выходной вектор
            for (int i = 0; i < countClasses; ++i)
                image.Output[i] = layers[countLayers - 1][i].output;

            image.ProcessPrediction();
        }

        // Параллельное обучение
        private void RunParallel(Sample image)
        {
            // Заполняем слой входных нейронов
            for (int i = 0; i < image.input.Length; ++i)
                layers[0][i].output = image.input[i];

            // Параллельная активация
            for (int i = 1; i < countLayers; ++i)
            {
                Parallel.For(0, layers[i].Length, j =>
                {
                    layers[i][j].Activate();
                });
            }

            // Заполняем выходной вектор
            for (int i = 0; i < countClasses; ++i)
                image.Output[i] = layers[countLayers - 1][i].output;

            image.ProcessPrediction();
        }


        // Запускаем параллельное или непараллельное обратное распространение ошибки
        private void StartBackPropagation(Sample image, bool parallel = true)
        {
            if (parallel)
                BackPropagationParallel(image);
            else
                BackPropagation(image);
        }

        private void BackPropagation(Sample image)
        {
            // Заполняем ошибку для выходного слоя
            for (int i = 0; i < countClasses; i++)
                layers[layers.Length - 1][i].error = image.error[i];

            // Обратное распространение ошибки
            for (int i = countLayers - 1; i >= 0; --i)
                for (int j = 0; j < layers[i].Length; ++j)
                    layers[i][j].BackPropagation(alpha);
        }

        private void BackPropagationParallel(Sample image)
        {
            // Количество потоков
            int countThreads = 16;

            // Заполняем ошибку для выходного слоя
            for (int i = 0; i < countClasses; i++)
                layers[countLayers - 1][i].error = image.error[i];

            for (int layer = countLayers - 1; layer > 0; --layer)
            {
                int len = layers[layer - 1].Length / countThreads;

                // Обратное распространение ошибки
                for (int j = 0; j < layers[layer].Length; ++j)
                    layers[layer][j].StartBackPropagation(alpha);
                Parallel.For(0, countThreads, i =>
                {
                    for (int j = 0; j < layers[layer].Length; ++j)
                        layers[layer][j].BackPropagationParallel(alpha, len * i, i == countThreads ? layers[layer - 1].Length : len * (i + 1));
                });

            }
        }


        public override NumberType Predict(Sample sample)
        {
            Run(sample);
            return sample.recognizedClass;
        }

        public override int Train(Sample sample, double acceptableError, bool parallel = true)
        {
            int countIters = 0;

            while (true)
            {
                countIters++;
                StartRun(sample, parallel);

                if (sample.Correct() && sample.EstimatedError() < 0.2)
                    return countIters;

                StartBackPropagation(sample, parallel);

            }
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            double countCorrect = 0;
            samplesSet.Randomize();

            for (int epoch = 0; epoch < epochsCount; ++epoch)
            {
                countCorrect = 0;
                for (int i = 0; i < samplesSet.samples.Count; ++i)
                {
                    if (Train(samplesSet.samples[i], acceptableError, parallel) <= 1)
                        countCorrect++;
                }

                if (1 - countCorrect / samplesSet.samples.Count <= acceptableError)
                    break;
            }

            return ((double)countCorrect / samplesSet.samples.Count) * 100.0;
        }


        public override double TestOnDataSet(SamplesSet testSet)
        {
            int countCorrect = 0;
            for (int i = 0; i < testSet.Count; ++i)
            {
                Sample s = testSet.samples[i];
                Predict(s);
                if (s.Correct())
                    countCorrect++;
            }
            return (double)countCorrect / testSet.Count;
        }

        public override double[] GetOutput()
        {
            return layers.Last().Select(n => n.output).ToArray();
        }
    }
}
