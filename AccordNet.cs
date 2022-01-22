using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AForge;
using Accord.Neuro;
using Accord.Neuro.Learning;
using System.IO;

namespace AForge.WindowsForms
{
    class AccordNet : BaseNetwork
    {
        /// <summary>
        /// Реализация нейронной сети из Accord.NET
        /// </summary>
        private ActivationNetwork network = null;
        
        /// <summary>
        /// Значение ошибки при обучении единичному образцу. При обучении на наборе образов не используется
        /// </summary>
        public double desiredErrorValue = 0.0005;

        //  Секундомер спортивный, завода «Агат», измеряет время пробегания стометровки, ну и время затраченное на обучение тоже умеет
        public System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch();

        /// <summary>
        /// Конструктор сети с указанием структуры (количество слоёв и нейронов в них)
        /// </summary>
        /// <param name="structure"></param>
        public AccordNet(int[] structure)
        {
            InitializeNetwork(structure);
        }
        
        /// <summary>
        /// Реиницализация сети - создаём заново сеть с указанной структурой
        /// </summary>
        /// <param name="structure">Массив с указанием нейронов на каждом слое (включая сенсорный)</param>
        /// <param name="initialLearningRate">Начальная скорость обучения</param>
        public override void InitializeNetwork(int[] structure, double initialLearningRate = 0.05)
        {
            // Создаём сеть - вроде того
            network = new ActivationNetwork((IActivationFunction)new SigmoidFunction(2.0), structure[0],
                structure.Skip(1).ToArray<int>());

            //  Встряска "мозгов" сети - рандомизируем веса связей
            new NguyenWidrow(network).Randomize();
        }
        
        /// <summary>
        /// Обучение сети одному образу
        /// </summary>
        /// <param name="sample"></param>
        /// <returns>Количество итераций для достижения заданного уровня ошибки</returns>
        public override int Train(Sample sample, double acceptable_error, bool parallel = true)
        {
            //  Создаём "обучателя" - либо параллельного, либо последовательного  
            ISupervisedLearning teacher;
            if (parallel)
                teacher = new ParallelResilientBackpropagationLearning(network);
            else
                teacher = new ResilientBackpropagationLearning(network);

            int iters = 1;
            while (teacher.Run(sample.input, sample.Output) > desiredErrorValue) { ++iters; }
            return iters;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochs_count, double acceptable_error, bool parallel = true)
        {
            //  Сначала надо сконструировать массивы входов и выходов
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];
            
            //  Теперь массивы из samplesSet группируем в inputs и outputs
            for(int i = 0; i < samplesSet.Count; ++i)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].Output;
            }

            //  Текущий счётчик эпох
            int epoch_to_run = 0;

            //  Создаём "обучателя" - либо параллельного, либо последовательного  
            ISupervisedLearning teacher;
            if (parallel)
                teacher = new ParallelResilientBackpropagationLearning(network);
            else
                teacher = new ResilientBackpropagationLearning(network);

            double error = double.PositiveInfinity;
            
            #if DEBUG
            StreamWriter errorsFile = File.CreateText("errors.csv");
            #endif

            stopWatch.Restart();

            while (epoch_to_run < epochs_count && error > acceptable_error)
            {
                epoch_to_run++;
                error = teacher.RunEpoch(inputs, outputs);
                #if DEBUG
                errorsFile.WriteLine(error);
                #endif
                //updateDelegate((epoch_to_run * 1.0) / epochs_count, error, stopWatch.Elapsed);
            }

            #if DEBUG
            errorsFile.Close();
            #endif

            //updateDelegate(1.0, error, stopWatch.Elapsed);

            stopWatch.Stop();

            return error;
        }

        public override NumberType Predict(Sample sample)
        {
            sample.Output = network.Compute(sample.input);
            sample.ProcessPrediction();
            return sample.recognizedClass;
        }

        public override double TestOnDataSet(SamplesSet testSet)
        {
            double correct = 0.0;
            for (int i = 0; i < testSet.Count; ++i)
            {
                testSet[i].Output = network.Compute(testSet[i].input);
                testSet[i].ProcessPrediction();
                if (testSet[i].actualClass == testSet[i].recognizedClass) correct += 1;
            }
            return correct/testSet.Count;
        }

        public override double[] GetOutput()
        {
            return network.Output;
        }
    }

}
