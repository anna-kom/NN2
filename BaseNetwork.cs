using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AForge.WindowsForms           
{
    public enum NumberType : byte { Zero = 0, One, Two, Three, Four, Five, Six, Seven, Eight, Nine, Undef };
    /// <summary>
    /// Базовый класс для реализации как самодельного персептрона, так и обёртки для ActivationNetwork из Accord.Net
    /// </summary>
    abstract public class BaseNetwork
    {
        public abstract void InitializeNetwork(int[] structure, double initialLearningRate = 0.1);

        public abstract int Train(Sample sample, double acceptable_erorr, bool parallel = true);

        public abstract double TrainOnDataSet(SamplesSet samplesSet, int epochs_count, double acceptable_erorr, bool parallel = true);

        public abstract NumberType Predict(Sample sample);

        public abstract double TestOnDataSet(SamplesSet testSet);

        public abstract double[] GetOutput();
    }
}
