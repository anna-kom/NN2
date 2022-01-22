using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Windows.Forms;
using System.Collections;


using System.Threading;

namespace AForge.WindowsForms
{
    /// <summary>
    /// Класс для хранения образа – входной массив сигналов на сенсорах, выходные сигналы сети, и прочее
    /// </summary>
    public class Sample
    {
        /// <summary>
        /// Входной вектор
        /// </summary>
        public double[] input = null;

        /// <summary>
        /// Выходной вектор, задаётся извне как результат распознавания
        /// </summary>
        public double[] Output = null;

        /// <summary>
        /// Вектор ошибки, вычисляется по какой-нибудь хитрой формуле
        /// </summary>
        public double[] error = null;

        /// <summary>
        /// Действительный класс образа. Указывается учителем
        /// </summary>
        public NumberType actualClass;

        /// <summary>
        /// Распознанный класс - определяется после обработки
        /// </summary>
        public NumberType recognizedClass;

        /// <summary>
        /// Конструктор образа - на основе входных данных для сенсоров, при этом можно указать класс образа, или не указывать
        /// </summary>
        /// <param name="inputValues"></param>
        /// <param name="sampleClass"></param>
        public Sample(double[] inputValues, int classesCount, NumberType sampleClass = NumberType.Undef)
        {
            //  Клонируем массивчик
            input = (double[])inputValues.Clone();
            Output = new double[classesCount];
            if (sampleClass != NumberType.Undef) Output[(int)sampleClass] = 1;


            recognizedClass = NumberType.Undef;
            actualClass = sampleClass;
        }

        /// <summary>
        /// Обработка реакции сети на данный образ на основе вектора выходов сети
        /// </summary>
        public void ProcessPrediction()
        {
            if (error == null)
                error = new double[Output.Length];

            //  Нам так-то выход не нужен, нужна ошибка и определённый класс
            recognizedClass = 0;
            for (int i = 0; i < Output.Length; ++i)
            {
                error[i] = ((i == (int)actualClass ? 1 : 0) - Output[i]);
                if (Output[i] > Output[(int)recognizedClass]) recognizedClass = (NumberType)i;
            }
        }

        /// <summary>
        /// Вычисленная суммарная квадратичная ошибка сети. Предполагается, что целевые выходы - 1 для верного, и 0 для остальных
        /// </summary>
        /// <returns></returns>
        public double EstimatedError()
        {
            double Result = 0;
            for (int i = 0; i < Output.Length; ++i)
                Result += System.Math.Pow(error[i], 2);
            return Result;
        }

        /// <summary>
        /// Добавляет к аргументу ошибку, соответствующую данному образу (не квадратичную!!!)
        /// </summary>
        /// <param name="errorVector"></param>
        /// <returns></returns>
        public void updateErrorVector(double[] errorVector)
        {
            for (int i = 0; i < errorVector.Length; ++i)
                errorVector[i] += error[i];
        }

        /// <summary>
        /// Представление в виде строки
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string result = "Sample decoding : " + actualClass.ToString() + "(" + ((int)actualClass).ToString() + "); " + Environment.NewLine + "Input : ";
            for (int i = 0; i < input.Length; ++i) result += input[i].ToString() + "; ";
            result += Environment.NewLine + "Output : ";
            if (Output == null) result += "null;";
            else
                for (int i = 0; i < Output.Length; ++i) result += Output[i].ToString() + "; ";
            result += Environment.NewLine + "Error : ";

            if (error == null) result += "null;";
            else
                for (int i = 0; i < error.Length; ++i) result += error[i].ToString() + "; ";
            result += Environment.NewLine + "Recognized : " + recognizedClass.ToString() + "(" + ((int)recognizedClass).ToString() + "); " + Environment.NewLine;


            return result;
        }

        /// <summary>
        /// Правильно ли распознан образ
        /// </summary>
        /// <returns></returns>
        public bool Correct() { return actualClass == recognizedClass; }
    }

    /// <summary>
    /// Выборка образов. Могут быть как классифицированные (обучающая, тестовая выборки), так и не классифицированные (обработка)
    /// </summary>
    public class SamplesSet : IEnumerable
    {
        /// <summary>
        /// Накопленные обучающие образы
        /// </summary>
        public List<Sample> samples = new List<Sample>();

        private Random rnd = new Random();

        /// <summary>
        /// Добавление образа к коллекции
        /// </summary>
        /// <param name="image"></param>
        public void AddSample(Sample image)
        {
            samples.Add(image);
        }
        public int Count { get { return samples.Count; } }

        public IEnumerator GetEnumerator()
        {
            return samples.GetEnumerator();
        }

        public void Randomize()
        {
            for (int i = 0; i < samples.Count; i++)
            {
                var temp = samples[i];
                var rand = rnd.Next(0, samples.Count);
                samples[i] = samples[rand];
                samples[rand] = temp;
            }
        }

        /// <summary>
        /// Реализация доступа по индексу
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public Sample this[int i]
        {
            get { return samples[i]; }
            set { samples[i] = value; }
        }

        public double ErrorsCount()
        {
            double correct = 0;
            double wrong = 0;
            foreach (var sample in samples)
                if (sample.Correct()) ++correct; else ++wrong;
            return correct / (correct + wrong);
        }
    }
}

