//Mariano-Gutiérrez
//Carne 1535719
using System;
namespace BackPropagationXor
{
    class principal
    {
        static void Main(string[] args)
        {
            train();
        }

        class modificar
        {
            public static double prod(double x)
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }

            public static double der(double x)
            {
                return x * (1 - x);
            }
        }

        class Neurona
        {
            public double[] inputs = new double[2];
            public double[] pesos = new double[2];
            public double error;

            private double biasWeight;

            private Random r = new Random();

            public double prod
            {
                get { return modificar.prod(pesos[0] * inputs[0] + pesos[1] * inputs[1] + biasWeight); }
            }

            public void peso_random()
            {
                pesos[0] = r.NextDouble();
                pesos[1] = r.NextDouble();
                biasWeight = r.NextDouble();
            }

            public void ajuste_pesos()
            {
                pesos[0] += error * inputs[0];
                pesos[1] += error * inputs[1];
                biasWeight += error;
            }
        }

        private static void train()
        {
            
            double[,] inputs =
            {
                { 0, 0},
                { 0, 1},
                { 1, 0},
                { 1, 1}
            };

          
            double[] resultados = { 0, 1, 1, 0 };

           
            Neurona neurona1 = new Neurona();
            Neurona neurona2 = new Neurona();
            Neurona pNeurona = new Neurona();
            neurona1.peso_random();
            neurona2.peso_random();
            pNeurona.peso_random();

            int epoca = 0;

        Retry:
            epoca++;
            for (int i = 0; i < 4; i++) 
            {
                neurona1.inputs = new double[] { inputs[i, 0], inputs[i, 1] };
                neurona2.inputs = new double[] { inputs[i, 0], inputs[i, 1] };
                pNeurona.inputs = new double[] { neurona1.prod, neurona2.prod };
                Console.WriteLine("{0} xor {1} = {2}", inputs[i, 0], inputs[i, 1], pNeurona.prod);
                pNeurona.error = modificar.der(pNeurona.prod) * (resultados[i] - pNeurona.prod);
                pNeurona.ajuste_pesos();
                neurona1.error = modificar.der(neurona1.prod) * pNeurona.error * pNeurona.pesos[0];
                neurona2.error = modificar.der(neurona2.prod) * pNeurona.error * pNeurona.pesos[1];
                neurona1.ajuste_pesos();
                neurona2.ajuste_pesos();
            }

            if (epoca < 2000)
                goto Retry;
            Console.ReadLine();
        }
    }
}