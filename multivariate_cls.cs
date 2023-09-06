using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text.RegularExpressions;
using Accord.Math;

namespace multivariate_cls
{
    class Program
    {
        static void Main(string[] args)
        {
            //Training data loading
            List<List<double[]>> allTrainingData = new List<List<double[]>>();
            List<double[]> dataX_1 = new List<double[]>(); //1st class
            List<double[]> dataX_2 = new List<double[]>(); //2nd class
            List<double[]> dataX_3 = new List<double[]>(); //3rd class
            List<double[]> dataX_4 = new List<double[]>(); //4th class
            List<double[]> dataX_5 = new List<double[]>(); //5th class
            List<double[]> dataX_6 = new List<double[]>(); //6th class
            string line;
            Regex dotPattern = new Regex("[.]");
            StreamReader sr = new StreamReader("training_data_classification.txt"); //Input file
            line = sr.ReadLine(); //Second line of the input
            while (line != null) //While there is something to read from the file
            {
                string[] split = line.Split(',');

                for (int tempik = 0; tempik < split.Length; tempik++)
                    split[tempik] = dotPattern.Replace(split[tempik], ",");

                double[] itemik = new double[split.Length - 1]; //Mínus 1 protože poslední je classka
                for (int i = 0; i < split.Count() - 1; i++)
                    itemik[i] = double.Parse(split[i]);

                if (split[split.Count() - 1].Trim() == "1")
                    dataX_1.Add(itemik);
                else if (split[split.Count() - 1].Trim() == "2")
                    dataX_2.Add(itemik);
                else if (split[split.Count() - 1].Trim() == "3")
                    dataX_3.Add(itemik);
                else if (split[split.Count() - 1].Trim() == "4")
                    dataX_4.Add(itemik);
                else if (split[split.Count() - 1].Trim() == "5")
                    dataX_5.Add(itemik);
                else if (split[split.Count() - 1].Trim() == "6")
                    dataX_6.Add(itemik);
                line = sr.ReadLine(); //Read next line
            }
            sr.Close(); //Close the file

            allTrainingData.Add(dataX_1);
            allTrainingData.Add(dataX_2);
            allTrainingData.Add(dataX_3);
            allTrainingData.Add(dataX_4);
            allTrainingData.Add(dataX_5);
            allTrainingData.Add(dataX_6);

            int numberOfData = 0;
            foreach (List<double[]> l in allTrainingData)
                numberOfData += l.Count;

            Console.WriteLine(allTrainingData.Count);
            Console.WriteLine(dataX_1.Count);
            Console.WriteLine(dataX_2.Count);

            int numOfDimensions = dataX_1[0].Length;
            Console.WriteLine("Data loaded! Number of data: {0} Number of dimensions: {1}", numberOfData, numOfDimensions);

            //Calculate probability of each class
            double pc1 = (double)dataX_1.Count / (double)numberOfData;
            double pc2 = (double)dataX_2.Count / (double)numberOfData;
            double pc3 = (double)dataX_3.Count / (double)numberOfData;
            double pc4 = (double)dataX_4.Count / (double)numberOfData;
            double pc5 = (double)dataX_5.Count / (double)numberOfData;
            double pc6 = (double)dataX_6.Count / (double)numberOfData;

            //Calculate mean of each class
            double[] mi_1 = Mi(dataX_1, numOfDimensions);
            double[] mi_2 = Mi(dataX_2, numOfDimensions);
            double[] mi_3 = Mi(dataX_3, numOfDimensions);
            double[] mi_4 = Mi(dataX_4, numOfDimensions);
            double[] mi_5 = Mi(dataX_5, numOfDimensions);
            double[] mi_6 = Mi(dataX_6, numOfDimensions);

            //Calculate variance of each class
            double[] variance_1 = Variance(dataX_1, numOfDimensions, mi_1);
            double[] variance_2 = Variance(dataX_2, numOfDimensions, mi_2);
            double[] variance_3 = Variance(dataX_3, numOfDimensions, mi_3);
            double[] variance_4 = Variance(dataX_4, numOfDimensions, mi_4);
            double[] variance_5 = Variance(dataX_5, numOfDimensions, mi_5);
            double[] variance_6 = Variance(dataX_6, numOfDimensions, mi_6);

            //Calculate covariance of each class
            double[,] coverianceMatrix_1 = Covariance(dataX_1, numOfDimensions, mi_1, variance_1);
            double[,] coverianceMatrix_2 = Covariance(dataX_2, numOfDimensions, mi_2, variance_2);
            double[,] coverianceMatrix_3 = Covariance(dataX_3, numOfDimensions, mi_3, variance_3);
            double[,] coverianceMatrix_4 = Covariance(dataX_4, numOfDimensions, mi_4, variance_4);
            double[,] coverianceMatrix_5 = Covariance(dataX_5, numOfDimensions, mi_5, variance_5);
            double[,] coverianceMatrix_6 = Covariance(dataX_6, numOfDimensions, mi_6, variance_6);

            //Clean data - Remove outliers using Mahalanobis distance
            for (int i = 0; i < dataX_1.Count; i++)
                if (Mahalanobis(dataX_1[i], mi_1, coverianceMatrix_1) > 30)
                    dataX_1.RemoveAt(i);
            for (int i = 0; i < dataX_2.Count; i++)
                if (Mahalanobis(dataX_2[i], mi_2, coverianceMatrix_2) > 30)
                    dataX_2.RemoveAt(i);
            for (int i = 0; i < dataX_3.Count; i++)
                if (Mahalanobis(dataX_3[i], mi_3, coverianceMatrix_3) > 30)
                    dataX_3.RemoveAt(i);
            for (int i = 0; i < dataX_4.Count; i++)
                if (Mahalanobis(dataX_4[i], mi_4, coverianceMatrix_4) > 30)
                    dataX_4.RemoveAt(i);
            for (int i = 0; i < dataX_5.Count; i++)
                if (Mahalanobis(dataX_5[i], mi_5, coverianceMatrix_5) > 30)
                    dataX_5.RemoveAt(i);
            for (int i = 0; i < dataX_6.Count; i++)
                if (Mahalanobis(dataX_6[i], mi_6, coverianceMatrix_6) > 30)
                    dataX_6.RemoveAt(i);

            //Load testing data
            List<string> initialTestingData = new List<string>();
            List<double[]> testingData = new List<double[]>();
            sr = new StreamReader("testing_data_classification.txt"); //Input file
            line = sr.ReadLine();
            while (line != null) //While there is something to read from the file
            {
                initialTestingData.Add(line);

                string[] split = line.Split(',');

                for (int tempik = 0; tempik < split.Length; tempik++)
                    split[tempik] = dotPattern.Replace(split[tempik], ",");

                double[] itemik = new double[split.Length];
                for (int i = 0; i < split.Count(); i++)
                {
                    itemik[i] = double.Parse(split[i]);
                }

                testingData.Add(itemik);

                line = sr.ReadLine(); //Read next line
            }
            sr.Close(); //Close the file

            FileStream stream = new FileStream("answer_data_classification.txt", FileMode.Create);
            StreamWriter file = new StreamWriter(stream);

            using (file)
            {
                for (int i = 0; i < testingData.Count; i++)
                {
                    double g1 = Discriminant(testingData[i], pc1, coverianceMatrix_1, mi_1);
                    double g2 = Discriminant(testingData[i], pc2, coverianceMatrix_2, mi_2);
                    double g3 = Discriminant(testingData[i], pc3, coverianceMatrix_3, mi_3);
                    double g4 = Discriminant(testingData[i], pc4, coverianceMatrix_4, mi_4);
                    double g5 = Discriminant(testingData[i], pc5, coverianceMatrix_5, mi_5);
                    double g6 = Discriminant(testingData[i], pc6, coverianceMatrix_6, mi_6);

                    //Get max value from all discriminants
                    double maxG = new double[] { g1, g2, g3, g4, g5, g6 }.Max();

                    string resultClass = "None";
                    //Select corresponding class
                    if (maxG == g1)
                        resultClass = "1";
                    else if (maxG == g2)
                        resultClass = "2";
                    else if (maxG == g3)
                        resultClass = "3";
                    else if (maxG == g4)
                        resultClass = "4";
                    else if (maxG == g5)
                        resultClass = "5";
                    else if (maxG == g6)
                        resultClass = "6";

                    Console.WriteLine("{0}={7} [{1}, {2}, {3}, {4}, {5}, {6}]", i + 1, g1, g2, g3, g4, g5, g6, resultClass);

                    file.WriteLine("{0},{1}", initialTestingData[i], resultClass);
                }
            }

            Console.WriteLine("Done!");
            Console.ReadKey();
        }

        public static double Mahalanobis(double[] x_trainingData, double[] mi, double[,] cov_i)
        {
            return Matrix.Dot(Matrix.Transpose(Matrix.Transpose(Matrix.Dot(Elementwise.Subtract(x_trainingData, mi), Matrix.Inverse(cov_i)))), Matrix.Transpose(Elementwise.Subtract(x_trainingData, mi)))[0, 0] / 2;
        }

        public static double Discriminant(double[] x_testingData, double pci, double[,] cov_i, double[] mi)
        {
            //Calculate Part 1
            double part_1 = Math.Log(Matrix.Determinant(cov_i)) * -1 / 2;

            //Calculate Part 2
            double[] xMinusMi = Elementwise.Subtract(x_testingData, mi);
            double[,] temp = Matrix.Transpose(Matrix.Transpose(Matrix.Dot(xMinusMi, Matrix.Inverse(cov_i)))); //These two transposes are here as a work-around to get matrix datatype (from one-d array) for next calculation.
            double part_2 = Matrix.Dot(temp, Matrix.Transpose(xMinusMi))[0, 0] / 2;

            //Calculate Part 3
            double part_3 = Math.Log(pci);

            //Return result
            return part_1 - part_2 + part_3;
        }

        public static double[] Mi(List<double[]> dataX, int pocetDimenzi)
        {
            double[] mi = new double[pocetDimenzi];
            for (int y = 0; y < dataX.Count; y++) //For every N
                for (int x = 0; x < dataX[y].Length; x++) //For every element in transaction
                    mi[x] += dataX[y][x];

            for (int d = 0; d < pocetDimenzi; d++)
            {
                mi[d] = mi[d] / dataX.Count;
            }
            return mi;
        }

        public static double[] Variance(List<double[]> dataX, int dimensions, double[] mi)
        {
            double[] varOfEachColumn = new double[dimensions];
            for (int i = 0; i < dimensions; i++)
            {
                for (int j = 0; j < dataX.Count; j++)
                {
                    varOfEachColumn[i] += Math.Pow(dataX[j][i] - mi[i], 2);
                }
            }
            for (int d = 0; d < varOfEachColumn.Length; d++)
            {
                varOfEachColumn[d] = varOfEachColumn[d] / dataX.Count;
            }
            return varOfEachColumn;
        }

        public static double Coveriance(double[] dimenzeA, double meanA, double[] dimenzeB, double meanB, int numberOfData)
        {
            double result = 0;
            for (int a = 0; a < dimenzeA.Length; a++) //Both dimensions are same lenght, so it doesnt matter which I put here
                result += (dimenzeA[a] - meanA) * (dimenzeB[a] - meanB);
            return result / numberOfData;
        }

        public static double[,] Covariance(List<double[]> dataX, int pocetDimenzi, double[] mi, double[] varKazdehoSloupce)
        {
            //Covariance (dimenze x dimenze matrix)
            double[,] coverianceMatrix = new double[pocetDimenzi, pocetDimenzi];
            double[] dimenzeA_temp = new double[dataX.Count];
            double[] dimenzeB_temp = new double[dataX.Count];

            for (int x = 0; x < pocetDimenzi; x++)
            {
                for (int y = 0; y < pocetDimenzi; y++)
                {
                    if (x == y) //Variance on the diagonal
                    {
                        coverianceMatrix[x, y] = varKazdehoSloupce[x];
                    }
                    else
                    {
                        for (int i = 0; i < dataX.Count; i++)
                        {
                            dimenzeA_temp[i] = dataX[i][x];
                            dimenzeB_temp[i] = dataX[i][y];
                        }

                        coverianceMatrix[x, y] = Coveriance(dimenzeA_temp, mi[x], dimenzeB_temp, mi[y], dataX.Count);
                    }
                }
            }
            return coverianceMatrix;
        }
    }
}
