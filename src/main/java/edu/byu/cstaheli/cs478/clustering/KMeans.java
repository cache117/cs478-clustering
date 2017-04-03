package edu.byu.cstaheli.cs478.clustering;

import edu.byu.cstaheli.cs478.toolkit.learner.unsupervised.UnsupervisedLearner;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;
import org.jetbrains.annotations.NotNull;

import java.util.*;

/**
 * Implements k-means clustering.
 */
public class KMeans extends UnsupervisedLearner
{
    private int k;
    private Random random;
    private List<Cluster> clusters;
    private double bestSilhouetteMetric;
    private int timesSilhouetteDecreased;

    public KMeans(int k, Random random)
    {
        this.k = k;
        this.random = random;
        bestSilhouetteMetric = 0;
        clusters = new ArrayList<>(0);
    }

    @NotNull
    private ArrayList<Cluster> populateInitialCentroids(Matrix dataset)
    {
        Map<Integer, Cluster> centroids = new HashMap<>(k);
        for (int i = 0; i < k; ++i)
        {
            int row = getRandomRow(dataset.rows());
            double[] centroid = dataset.row(row);
            centroids.put(row, new Cluster(centroid, dataset));
        }
        return new ArrayList<>(centroids.values());
    }

    @Override
    public void cluster(Matrix dataset)
    {
        clusters = populateInitialCentroids(dataset);
        boolean keepTraining;
        int count = 0;
        do
        {
            ++count;
            printIterationHeader(count);
            printCentroids();
            clearClusters();
            addRowsToClusters(dataset);
            calculateNewCentroids();
            keepTraining = shouldKeepTraining();
            printSilhouetteInfo();
        } while (keepTraining);
    }

    private void printSilhouetteInfo()
    {
        System.out.printf("\nSilhouette : %s\n", bestSilhouetteMetric);
    }

    private void printCentroids()
    {
        int counter = 0;
        printHeader("Printing Centroids");
        for (Cluster cluster : clusters)
        {
            System.out.printf("Centroid %s = ", counter);
            System.out.println(cluster.getCentroidString());
            ++counter;
        }
    }

    private void printIterationHeader(int count)
    {
        System.out.printf("\n***************\nIteration %s\n***************\n", count);
    }

    private void clearClusters()
    {
        for (Cluster cluster : clusters)
        {
            cluster.clear();
        }
    }

    private void addRowsToClusters(Matrix dataset)
    {
        printHeader("Making Assignments");
        for (int i = 0; i < dataset.rows(); ++i)
        {
            if (i % 10 == 0)
            {
                System.out.println();
            }
            double[] row = dataset.row(i);
            Cluster bestCluster = getBestClusterForRow(row);
            bestCluster.add(row);
            printAssignment(i, bestCluster);
        }
    }

    private void printHeader(String header)
    {
        System.out.println(header);
    }

    private void printAssignment(int i, Cluster bestCluster)
    {
        System.out.printf("%s = %s\t", i, clusters.indexOf(bestCluster));
    }

    private Cluster getBestClusterForRow(double[] row)
    {
        double bestDistance = Double.MAX_VALUE;
        Cluster bestCluster = clusters.get(0);
        for (Cluster cluster : clusters)
        {
            double distance = cluster.calcDistanceFromCentroid(row);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestCluster = cluster;
            }
        }
        return bestCluster;
    }

    private void calculateNewCentroids()
    {
        for (Cluster cluster : clusters)
        {
            cluster.calculateNewCentroid();
        }
    }

    private boolean shouldKeepTraining()
    {
        double silhouetteMetric = 0;
        for (Cluster cluster : clusters)
        {
            double internalDissimilarity = cluster.calculateAverageInternalDissimilarity();
            double externalDissimilarity = Double.MAX_VALUE;
            for (Cluster otherCluster : clusters)
            {
                if (!cluster.equals(otherCluster))
                {
                    double dissimilarity = cluster.calculateAverageExternalDissimilarity(otherCluster);
                    if (dissimilarity < externalDissimilarity)
                    {
                        externalDissimilarity = dissimilarity;
                    }
                }
            }
            silhouetteMetric += silhouetteMetric(internalDissimilarity, externalDissimilarity);
        }
        double difference = Math.abs(silhouetteMetric - bestSilhouetteMetric);
        if (silhouetteMetric > bestSilhouetteMetric)
        {
            bestSilhouetteMetric = silhouetteMetric;
        }
        else
        {
            if (++timesSilhouetteDecreased > 2)
            {
                return false;
            }
        }
        return difference > .00001;
    }

    public int getK()
    {
        return k;
    }

    public void setK(int k)
    {
        this.k = k;
    }

    private int getRandomRow(int rows)
    {
        return random.nextInt(rows);
    }

    private double silhouetteMetric(double clusterDissimilarity, double otherClusterDissimilarity)
    {
        if (Double.compare(clusterDissimilarity, otherClusterDissimilarity) == -1)
        {
            return 1 - (clusterDissimilarity / otherClusterDissimilarity);
        }
        else if (Double.compare(clusterDissimilarity, otherClusterDissimilarity) == 0)
        {
            return 0;
        }
        else
        {
            return (otherClusterDissimilarity / clusterDissimilarity) - 1;
        }
    }
}
