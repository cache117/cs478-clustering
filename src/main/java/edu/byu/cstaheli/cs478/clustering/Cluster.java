package edu.byu.cstaheli.cs478.clustering;

import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

import java.util.*;

import static edu.byu.cstaheli.cs478.toolkit.utility.Utility.euclideanDistance;
import static edu.byu.cstaheli.cs478.toolkit.utility.Utility.getFormattedDouble;

/**
 * Represents a Cluster in a clustering, unsupervised algorithm, such as k-means or HAC.
 */
public class Cluster
{
    private List<double[]> rows;
    private double[] centroid;
    private Matrix dataset;

    public Cluster(double[] centroid, Matrix dataset)
    {
        this.centroid = centroid;
        rows = new ArrayList<>();
        this.dataset = dataset;
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Cluster cluster = (Cluster) o;

        if (!rows.equals(cluster.rows)) return false;
        return Arrays.equals(centroid, cluster.centroid);
    }

    @Override
    public int hashCode()
    {
        int result = rows.hashCode();
        result = 31 * result + Arrays.hashCode(centroid);
        return result;
    }

    private boolean isColumnContinuous(int column)
    {
        return dataset.valueCount(column) == 0;
    }

    public double calcDistanceFromCentroid(double[] row)
    {
        return calcDistance(row, centroid);
    }

    public double calcDistance(double[] first, double[] second)
    {
        assert first.length == second.length;
        double distance = 0;
        for (int i = 0; i < first.length; ++i)
        {
            double firstValue = first[i];
            double secondValue = second[i];
            if (isValueUnknown(firstValue) || isValueUnknown(secondValue))
            {
                distance += 1;
            }
            else if (isColumnContinuous(i))
            {
                distance += euclideanDistance(firstValue, secondValue);
            }
            else
            {
                distance += Double.compare(firstValue, secondValue) == 0 ? 0 : 1;
            }
        }
        return distance;
    }

    private boolean isValueUnknown(double value)
    {
        return Double.compare(value, Matrix.MISSING) == 0;
    }

    public void clear()
    {
        rows.clear();
    }

    public void add(double[] row)
    {
        rows.add(row);
    }

    public void calculateNewCentroid()
    {
        this.centroid = calcAverageCentroid();
    }

    protected double[] calcAverageCentroid()
    {
        double[] centroid = new double[this.centroid.length];
        for (int i = 0; i < this.centroid.length; ++i)
        {
            centroid[i] = calcColumnAverage(i);
        }
        return centroid;
    }

    private double calcColumnAverage(int column)
    {
        double[] columnValues = getColumnValues(column);
        if (isColumnContinuous(column))
        {
            return calculateArrayAverage(columnValues);
        }
        else
        {
            return calculateArrayMode(columnValues);
        }
    }

    private double calculateArrayAverage(double[] columnValues)
    {
        double sum = 0;
        int counter = 0;
        for (double value : columnValues)
        {
            if (Double.compare(value, Matrix.MISSING) != 0)
            {
                sum += value;
                ++counter;
            }
        }
        if (sum == 0)
        {
            return Matrix.MISSING;
        }
        else
        {
            return sum / counter;
        }
    }

    private double calculateArrayMode(double[] columnValues)
    {
//        Arrays.sort(columnValues);
//        double median;
//        if (columnValues.length % 2 == 0)
//        {
//            median = (columnValues[columnValues.length / 2] + columnValues[columnValues.length / 2 - 1]) / 2;
//        }
//        else
//        {
//            median = columnValues[columnValues.length / 2];
//        }
//        return median;

        Map<Double, Integer> occurrences = new TreeMap<>();
        for (double v : columnValues)
        {
            occurrences.merge(v, 1, (a, b) -> a + b);
        }
        int highestCount = 0;
        double mode = -1;
        assert occurrences.entrySet().size() > 0;
        for (Map.Entry<Double, Integer> entry : occurrences.entrySet())
        {
            if (Double.compare(entry.getKey(), Matrix.MISSING) != 0 && entry.getValue() > highestCount)
            {
                highestCount = entry.getValue();
                mode = entry.getKey();
            }
        }
        return mode;
    }

    /**
     * Adds all of the values in the cluster from the specified column to an array. This doesn't add missing values
     *
     * @param column the column to take from.
     * @return the values in the column
     */
    private double[] getColumnValues(int column)
    {
        double[] columnValues = new double[rows.size()];
        for (int i = 0; i < rows.size(); ++i)
        {
            double value = get(i, column);
            columnValues[i] = value;
        }
        return columnValues;
    }

    private double get(int rowNumber, int column)
    {
        return rows.get(rowNumber)[column];
    }

    public double calculateAverageInternalDissimilarity()
    {
        return calculateAverageDissimilarity(rows, rows);
    }

    public double calculateAverageExternalDissimilarity(Cluster other)
    {
        return calculateAverageDissimilarity(rows, other.rows);
    }

    private double calculateAverageDissimilarity(List<double[]> first, List<double[]> second)
    {
        double sum = 0;
        double count = 0;
        for (int i = 0; i < first.size(); ++i)
        {
            for (int j = 0; j < second.size(); ++j)
            {
                if (i != j)
                {
                    sum += calcDistance(first.get(i), second.get(j));
                    ++count;
                }
            }
        }
        return sum / count;
    }

    public double[] getCentroid()
    {
        return centroid;
    }

    public String getCentroidString(String delimiter)
    {
        StringJoiner joiner = new StringJoiner(delimiter);
        for (int i = 0; i < centroid.length; i++)
        {
            double value = centroid[i];
            if (Double.compare(value, Matrix.MISSING) == 0)
            {
                joiner.add("?");
            }
            else if (!isColumnContinuous(i))
            {
                joiner.add(dataset.attrValue(i, (int) value));
            }
            else
            {
                joiner.add(getFormattedDouble(value));
            }
        }
        return joiner.toString();
    }

    public int size()
    {
        return rows.size();
    }

    public boolean empty()
    {
        return size() == 0;
    }

    public double calcSSE()
    {
        double value = 0;
        for (double[] row : rows)
        {
            value += calcDistanceFromCentroid(row);
        }
        return value;
    }
}
