package edu.byu.cstaheli.cs478.clustering;

import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.*;

import static edu.byu.cstaheli.cs478.toolkit.utility.Utility.euclideanDistance;

/**
 * Represents a Cluster in a clustering, unsupervised algorithm, such as k-means or HAC.
 */
public class Cluster
{
    private List<double[]> rows;
    private double[] centroid;
    private List<Boolean> continuousColumn;
    private List<Map<Integer, String>> stringRepresentations;

    public Cluster(double[] centroid, Matrix dataset)
    {
        this.centroid = centroid;
        rows = new ArrayList<>();
        continuousColumn = populateContinuousColumnIndicators(dataset);
        stringRepresentations = dataset.getStringRepresentations();
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Cluster cluster = (Cluster) o;

        if (!rows.equals(cluster.rows)) return false;
        if (!Arrays.equals(centroid, cluster.centroid)) return false;
        return continuousColumn.equals(cluster.continuousColumn);
    }

    @Override
    public int hashCode()
    {
        int result = rows.hashCode();
        result = 31 * result + Arrays.hashCode(centroid);
        result = 31 * result + continuousColumn.hashCode();
        return result;
    }

    private List<Boolean> populateContinuousColumnIndicators(Matrix dataset)
    {
        List<Boolean> continuousColumnIndicators = new ArrayList<>(dataset.cols());
        for (int i = 0; i < dataset.cols(); ++i)
        {
            boolean indicator = dataset.valueCount(i) == 0;
            continuousColumnIndicators.add(indicator);
        }
        return continuousColumnIndicators;
    }

    private boolean isColumnContinuous(int column)
    {
        return continuousColumn.get(column);
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
            if (isColumnContinuous(i))
            {
                distance += Double.compare(firstValue, secondValue) == 0 ? 0 : 1;
            }
            else if (isValueUnknown(firstValue) || isValueUnknown(secondValue))
            {
                distance += 1;
            }
            else
            {
                distance += euclideanDistance(firstValue, secondValue);
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

    private double[] calcAverageCentroid()
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
            return calculateArrayMedian(columnValues);
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

    private double calculateArrayMedian(double[] columnValues)
    {
        Arrays.sort(columnValues);
        double median;
        if (columnValues.length % 2 == 0)
        {
            median = (columnValues[columnValues.length / 2] + columnValues[columnValues.length / 2 - 1]) / 2;
        }
        else
        {
            median = columnValues[columnValues.length / 2];
        }
        return median;
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
            if (!isValueUnknown(value))
            {
                columnValues[i] = value;
            }
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

    public String getCentroidString()
    {
        StringJoiner joiner = new StringJoiner(",\t");
        for (int i = 0; i < centroid.length; i++)
        {
            double value = centroid[i];
            if (Double.compare(value, Matrix.MISSING) == 0)
            {
                joiner.add("?");
            }
            else if (!isColumnContinuous(i))
            {
                joiner.add(stringRepresentations.get(i).get((int) value));
            }
            else
            {
                DecimalFormat format = new DecimalFormat("#.###");
                format.setRoundingMode(RoundingMode.HALF_UP);
                joiner.add(format.format(value));
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
}
