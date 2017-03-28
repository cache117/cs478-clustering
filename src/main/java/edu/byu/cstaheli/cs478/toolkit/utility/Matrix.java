package edu.byu.cstaheli.cs478.toolkit.utility;
// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import edu.byu.cstaheli.cs478.toolkit.exception.ARFFParseException;
import edu.byu.cstaheli.cs478.toolkit.exception.IncompatibleMatrixException;
import edu.byu.cstaheli.cs478.toolkit.exception.MatrixException;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.Map.Entry;

import static edu.byu.cstaheli.cs478.toolkit.utility.Utility.removeColumnFromRow;
import static edu.byu.cstaheli.cs478.toolkit.utility.Utility.square;

/**
 * Class that represents an arff file
 */
public class Matrix
{
    /**
     * Representation of missing values in the dataset
     */
    public static double MISSING = Double.MAX_VALUE;

    // Data
    private List<double[]> m_data;

    // Meta-data
    private List<String> m_attr_name;
    private List<Map<String, Integer>> m_str_to_enum;
    private List<Map<Integer, String>> m_enum_to_str;
    private String datasetName;
    private boolean binRealValues;

    /**
     * Creates a matrix from a arff file. This is exactly the same as creating an empty matrix and then loading the file.
     *
     * @param fileName the path to the arff file.
     * @throws FileNotFoundException if the file doesn't exist
     * @throws ARFFParseException    if there is something wrong with the setup of the arff file.
     */
    public Matrix(String fileName) throws FileNotFoundException, ARFFParseException
    {
        this();
        loadArff(fileName);
    }

    /**
     * Creates a 0x0 matrix. You should call loadARFF or setSize next.
     */
    public Matrix()
    {
    }

    /**
     * Creates a copy of the entire matrix
     *
     * @param that the other matrix
     */
    public Matrix(Matrix that)
    {
        this(that, 0, 0, that.rows(), that.cols());
    }

    /**
     * Copies the specified portion of that matrix into this matrix
     *
     * @param that     the other matrix
     * @param rowStart what row to start on
     * @param colStart what column to start on
     * @param rowCount how many rows to copy
     * @param colCount how many columns to copy
     */
    public Matrix(Matrix that, int rowStart, int colStart, int rowCount, int colCount)
    {
        m_data = new ArrayList<>();
        for (int j = 0; j < rowCount; j++)
        {
            double[] rowSrc = that.row(rowStart + j);
            double[] rowDest = new double[colCount];
            for (int i = 0; i < colCount; i++)
            {
                rowDest[i] = rowSrc[colStart + i];
            }
            m_data.add(rowDest);
        }
        m_attr_name = new ArrayList<>();
        m_str_to_enum = new ArrayList<>();
        m_enum_to_str = new ArrayList<>();
        for (int i = 0; i < colCount; i++)
        {
            m_attr_name.add(that.attrName(colStart + i));
            m_str_to_enum.add(that.m_str_to_enum.get(colStart + i));
            m_enum_to_str.add(that.m_enum_to_str.get(colStart + i));
        }
    }

    /**
     * Returns the statistical z-score for the given value with the given mean and standard deviation.
     * This is equivalent to the number of standard deviations that the value is from the mean.
     *
     * @param value             the value
     * @param mean              the mean of the column
     * @param standardDeviation the standard deviation for the column
     * @return the z score
     */
    private static double zScore(double value, double mean, double standardDeviation)
    {
        return (value - mean) / standardDeviation;
    }

    /**
     * Loads from an ARFF file
     *
     * @param fileName the path to the arff file.
     * @throws FileNotFoundException if the file doesn't exist
     * @throws ARFFParseException    if there is something wrong with the setup of the arff file.
     */
    public void loadArff(String fileName) throws ARFFParseException, FileNotFoundException
    {
        m_data = new ArrayList<>();
        m_attr_name = new ArrayList<>();
        m_str_to_enum = new ArrayList<>();
        m_enum_to_str = new ArrayList<>();
        boolean readData = false;
        Scanner s = new Scanner(new File(fileName));
        while (s.hasNext())
        {
            String line = s.nextLine()
                    .trim();
            if (line.length() > 0 && line.charAt(0) != '%')
            {
                if (!readData)
                {
                    Scanner t = new Scanner(line);
                    String firstToken = t.next()
                            .toUpperCase();

                    if (firstToken.equals("@RELATION"))
                    {
                        datasetName = t.nextLine();
                    }

                    if (firstToken.equals("@ATTRIBUTE"))
                    {
                        addAttribute(line);
                    }
                    if (firstToken.equals("@DATA"))
                    {
                        readData = true;
                    }
                }
                else
                {
                    addNewRow(line);
                }
            }
        }
        if (binRealValues)
        {
            binRealValues();
        }
    }

    /**
     * Adds an attribute to the Matrix from a line in the arff file
     *
     * @param line the line in the arff file
     * @throws ARFFParseException if the line can't be parsed correctly
     */
    private void addAttribute(String line) throws ARFFParseException
    {
        Map<String, Integer> ste = new TreeMap<>();
        m_str_to_enum.add(ste);
        Map<Integer, String> ets = new TreeMap<>();
        m_enum_to_str.add(ets);

        Scanner u = new Scanner(line);
        if (line.contains("'"))
        {
            u.useDelimiter("'");
        }
        u.next();
        String attributeName = u.next();
        if (line.contains("'"))
        {
            attributeName = "'" + attributeName + "'";
        }
        m_attr_name.add(attributeName);

        String type = u.next()
                .trim()
                .toUpperCase();
        if (!type.equals("REAL") && !type.equals("CONTINUOUS") && !type.equals("INTEGER"))
        {
            addNominalAttribute(line, ste, ets);
        }
    }

    /**
     * Adds a nominal attribute to the Matrix from a line in the arff file
     *
     * @param line the line in the arff file
     * @param ste  a new Map that exists in the Matrix representing Attribute Names mapped to their values
     * @param ets  a new Map that exists in the Matrix representing values mapped to their Attribute Names
     * @throws ARFFParseException if the line can't be parsed correctly
     */
    private void addNominalAttribute(String line, Map<String, Integer> ste, Map<Integer, String> ets) throws ARFFParseException
    {
        int vals = 0;
        try
        {
            String values = line.substring(line.indexOf("{") + 1, line.indexOf("}"));
            Scanner v = new Scanner(values);
            v.useDelimiter(",");
            while (v.hasNext())
            {
                String value = v.next()
                        .trim();
                if (value.length() > 0)
                {
                    ste.put(value, vals);
                    ets.put(vals, value);
                    vals++;
                }
            }
        }
        catch (Exception e)
        {
            throw new ARFFParseException("Error parsing line: " + line + "\n" + e.toString());
        }
    }

    /**
     * Adds a new row to the Matrix from a line in the arff file
     *
     * @param line the line in the arff file
     * @throws ARFFParseException if the line can't be parsed correctly
     */
    private void addNewRow(String line) throws ARFFParseException
    {
        double[] newRow = new double[cols()];

        try
        {
            Scanner scanner = new Scanner(line)
                    .useDelimiter(",");
            int curPos = 0;
            while (scanner.hasNext())
            {
                String textValue = scanner.next()
                        .trim();

                if (textValue.length() > 0)
                {
                    double doubleValue;
                    int vals = valueCount(curPos);

                    //Missing instances appear in the dataset as a double defined as MISSING
                    if (textValue.equals("?"))
                    {
                        doubleValue = MISSING;
                    }
                    // Continuous values appear in the instance vector as they are
                    else if (vals == 0)
                    {
                        doubleValue = Double.parseDouble(textValue);
                    }
                    // Discrete values appear as an index to the "name"
                    // of that value in the "attributeValue" structure
                    else
                    {
                        doubleValue = m_str_to_enum.get(curPos)
                                .get(textValue);
                        if (doubleValue == -1)
                        {
                            throw new ARFFParseException("Error parsing the value '" + textValue + "' on line: " + line);
                        }
                    }

                    newRow[curPos] = doubleValue;
                    curPos++;
                }
            }
        }
        catch (Exception e)
        {
            throw new ARFFParseException("Error parsing line: " + line + "\n" + e.toString());
        }
        m_data.add(newRow);
    }

    /**
     * Returns the number of columns (or attributes) in the matrix
     *
     * @return the number of columns (or attributes) in the matrix
     */
    public int cols()
    {
        return m_attr_name.size();
    }

    /**
     * Returns the number of values associated with the specified attribute (or column).
     *
     * @param col the column index.
     * @return 0=continuous, 2=binary, 3=trinary, etc
     */
    public int valueCount(int col)
    {
        return m_enum_to_str.get(col)
                .size();
    }

    /**
     * Takes all of the real values and bins them based on the number of standard deviations from the mean.
     */
    private void binRealValues()
    {
        for (int i = 0; i < m_data.size(); ++i)
        {
            double[] row = m_data.get(i);
            for (int j = 0; j < row.length; ++j)
            {
                if (valueCount(j) == 0)
                {
                    double value = row[j];
                    double mean = columnMean(j);
                    double standardDeviation = columnStandardDeviation(j);
                    double zScore = zScore(value, mean, standardDeviation);
                    int numberOfStandardDeviations = (int) zScore;
                    set(i, j, numberOfStandardDeviations);
                }
            }
        }
        for (int i = 0; i < cols(); ++i)
        {
            if (valueCount(i) == 0)
            {
                Map<String, Integer> ste = m_str_to_enum.get(i);
                Map<Integer, String> ets = m_enum_to_str.get(i);
                Map<Double, Integer> columnOccurrences = getColumnOccurrences(i);
                int counter = 0;
                for (Entry<Double, Integer> entry : columnOccurrences.entrySet())
                {
                    ste.put(String.valueOf(entry.getKey()), counter);
                    ets.put(counter, String.valueOf(entry.getKey()));
                    ++counter;
                }
            }
        }
    }

    /**
     * Sets the value at the specified row and column
     *
     * @param row    the row index
     * @param column the column index
     * @param value  the new value
     */
    public void set(int row, int column, double value)
    {
        row(row)[column] = value;
    }

    /**
     * Returns the specified row
     *
     * @param row the row index
     * @return the specified row
     */
    public double[] row(int row)
    {
        return m_data.get(row);
    }

    /**
     * Returns the mean of the specified column.
     *
     * @param col the column index.
     * @return the mean of the specified column.
     */
    public double columnMean(int col)
    {
        double sum = 0;
        int count = 0;
        for (int i = 0; i < rows(); i++)
        {
            double v = get(i, col);
            if (v != MISSING)
            {
                sum += v;
                count++;
            }
        }
        return sum / count;
    }

    /**
     * Returns the number of rows in the matrix
     *
     * @return the number of rows in the matrix
     */
    public int rows()
    {
        return m_data.size();
    }

    /**
     * Returns the element at the specified row and column
     *
     * @param row    the row index
     * @param column the column index
     * @return the element at the specified row and column
     */
    public double get(int row, int column)
    {
        return m_data.get(row)[column];
    }

    /**
     * Returns the standard deviation of the specified column.
     *
     * @param col the column index
     * @return the standard deviation of the specified column.
     */
    private double columnStandardDeviation(int col)
    {
        double sum = 0;
        double average = columnMean(col);
        for (int i = 0; i < rows(); ++i)
        {
            double value = get(i, col);
            if (value != MISSING)
            {
                sum += square(value - average) / rows();
            }
        }
        return Math.sqrt(sum);
    }

    /**
     * Returns a Map of the possible values in the given column along with the number of times they occur.
     *
     * @param col the column index
     * @return the occurrences of values in the column.
     */
    public Map<Double, Integer> getColumnOccurrences(int col)
    {
        Map<Double, Integer> occurrences = new TreeMap<>();
        for (int i = 0; i < rows(); i++)
        {
            double v = get(i, col);
            if (v != MISSING || binRealValues)
            {
                Integer count = occurrences.get(v);
                if (count == null)
                {
                    occurrences.put(v, 1);
                }
                else
                {
                    occurrences.put(v, count + 1);
                }
            }
        }
        return occurrences;
    }

    /**
     * Returns the name of the specified attribute
     *
     * @param column the column index
     * @return the name of the specified attribute
     */
    public String attrName(int column)
    {
        return m_attr_name.get(column);
    }

    public String getDatasetName()
    {
        return datasetName;
    }

    public void setDatasetName(String datasetName)
    {
        this.datasetName = datasetName;
    }

    /**
     * Returns if real values are binned based on standard deviations.
     *
     * @return true if real values are binned, false otherwise.
     */
    public boolean areRealValuesBinned()
    {
        return binRealValues;
    }

    /**
     * This instructs the Matrix to bin real values based on standard deviations.
     */
    public void doBinRealValues()
    {
        this.binRealValues = true;
    }

    /**
     * Adds a copy of the specified portion of that matrix to this matrix
     *
     * @param that     the other matrix
     * @param rowStart what row to start on
     * @param colStart what column to start on
     * @param rowCount how many rows to copy
     * @throws MatrixException if this Matrix and other Matrix are incompatible
     */
    public void add(Matrix that, int rowStart, int colStart, int rowCount) throws MatrixException
    {
        if (colStart + cols() > that.cols())
        {
            throw new IndexOutOfBoundsException(String.format("Out of Range: %d + %d > %d", colStart, cols(), that.cols()));
        }
        for (int i = 0; i < cols(); i++)
        {
            if (that.valueCount(colStart + i) != valueCount(i))
            {
                throw new IncompatibleMatrixException(String.format("incompatible Relations: %d != %d", that.valueCount(colStart + i), valueCount(i)));
            }
        }
        for (int j = 0; j < rowCount; j++)
        {
            double[] rowSrc = that.row(rowStart + j);
            double[] rowDest = new double[cols()];
            for (int i = 0; i < cols(); i++)
            {
                rowDest[i] = rowSrc[colStart + i];
            }
            m_data.add(rowDest);
        }
    }

    /**
     * Resizes this matrix (and sets all attributes to be continuous)
     *
     * @param rows the number of rows in the matrix
     * @param cols the number of columns in the matrix
     */
    public void setSize(int rows, int cols)
    {
        m_data = new ArrayList<>();
        for (int j = 0; j < rows; j++)
        {
            double[] row = new double[cols];
            m_data.add(row);
        }
        m_attr_name = new ArrayList<>();
        m_str_to_enum = new ArrayList<>();
        m_enum_to_str = new ArrayList<>();
        for (int i = 0; i < cols; i++)
        {
            m_attr_name.add("");
            m_str_to_enum.add(new TreeMap<>());
            m_enum_to_str.add(new TreeMap<>());
        }
    }

    /**
     * Set the name of the specified attribute
     *
     * @param column the column index
     * @param name   the new name for the attribute
     */
    public void setAttrName(int column, String name)
    {
        m_attr_name.set(column, name);
    }

    /**
     * Returns the name of the specified value
     *
     * @param attr the attribute
     * @param val  the value
     * @return the name of the specified value
     */
    public String attrValue(int attr, int val)
    {
        return m_enum_to_str.get(attr)
                .get(val);
    }

    /**
     * Shuffles the row order
     *
     * @param rand the random to shuffle with
     */
    public void shuffle(Random rand)
    {
        for (int n = rows(); n > 0; n--)
        {
            int i = rand.nextInt(n);
            double[] tmp = row(n - 1);
            m_data.set(n - 1, row(i));
            m_data.set(i, tmp);
        }
    }

    /**
     * Shuffles the row order with a buddy matrix
     *
     * @param rand  the random to shuffle with
     * @param buddy the buddy matrix
     */
    public void shuffle(Random rand, Matrix buddy)
    {
        for (int n = rows(); n > 0; n--)
        {
            int i = rand.nextInt(n);
            double[] tmp = row(n - 1);
            m_data.set(n - 1, row(i));
            m_data.set(i, tmp);

            double[] tmp1 = buddy.row(n - 1);
            buddy.m_data.set(n - 1, buddy.row(i));
            buddy.m_data.set(i, tmp1);
        }
    }

    /**
     * Returns the most common value in the specified column
     *
     * @param col the column index
     * @return the most common value in the specified column
     */
    public double mostCommonValue(int col)
    {
        Map<Double, Integer> tm = getColumnOccurrences(col);
        int maxCount = 0;
        double val = MISSING;
        for (Entry<Double, Integer> e : tm.entrySet())
        {
            if (e.getValue() > maxCount)
            {
                maxCount = e.getValue();
                val = e.getKey();
            }
        }
        return val;
    }

    /**
     * Returns a copy of this Matrix with only rows that have the matching value in the given column.
     *
     * @param columnClass the index of the column in question
     * @param value       the value to match against
     * @return a matrix with only matching rows.
     * @throws MatrixException
     */
    public Matrix getRowsWithColumnClass(int columnClass, double value) throws MatrixException
    {
        Matrix newMatrix = new Matrix(this);
        Double doubleValue = value;

        newMatrix.m_data.removeIf(row -> !doubleValue.equals(row[columnClass]));
        newMatrix.removeColumn(columnClass);
        return newMatrix;
    }

    /**
     * Removes the specified column from the matrix
     *
     * @param column the column index
     */
    public void removeColumn(int column)
    {
        m_attr_name.remove(column);
        m_str_to_enum.remove(column);
        m_enum_to_str.remove(column);
        for (int i = 0; i < m_data.size(); ++i)
        {
            m_data.set(i, removeColumnFromRow(column, m_data.get(i)));
        }
    }

    /**
     * Normalizes the data in all of the columns between 0 and 1.
     */
    public void normalize()
    {
        for (int i = 0; i < cols(); i++)
        {
            if (valueCount(i) == 0)
            {
                double min = columnMin(i);
                double max = columnMax(i);
                for (int j = 0; j < rows(); j++)
                {
                    double v = get(j, i);
                    if (v != MISSING)
                    {
                        set(j, i, (v - min) / (max - min));
                    }
                }
            }
        }
    }

    /**
     * Returns the min value in the specified column
     *
     * @param col the column index
     * @return the min value in the specified column
     */
    public double columnMin(int col)
    {
        double m = MISSING;
        for (int i = 0; i < rows(); i++)
        {
            double v = get(i, col);
            if (v != MISSING)
            {
                if (m == MISSING || v < m)
                {
                    m = v;
                }
            }
        }
        return m;
    }

    /**
     * Returns the max value in the specified column
     *
     * @param col the column index
     * @return the max value in the specified column
     */
    public double columnMax(int col)
    {
        double m = MISSING;
        for (int i = 0; i < rows(); i++)
        {
            double v = get(i, col);
            if (v != MISSING)
            {
                if (m == MISSING || v > m)
                {
                    m = v;
                }
            }
        }
        return m;
    }

    /**
     * Prints out the Matrix
     */
    public void print()
    {
        System.out.print(this);
    }

    @Override
    public String toString()
    {
        StringBuilder builder = new StringBuilder();
        builder.append("@RELATION ")
                .append(datasetName)
                .append("\n");
        for (int i = 0; i < m_attr_name.size(); i++)
        {
            builder.append("@ATTRIBUTE ")
                    .append(m_attr_name.get(i));
            int values = valueCount(i);
            if (values == 0)
            {
                builder.append(" CONTINUOUS\n");
            }
            else
            {
                builder.append(" {");
                for (int j = 0; j < values; j++)
                {
                    if (j > 0)
                    {
                        builder.append(", ");
                    }
                    builder.append(m_enum_to_str.get(i)
                            .get(j));
                }
                builder.append("}\n");
            }
        }
        builder.append("@DATA\n");
        for (int i = 0; i < rows(); i++)
        {
            double[] r = row(i);
            for (int j = 0; j < r.length; j++)
            {
                if (j > 0)
                {
                    builder.append(", ");
                }
                if (valueCount(j) == 0)
                {
                    builder.append(r[j]);
                }
                else
                {
                    builder.append(m_enum_to_str.get(j)
                            .get((int) r[j]));
                }
            }
            builder.append("\n");
        }
        return builder.toString();
    }
}
