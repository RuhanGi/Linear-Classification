/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   maths.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/01/30 18:53:59 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/01/30 18:53:59 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Linear.hpp"


Matrix randomizeMatrix(size_t rows, size_t cols)
{
    Matrix res(rows, Row(cols));
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            res[i][j] = std::rand() % 21 - 10;
    return res;
}


bool checkRectangle(const Matrix& A, bool Square)
{
    if (A.empty())
        return false; // * Might change depending on implementation

    size_t rows = A.size();
    size_t cols = A[0].size();
    if (cols == 0 || (Square && rows != cols))
        return false;
    for (size_t i = 1; i < rows; i++)
        if (A[i].size() != cols)
            return false;    
    return true;
}


Matrix transpose(const Matrix& m)
{
    if (m.empty())
        return {};

    Matrix res;
    for (size_t j = 0; j < m[0].size(); j++)
    {
        Row row;
        for (size_t i = 0; i < m.size(); i++)
            row.push_back(m[i][j]);
        res.push_back(row);
    }
    return res;
}


double swap(Matrix& m, size_t row_1, size_t row_2)
{
    if (row_1 >= m.size() || row_2 >= m.size())
        throw std::invalid_argument("Invalid Index");
    if (row_1 == row_2)
        return 1;

    std::swap(m[row_1], m[row_2]);
    return -1;
}


double scale(Matrix& m, double k, size_t row)
{
    if (row >= m.size())
        throw std::invalid_argument("Invalid Index");

    for (size_t i = 0; i < m[row].size(); i++)
        m[row][i] *= k;
    return k;
}


double replace(Matrix& m, size_t row_1, double k, size_t row_2)
{
    if (row_1 >= m.size() || row_2 >= m.size())
        throw std::invalid_argument("Invalid Index");
    if (m[row_1].size() != m[row_2].size())
        throw std::invalid_argument("Row Dimensions Mismatch");

    for (size_t i = 0; i < m[row_1].size(); i++)
        m[row_1][i] += k * m[row_2][i];
    return 1;
}


double gaussJordan(Matrix& aug, size_t n) 
{
    double det_val = 1.0;

    for (size_t j = 0; j < n; j++)
    {
        size_t pivot = j;
        for (size_t i = j + 1; i < n; i++)
            if (std::abs(aug[i][j]) > std::abs(aug[pivot][j]))
                pivot = i;
        det_val *= swap(aug, j, pivot);
        if (std::abs(aug[j][j]) < 1e-15) 
            return 0.0;
        det_val *= 1.0 / scale(aug, 1.0/aug[j][j], j);
        for (size_t i = 0; i < n; i++)
            if (i != j)
                det_val *= replace(aug, i, -aug[i][j], j);
    }
    return det_val;
}


double det(Matrix m)
{
    if (m.empty())
        return 1.0;
    size_t n = m.size();
    if (!checkRectangle(m, true))
        throw std::invalid_argument("Non-Square Matrix");

    return gaussJordan(m, n);
}


Matrix invert(const Matrix& m)
{
    if (m.empty())
        return {};

    size_t n = m.size();
    if (!checkRectangle(m, true))
        throw std::invalid_argument("Can't Invert a non-Square Matrix");

    Matrix aug = m;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            aug[i].push_back(i==j);

    double d = gaussJordan(aug, n);
    if (std::abs(d) < 1e-15) 
        throw std::invalid_argument("Singular Matrix has no Inverse");

    Matrix res(n, Row(n));
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            res[i][j] = aug[i][n + j];
    return res;
}
