/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   overload.cpp                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/02/01 17:18:12 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/02/01 17:18:12 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Linear.hpp"

std::string smartFormat(double val) {
    if (std::abs(val) < 1e-9)
        return "0";
    
    if (std::abs(val) > 1e6 || std::abs(val) < 1e-4) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(1) << val;
        return oss.str();
    }
    if (std::abs(val - std::round(val)) < 1e-9)
        return std::to_string((long long) std::round(val));

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << val;
    std::string s = oss.str();
    s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    if (s.back() == '.')
        s.pop_back();
    return s;
}


std::ostream& operator<<(std::ostream& os, const Matrix& m)
{
    const int W_NUM  = 10;
    for (size_t i = 0; i < m.size(); i++)
    {
        os << "[";
        for (size_t j = 0; j < m[i].size(); j++)
            os << std::setw(W_NUM) << smartFormat(m[i][j]);
        os << "]\n";
    }
    return os;
}


std::ostream& operator<<(std::ostream& os, const Row& row)
{
    const int W_NUM  = 10;
    os << "[";
    for (size_t i = 0; i < row.size(); i++)
        os << std::setw(W_NUM) << smartFormat(row[i]);
    os << "]\n";
    return os;
}


Matrix operator*(const double k, const Matrix& m)
{
    if (m.empty())
        return {};

    Matrix res(m.size(), Row(m[0].size()));
    for (size_t i = 0; i < m.size(); i++)
        for (size_t j = 0; j < m[0].size(); j++)
            res[i][j] = k * m[i][j];
    return res;
}


Matrix operator*(const Matrix& A, const Matrix& B)
{
    if (A.empty() || B.empty())
        return {};
    if (A[0].size() != B.size())
        throw std::invalid_argument("Matrix Mismath Mid Multiplication");

    Matrix res(A.size(), Row(B[0].size(), 0.0));
    std::for_each(std::execution::par, res.begin(), res.end(),
        [&](Row& res_row) {
            size_t i = &res_row - &res[0];
            for (size_t m = 0; m < A[0].size(); m++)
                for (size_t j = 0; j < B[0].size(); j++)
                    res_row[j] += A[i][m] * B[m][j];
        }
    );
    return res;
}


Row operator+(const Row& A, const Row& B)
{
    if (A.empty() || B.empty())
        return {};
    size_t n = A.size();
    if (n != B.size())
        throw std::invalid_argument("Vector Mismath for Multiplication");

    Row res(n);
    for (size_t i = 0; i < n; i++)
        res[i] = A[i] + B[i];
    return res;
}


Row& operator+=(Row& A, const Row& B)
{
    if (A.empty() || B.empty())
        return A;
    size_t n = A.size();
    if (n != B.size())
        throw std::invalid_argument("Vector Mismath for Multiplication");

    for (size_t i = 0; i < n; i++)
        A[i] += B[i];
    return A;
}


Row operator-(const Row& A, const Row& B)
{
    if (A.empty() || B.empty())
        return {};
    size_t n = A.size();
    if (n != B.size())
        throw std::invalid_argument("Vector Mismath for Multiplication");

    Row res(n);
    for (size_t i = 0; i < n; i++)
        res[i] = A[i] - B[i];
    return res;
}


Row& operator-=(Row& A, const Row& B)
{
    if (A.empty() || B.empty())
        return A;
    size_t n = A.size();
    if (n != B.size())
        throw std::invalid_argument("Vector Mismath for Multiplication");

    for (size_t i = 0; i < n; i++)
        A[i] -= B[i];
    return A;
}


Row operator*(const double k, const Row& r)
{
    if (r.empty())
        return {};

    size_t n = r.size();
    Row res(n);
    for (size_t i = 0; i < n; i++)
        res[i] = k * r[i];
    return res;
}


Row operator*(const Row& A, const Row& B)
{
    if (A.empty() || B.empty())
        return {};
    size_t n = A.size();
    if (n != B.size())
        throw std::invalid_argument("Vector Mismath for Multiplication");

    Row res(n);
    for (size_t i = 0; i < n; i++)
        res[i] = A[i] * B[i];
    return res;
}


Row& operator*=(Row& A, const Row& B)
{
    if (A.empty() || B.empty())
        return A;
    size_t n = A.size();
    if (n != B.size())
        throw std::invalid_argument("Vector Mismath for Multiplication");

    for (size_t i = 0; i < n; i++)
        A[i] *= B[i];
    return A;
}


Row operator*(const Matrix& X, const Row& r)
{
    if (X.empty() || X[0].size() != r.size())
        throw std::invalid_argument("Dimension Mismatch: Matrix Cols != Vector Size");
    
    Row res(X.size(), 0.0);
    for (size_t i = 0; i < X.size(); i++)
        for (size_t j = 0; j < r.size(); j++)
            res[i] += X[i][j] * r[j];
    return res;
}
