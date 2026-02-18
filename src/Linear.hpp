/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Wine.hpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/01/28 18:35:47 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/01/28 18:35:47 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#pragma once

#include <unordered_set>
#include <algorithm>
#include <execution>
#include <stdexcept>
#include <iterator>
#include <iostream>
#include <fstream>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <ctime>
#include <map>

#define GREY	"\x1B[30m"
#define RED		"\x1B[31m"
#define GREEN	"\x1B[32m"
#define YELLOW	"\x1B[33m"
#define BLUE	"\x1B[34m"
#define PURPLE	"\x1B[35m"
#define CYAN	"\x1B[36m"
#define RESET	"\x1B[0m"

#define DEGREE 3

#define SPLIT 0.2
#define LR 0.05
#define TOLERANCE 0.01
#define MAX_EPOCHS 10000
#define LAMBDA 20

#define OUTFILE "res.csv"

using Matrix = std::vector<std::vector<double>>;
using Row = std::vector<double>;

typedef struct s_csv
{
	std::vector<std::string>    headers;
	std::string                 label;
	Matrix                      data;
	Row                         output;
	Row                         means;
	Row                         stds;
}   t_csv;

t_csv parseCSV(std::string fil);
void printStats(const t_csv& csv);
void normalize(t_csv& csv);
void denormalizeData(t_csv& csv);
void denormalizeWeights(const t_csv& csv, Row& w);
double rSqr(const Row& actual, const Row& pred);

std::string smartFormat(double val);
std::ostream& operator<<(std::ostream& os, const Matrix& m);
std::ostream& operator<<(std::ostream& os, const Row& row);
Matrix operator*(const double k, const Matrix& m);
Matrix operator*(const Matrix& A, const Matrix& B);
Row operator+(const Row& A, const Row& B);
Row& operator+=(Row& A, const Row& B);
Row operator-(const Row& A, const Row& B);
Row& operator-=(Row& A, const Row& B);
Row operator*(const double k, const Row& r);
Row operator*(const Row& A, const Row& B);
Row& operator*=(Row& A, const Row& B);
Row operator*(const Matrix& X, const Row& r);

Matrix randomizeMatrix(size_t rows, size_t cols);
Matrix transpose(const Matrix& m);
Matrix invert(const Matrix& m);
double det(Matrix m);

Row makePreds(const Matrix& X, const Row& w, int m = 1);
Row	closedForm(const Matrix& X, const Row& y, double lambda = 0.0);
Row gradDescent(const Matrix& X, const Row& y, double lambda = 0.0);
Matrix makePolyData(const Matrix& X, int m);

Row	doGradientDescent(t_csv& csv, double lambda = 0.0);
Row	doPolynomial(t_csv& csv, int m, Row (*func)(const Matrix&, const Row&, double), double lambda = 0.0);
void exportPredictions(t_csv& csv, int degree);

Row	doInteract(t_csv& csv, Row (*func)(const Matrix&, const Row&, double), double lambda = 0.0);
Row predInteract(const Matrix& X, const Row& w);
void testInteract(t_csv& csv);

void calcResults(t_csv& csv);
