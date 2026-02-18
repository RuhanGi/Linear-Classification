/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   basis.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/02/17 18:00:49 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/02/17 18:00:49 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Linear.hpp"


Matrix makePolyData(const Matrix& X, int m)
{
	size_t n = X.size();
	size_t cols = X[0].size();
	size_t new_cols = cols * m;

	Matrix phi(n, Row(new_cols));
	for (size_t i = 0; i < n; i++)
	{		
		for (size_t j = 0; j < cols; j++)
		{
			phi[i][j * m] = X[i][j];
			for (int d = 1; d < m; d++)
				phi[i][j * m + d] = phi[i][j * m + d - 1] * X[i][j];
		}
	}
	return phi;
}


std::pair<double, double> getRange(const Matrix& X) {
    double min_x = 1e9;
    double max_x = -1e9;
    
    for (const auto& row : X) {
        if (row[0] < min_x)
            min_x = row[0];
        if (row[0] > max_x)
            max_x = row[0];
    }
    return {min_x, max_x};
}

Matrix makeGaussian(const Matrix& X, int M) {
    if (X.empty())
        return {};
    size_t N = X.size();
    size_t D = X[0].size();

    std::vector<double> min_x(D), step(D);
    for (size_t j = 0; j < D; j++)
    {
        double local_min = 1e9, local_max = -1e9;
        for (size_t i = 0; i < N; i++) {
            if (X[i][j] < local_min) local_min = X[i][j];
            if (X[i][j] > local_max) local_max = X[i][j];
        }
        min_x[j] = local_min;
        step[j] = (local_max - local_min) / (M - 1);
    }

    Matrix phi;
    phi.reserve(N);
    for (const auto& row : X)
    {
        Row new_features;
        new_features.push_back(1.0); 
        for (size_t j = 0; j < D; j++) {
            double x = row[j];
            double s = step[j];
            for (int k = 0; k < M; k++)
            {
                double mu = min_x[j] + (k * s);
                double numerator = -std::pow(x - mu, 2);
                double denominator = 2 * s * s;
                new_features.push_back(std::exp(numerator / denominator));
            }
        }
        phi.push_back(new_features);
    }
    return phi;
}


Matrix makeSigmoid(const Matrix& X, int M) {
    if (X.empty()) 
        return {};
    size_t N = X.size();
    size_t D = X[0].size();

    std::vector<double> min_x(D), step(D);
    for (size_t j = 0; j < D; j++)
    {
        double local_min = 1e9, local_max = -1e9;
        for (size_t i = 0; i < N; i++) {
            if (X[i][j] < local_min) local_min = X[i][j];
            if (X[i][j] > local_max) local_max = X[i][j];
        }
        min_x[j] = local_min;
        step[j] = (local_max - local_min) / (M - 1);
    }

    Matrix phi;
    phi.reserve(N);
    for (const auto& row : X) {
        Row new_features;
        new_features.push_back(1.0);
        for (size_t j = 0; j < D; j++) {
            double x = row[j];
            double s = 1.0 / step[j];
            for (int k = 0; k < M; k++)
            {
                double mu = min_x[j] + (k * step[j]);
                double z = -s * (x - mu);
                double val = 1.0 / (1.0 + std::exp(z));
                new_features.push_back(val);
            }
        }
        phi.push_back(new_features);
    }
    return phi;
}

Row	doPolynomial(t_csv& csv, int m, Row (*func)(const Matrix&, const Row&, double), double lambda)
{
	if (csv.data.empty())
		return {};

	t_csv poly_csv;
	poly_csv.data = makePolyData(csv.data, m);
	poly_csv.output = csv.output;
	normalize(poly_csv);
	Row w = func(poly_csv.data, poly_csv.output, lambda);
	denormalizeWeights(poly_csv, w);

	return w;
}


Row	doGaussian(t_csv& csv, int m, Row (*func)(const Matrix&, const Row&, double), double lambda = 0.0)
{
	if (csv.data.empty())
		return {};

	t_csv poly_csv;
	poly_csv.data = makeGaussian(csv.data, m);
	poly_csv.output = csv.output;
	normalize(poly_csv);
	Row w = func(poly_csv.data, poly_csv.output, lambda);
	denormalizeWeights(poly_csv, w);

	return w;
}


Row	doSigmoid(t_csv& csv, int m, Row (*func)(const Matrix&, const Row&, double), double lambda = 0.0)
{
	if (csv.data.empty())
		return {};

	t_csv poly_csv;
	poly_csv.data = makeSigmoid(csv.data, m);
	poly_csv.output = csv.output;
	normalize(poly_csv);
	Row w = func(poly_csv.data, poly_csv.output, lambda);
	denormalizeWeights(poly_csv, w);

	return w;
}


void doBasis(t_csv& csv)
{
    if (csv.data.empty())
		return ;

    std::string choice;
    std::cout << "Which basis (Polynomial - P, Gaussian - G, Sigmoid - S): ";
    std::cin >> choice;

    Matrix X;
    if (choice[0] == 'P')
        X = makePolyData(csv.data, DEGREE);
    else if (choice[0] == 'G')
        X = makeGaussian(csv.data, DEGREE);
    else if (choice[0] == 'S')
        X = makeSigmoid(csv.data, DEGREE);
    else
        throw std::invalid_argument("Start with S, P, or G only!");

    t_csv poly_csv;
	poly_csv.data = X;
	poly_csv.output = csv.output;
	normalize(poly_csv);
	Row w = closedForm(poly_csv.data, poly_csv.output, LAMBDA);
	// Row w = gradDescent(poly_csv.data, poly_csv.output, LAMBDA);

    Row preds = makePreds(poly_csv.data, w);
    std::cout << "Rsqr = " << rSqr(poly_csv.output, preds) << "\n";
	// if (choice[0] == 'P')
    //     Matrix X = makePolyData(csv.data, DEGREE);
    // else if (choice[0] == 'G')
    //     Matrix X = makeGaussian(csv.data, DEGREE);
    // else if (choice[0] == 'S')
    //     Matrix X = makeSigmoid(csv.data, DEGREE);
    // else
    //     throw std::invalid_argument("Start with S, P, or G only!");
}
