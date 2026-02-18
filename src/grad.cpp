/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   grad.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/01/31 16:57:14 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/01/31 16:57:14 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Linear.hpp"


Row closedForm(const Matrix& X, const Row& y, double lambda)
{
	size_t n = X.size();
	if (n != y.size())
        throw std::invalid_argument("I/O Mismatch");

	Matrix X_copy = X;

	for (Row& row : X_copy)
		row.push_back(1.0);
	Matrix Xt = transpose(X_copy);
	Matrix product = Xt * X_copy;
	size_t dim = product.size();
	if (lambda)
		for (size_t i = 0; i < dim - 1; i++)
			product[i][i] += lambda;
	Row w = invert(product) * (Xt * y);
	return w;
}


Row initRandom(size_t n)
{
	std::mt19937_64 gen(std::random_device{}());
	std::uniform_real_distribution<> dis(-0.1, 0.1);
	Row w(n);
	for (double& val : w)
		val = dis(gen);
	return w;
}


void setGradient(const Matrix& X, const Row& y, const Row&w, Row& gradients)
{
	size_t n = X.size();
    size_t m = X[0].size();
	double error;

	std::fill(gradients.begin(), gradients.end(), 0.0);
	for(size_t i = 0; i < n; i++)
	{
		error = w[m] - y[i];
		for(size_t j = 0; j < m; j++)
			error += w[j] * X[i][j];
		for(size_t j = 0; j < m; j++)
			gradients[j] += error * X[i][j];
		gradients[m] += error;
	}
}


double updateWeights(Row& w, Row& gradients, double lambda, size_t n)
{
	size_t m = w.size() - 1;
    double l2_norm = 0.0;
    double mean_grad;

	for(size_t j = 0; j < m; j++)
    {
        gradients[j] += lambda * w[j]; 
        mean_grad = gradients[j] / n;
        w[j] -= LR * mean_grad;
        l2_norm += mean_grad * mean_grad;
    }
	mean_grad = gradients[m] / n;
    w[m] -= LR * mean_grad;
    l2_norm += mean_grad * mean_grad;
	return l2_norm;
}


Row gradDescent(const Matrix& X, const Row& y, double lambda)
{
	if (X.empty())
		return {};
	size_t n = X.size();
	if (n != y.size())
		throw std::invalid_argument("I/O Mismatch");

	size_t m = X[0].size();
	Row w = initRandom(m + 1);
	Row gradients(m + 1);
	double l2_norm;
	size_t count = 0;
	do
	{
		if (count % 100 == 0)
			std::cout << GREY "Epoch [" << count << "/" << MAX_EPOCHS << "]\r" << std::flush;
		setGradient(X, y, w, gradients);
        l2_norm = updateWeights(w, gradients, lambda, n);
	} while (l2_norm > TOLERANCE * TOLERANCE && ++count < MAX_EPOCHS);
	std::cout << "Epoch [" << count << "/" << MAX_EPOCHS << "]" << RESET "\n";

	return w;
}


Row	doGradientDescent(t_csv& csv, double lambda)
{
	normalize(csv);
	Row w = gradDescent(csv.data, csv.output, lambda);
	denormalizeWeights(csv, w);
	denormalizeData(csv);
	return w;
}
