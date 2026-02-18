/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   utils.cpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/01/30 15:16:25 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/01/30 15:16:25 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Linear.hpp"


Row findMeans(const t_csv& csv)
{
    size_t rows = csv.data.size();
    size_t cols = csv.data[0].size();
    Row means(cols+1, 0.0);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
            means[j] += csv.data[i][j];
        means[cols] += csv.output[i];
    }
    for (size_t j = 0; j < cols+1; j++)
        means[j] /= rows;
    return means;
}


Row findVars(const t_csv& csv, const Row& means)
{
    size_t rows = csv.data.size();
    size_t cols = csv.data[0].size();
    Row vars(cols+1, 0.0);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
            vars[j] += std::pow(csv.data[i][j] - means[j], 2);
        vars[cols] += std::pow(csv.output[i] - means[cols], 2);
    }
    for (size_t j = 0; j < cols+1; j++)
        vars[j] /= rows - 1;
    return vars;
}


Row findStds(const Row& vars)
{
    Row stds;
    for (size_t j = 0; j < vars.size(); j++)
        stds.push_back(std::sqrt(vars[j]));
    return stds;
}


void printStats(const t_csv& csv)
{
    const int W_NAME = 25;
    const int W_NUM  = 15;

    std::cout << PURPLE "Samples: " GREEN << csv.data.size() << RESET "\n\n";
    
    Row means = findMeans(csv);
    Row vars  = findVars(csv, means);
    Row stds  = findStds(vars);
    size_t m = csv.headers.size();

    std::cout << YELLOW << std::left
              << std::setw(W_NAME) << ("Headers (" + std::to_string(m+1) + ")")
              << std::setw(W_NUM)  << "Mean"
              << std::setw(W_NUM)  << "Variance"
              << std::setw(W_NUM)  << "Std Dev"
              << RESET "\n";

    std::cout << std::string(W_NAME + 3 * W_NUM, '-') << "\n";
    for (size_t j = 0; j < m; j++)
    {
        std::cout << CYAN << std::left << std::setw(W_NAME) << csv.headers[j] << GREY;
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(W_NUM) << means[j]
                  << std::setw(W_NUM) << vars[j]
                  << std::setw(W_NUM) << stds[j] 
                  << RESET "\n";
    }
    std::cout << BLUE << std::left << std::setw(W_NAME) << csv.label << GREEN;
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(W_NUM) << means[m]
                  << std::setw(W_NUM) << vars[m]
                  << std::setw(W_NUM) << stds[m] 
                  << RESET "\n";
}


void normalize(t_csv& csv)
{
    csv.means = findMeans(csv);
    Row vars = findVars(csv, csv.means);
    csv.stds = findStds(vars);

    size_t rows = csv.data.size();
    size_t cols = csv.data[0].size();

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
            csv.data[i][j] = (csv.data[i][j] - csv.means[j]) / ((std::abs(csv.stds[j]) > 1e-9) ? csv.stds[j] : 1);
        csv.output[i] = (csv.output[i] - csv.means[cols]) / ((std::abs(csv.stds[cols]) > 1e-9) ? csv.stds[cols] : 1);
    }
}


void denormalizeRow(Row& r, double mean, double std)
{
    for (size_t i = 0; i < r.size(); i++)
        r[i] = mean + (r[i] * std);
}


void denormalizeData(t_csv& csv)
{
    size_t rows = csv.data.size();
    size_t cols = csv.data[0].size();

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
            csv.data[i][j] = csv.means[j] + (csv.data[i][j] * csv.stds[j]);
        csv.output[i] = csv.means[cols] + (csv.output[i] * csv.stds[cols]);
    }
}


void denormalizeWeights(const t_csv& csv, Row& w)
{
	size_t cols = csv.stds.size();
	double sum = 0;
	for (size_t i = 0; i < cols - 1; i++)
	{
		w[i] *= (std::abs(csv.stds[i]) > 1e-9) ? (csv.stds[cols - 1] / csv.stds[i]) : 0;
		sum += w[i] * csv.means[i];
	}
	w[cols-1] = csv.means[cols-1] + (w[cols-1] * csv.stds[cols-1]) - sum;
}


double rSqr(const Row& actual, const Row& pred)
{
	size_t n = actual.size();
	if (n != pred.size())
        throw std::invalid_argument("Difference in size");

	double mean = 0;
	for (size_t i = 0; i < n; i++)
		mean += actual[i];
	mean /= n;

	double sumRes = 0;
	double sumSqr = 0;
	for (size_t i = 0; i < n; i++)
	{
		sumRes += std::pow(actual[i] - pred[i], 2);
		sumSqr += std::pow(actual[i] - mean, 2);
	}
	return 1 - (sumRes / sumSqr);
}

