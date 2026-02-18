/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   predict.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: RuhanGi <mohammedruhan.goltay@kaust.edu    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/02/01 16:33:06 by RuhanGi           #+#    #+#             */
/*   Updated: 2026/02/01 16:33:06 by RuhanGi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "Linear.hpp"


Row makePreds(const Matrix& X, const Row& w, int m)
{
    if (X.empty())
        return {};
    size_t n = X.size();
    size_t cols = X[0].size();
    size_t new_cols = w.size();
    if (cols * m + 1 != new_cols)
        throw std::invalid_argument("Feature Mismatch");

    Row preds(n);
    std::for_each(std::execution::par, preds.begin(), preds.end(),
        [&](double& pred_val) {
            size_t i = &pred_val - &preds[0];
            pred_val = w[new_cols - 1];
            for (size_t j = 0; j < cols; j++)
            {
                double x = X[i][j];
                double val = x;
                for (int d = 0; d < m; d++)
                {
                    pred_val += w[j * m + d] * val;
                    val *= x;
                }
            }
        }
    );
    return preds;
}
